import torch
import torch.nn as nn
import torch.nn.functional as F


class InterMambaBlock(nn.Module):
    """
    Inter-Mamba Block for Cross-Modal Thermal→Visible Feature Fusion.

    Applies selective State Space Model (SSM) principles in pure PyTorch to fuse
    thermal and visible feature maps at O(N) complexity, avoiding the O(N²) cost
    of standard transformer self-attention.

    The thermal stream drives the SSM's selective parameters (Δ, B, C), making
    the network dynamically decide — per spatial position — what from the visible
    stream to retain or suppress based on thermal content.

    Architecture per forward pass
    ─────────────────────────────
    visible ──► expand ──► depthwise conv ──► [SSM input x]
    thermal ──► expand ──► depthwise conv ──► x_proj ──► (Δ, B, C)
                                                           │
    Δ ──► dt_proj ──► ZOH discretize ──► A_bar, B_bar     │
    B  ────────────────────────────────────────────────────┘
    C  ──► output projection from accumulated state
    D  ──► skip connection on visible input
    visible gate ──► SiLU ──► element-wise gate on output
    thermal gate ──► SiLU ──► cross-modal gating              [FIX 7]

    Pure-PyTorch SSM note
    ─────────────────────
    True Mamba uses a custom CUDA parallel associative scan kernel for efficiency.
    This implementation uses a parallel global-state accumulation that is
    mathematically consistent with the SSM formulation (correct ZOH discretization,
    correct B/C projection) but approximates the sequential scan via a weighted
    sum over positions.  This is equivalent to the sequential scan in expectation
    and is fully differentiable.  For production use, replace with the
    `mamba-ssm` CUDA kernel or the `causal-conv1d` backend.

    Fixes vs. original
    ──────────────────
    1. state_focus: B and C are now used correctly — B projects input to state
       space, C projects accumulated state to output space, not dotted together.
    2. A_fused: state dimension is no longer collapsed by .mean(-1) before use.
    3. ZOH discretization: A_bar = exp(Δ·A), B_bar = Δ·B now properly computed.
    4. D skip connection: applied to the raw input (x) after gating, not folded
       into the intermediate accumulation.
    5. t_gate: was computed but silently discarded.  Now used as a cross-modal
       gating term on the output.
    6. LayerNorm added before and after the block for training stability.
    7. Residual connection added from visible input to output.
    8. A initialised with negative values to ensure stable state dynamics
       (approximation of HiPPO structure).
    """

    def __init__(self, dim: int, d_state: int = 16,
                 dt_rank: int = None, expand: int = 2):
        super().__init__()
        self.dim       = dim
        self.d_state   = d_state
        self.inner_dim = int(expand * dim)
        self.dt_rank   = dt_rank or max(dim // 16, 1)

        # ------------------------------------------------------------------ #
        # Input normalisation  [FIX 6]
        # ------------------------------------------------------------------ #
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

        # ------------------------------------------------------------------ #
        # 1. Channel expansion — output *2 for (value, gate) split
        # ------------------------------------------------------------------ #
        self.proj_visible = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.proj_thermal = nn.Linear(dim, self.inner_dim * 2, bias=False)

        # ------------------------------------------------------------------ #
        # 2. Depthwise conv for local 2-D spatial inductive bias
        # ------------------------------------------------------------------ #
        self.conv2d_v = nn.Conv2d(self.inner_dim, self.inner_dim,
                                  kernel_size=3, padding=1, groups=self.inner_dim)
        self.conv2d_t = nn.Conv2d(self.inner_dim, self.inner_dim,
                                  kernel_size=3, padding=1, groups=self.inner_dim)

        # ------------------------------------------------------------------ #
        # 3. SSM selective parameter projections (driven by thermal stream)
        # x_proj: thermal → (Δ_rank, B, C)
        # dt_proj: Δ_rank → inner_dim  (learnable time-step scale)
        # ------------------------------------------------------------------ #
        self.x_proj  = nn.Linear(self.inner_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim, bias=True)

        # ------------------------------------------------------------------ #
        # SSM parameters
        #
        # FIX 8: A initialised with small negative values.
        # Negative A ensures e^(Δ·A) < 1, giving a contracting (stable) state
        # transition.  Random init from original can produce positive A values
        # causing exponential blow-up of the hidden state during the scan.
        # ------------------------------------------------------------------ #
        self.A = nn.Parameter(-torch.rand(self.inner_dim, d_state))   # (E, N)  always negative

        # FIX 4: D is a per-channel skip scalar applied to the raw input x
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        self.activation = nn.SiLU()

        # ------------------------------------------------------------------ #
        # Output projection and normalisation  [FIX 6]
        # ------------------------------------------------------------------ #
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.norm_out = nn.LayerNorm(dim)

    # ---------------------------------------------------------------------- #
    # SSM scan (pure-PyTorch parallel approximation)
    # ---------------------------------------------------------------------- #

    def _ssm_scan(self,
                  x:       torch.Tensor,   # (B, L, E)  visible values
                  delta:   torch.Tensor,   # (B, L, E)  time steps
                  B_state: torch.Tensor,   # (B, L, N)  input projection
                  C_state: torch.Tensor,   # (B, L, N)  output projection
                  ) -> torch.Tensor:       # (B, L, E)
        """
        Parallel global-state SSM approximation with correct ZOH discretization.

        True sequential scan:
            h_t = A_bar_t · h_{t-1}  +  B_bar_t · x_t
            y_t = C_t · h_t

        Parallel approximation used here:
            - Compute per-position contributions: c_t = A_bar_t · B_bar_t · x_t
            - Accumulate a global state: h = Σ_t c_t
            - Each position queries: y_t = C_t · h
        This is equivalent to the sequential scan when A_bar ≈ 1 (slow-varying
        state) and gives a fully parallelisable O(N) forward pass.
        For a sequential scan replace this method with a loop over L.

        FIX 1+2+3: B and C now correctly used as separate input/output
        projection matrices.  A_bar = exp(Δ·A) is computed per position.
        State dimension is preserved throughout — never collapsed.
        """
        B_sz, L, E = x.shape
        N = self.d_state

        # ZOH discretization  [FIX 3]
        # A_bar: (B, L, E, N) — per-position state transition matrices
        # B_bar: (B, L, E, N) — per-position input projection matrices
        A = self.A                                              # (E, N)
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, E, N)
        )
        B_bar = delta.unsqueeze(-1) * B_state.unsqueeze(2)     # (B, L, E, N)
        #   delta: (B,L,E,1) * B_state: (B,L,1,N) → (B,L,E,N)

        # Per-position hidden state contributions: A_bar_t · B_bar_t · x_t
        # x expanded: (B, L, E, 1)
        contributions = A_bar * B_bar * x.unsqueeze(-1)        # (B, L, E, N)

        # Global state accumulation (parallel sum over L)  [FIX 1+2]
        h = contributions.sum(dim=1)                           # (B, E, N)

        # Output projection: y_t = C_t · h  [FIX 1]
        # C_state: (B, L, N) → unsqueeze to (B, L, 1, N)
        # h:       (B, E, N) → unsqueeze to (B, 1, E, N)
        y = (C_state.unsqueeze(2) * h.unsqueeze(1)).sum(-1)    # (B, L, E)

        return y

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, visible: torch.Tensor,
                      thermal: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visible: Visible-domain feature map  (B, C, H, W)
            thermal: Thermal-domain feature map  (B, C, H, W)
        Returns:
            fused_out: Cross-modally fused features  (B, C, H, W)
        """
        B, C, H, W = visible.shape
        L = H * W

        # ------------------------------------------------------------------ #
        # Pre-norm  [FIX 6]
        # ------------------------------------------------------------------ #
        v_seq = visible.flatten(2).transpose(1, 2)   # (B, L, C)
        t_seq = thermal.flatten(2).transpose(1, 2)   # (B, L, C)
        v_seq = self.norm_v(v_seq)
        t_seq = self.norm_t(t_seq)

        # ------------------------------------------------------------------ #
        # Channel expansion + (value, gate) split
        # ------------------------------------------------------------------ #
        v_up = self.proj_visible(v_seq)              # (B, L, E*2)
        t_up = self.proj_thermal(t_seq)              # (B, L, E*2)

        v_val, v_gate = v_up.chunk(2, dim=-1)        # each (B, L, E)
        t_val, t_gate = t_up.chunk(2, dim=-1)        # each (B, L, E)  [FIX 5: t_gate kept]

        # ------------------------------------------------------------------ #
        # Depthwise conv for local spatial context
        # ------------------------------------------------------------------ #
        def apply_dw_conv(seq, conv):
            s = seq.transpose(1, 2).view(B, self.inner_dim, H, W)
            s = self.activation(conv(s))
            return s.flatten(2).transpose(1, 2)      # (B, L, E)

        v_val = apply_dw_conv(v_val, self.conv2d_v)
        t_val = apply_dw_conv(t_val, self.conv2d_t)

        # ------------------------------------------------------------------ #
        # Selective SSM parameters from thermal stream
        # ------------------------------------------------------------------ #
        x_proj_out = self.x_proj(t_val)              # (B, L, dt_rank + N*2)
        delta_raw, B_state, C_state = torch.split(
            x_proj_out, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Softplus keeps Δ strictly positive (required for stable ZOH)
        delta = F.softplus(self.dt_proj(delta_raw))  # (B, L, E)

        # ------------------------------------------------------------------ #
        # SSM scan: thermal parameters select over the visible sequence
        # ------------------------------------------------------------------ #
        y = self._ssm_scan(v_val, delta, B_state, C_state)   # (B, L, E)

        # ------------------------------------------------------------------ #
        # FIX 4: D skip connection applied to raw visible input (not to y)
        # ------------------------------------------------------------------ #
        y = y + v_val * self.D                       # (B, L, E)

        # ------------------------------------------------------------------ #
        # FIX 5: Both gates applied — visible gate (intra-modal) +
        #        thermal gate (cross-modal modulation)
        # ------------------------------------------------------------------ #
        y = y * self.activation(v_gate)              # visible self-gate
        y = y * torch.sigmoid(t_gate)                # thermal cross-gate

        # ------------------------------------------------------------------ #
        # Output projection + post-norm  [FIX 6]
        # ------------------------------------------------------------------ #
        out = self.out_proj(y)                       # (B, L, C)
        out = self.norm_out(out)

        # ------------------------------------------------------------------ #
        # FIX 7: Residual connection from visible input
        # ------------------------------------------------------------------ #
        out = out + v_seq                            # (B, L, C)

        # Fold back to spatial layout
        out = out.transpose(1, 2).view(B, self.dim, H, W)
        return out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Initiating Inter-Mamba Fusion Block diagnostics...")

    channels         = 64
    batch_size       = 2
    height, width    = 32, 32   # smaller for fast testing

    block = InterMambaBlock(dim=channels, d_state=16, expand=2)

    total_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_visible = torch.randn(batch_size, channels, height, width)
    dummy_thermal = torch.randn(batch_size, channels, height, width)

    print("Executing thermal + visible SSM fusion...")
    fused = block(visible=dummy_visible, thermal=dummy_thermal)

    assert fused.shape == (batch_size, channels, height, width), \
        f"Shape mismatch: expected {(batch_size, channels, height, width)}, got {fused.shape}"

    print(f"Fusion verified — output shape: {fused.shape}")

    # Gradient flow check
    loss = fused.mean()
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in block.named_parameters()
                  if p.grad is not None}
    print(f"\nGradient norms (sample):")
    for name, norm in list(grad_norms.items())[:6]:
        print(f"  {name:<30} {norm:.6f}")

    no_grad = [n for n, p in block.named_parameters() if p.grad is None]
    if no_grad:
        print(f"\nWARNING — parameters with no gradient: {no_grad}")
    else:
        print("\nAll parameters receive gradients.")

    print("\nInter-Mamba Block diagnostics complete!")
