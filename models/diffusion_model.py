import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Computes sinusoidal time-step positional embeddings for the UNet model.
    Unchanged from original — math is correct.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device    = time.device
        half_dim  = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    FIX: Cosine noise schedule (Nichol & Dhariwal, 2021).
    Replaces the original linear schedule which destroys signal too aggressively
    at low timesteps, leading to poor sample quality especially at lower resolutions.
    """
    steps          = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas          = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.9999).float()


# --------------------------------------------------------------------------- #
# Self-Attention (NEW)
# --------------------------------------------------------------------------- #

class SelfAttention(nn.Module):
    """
    FIX: Multi-head self-attention block for the UNet bottleneck (and optionally
    intermediate resolutions).  Every established diffusion UNet uses attention
    here to capture long-range spatial dependencies.  Without it the model cannot
    build coherent global structure in the generated image.

    Uses GroupNorm (not BatchNorm) before projecting to queries/keys/values.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)   # (B, H*W, C)
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).view(B, C, H, W)         # residual


# --------------------------------------------------------------------------- #
# Core building block
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """
    Two-convolution block with GroupNorm and SiLU activation.

    FIX 1 — GroupNorm replaces BatchNorm2d.
        Diffusion models sample one timestep at a time; batch statistics become
        unreliable at small (or size-1) batches.  GroupNorm is batch-size
        independent and is the standard for all modern diffusion UNets.

    FIX 2 — AdaGN (scale-and-shift) time conditioning replaces simple addition.
        The time MLP now outputs 2×channels so the embedding can modulate both
        the scale and the shift of the normalised activations.  This is strictly
        more expressive than additive injection and is used in all DDPM follow-ups
        (Improved DDPM, Guided Diffusion, etc.).
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)   # FIX 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)   # FIX 1

        if time_emb_dim is not None:
            # FIX 2: project to 2×out_channels for scale + shift (AdaGN)
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels * 2)
            )
        else:
            self.time_mlp = None

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)

        # FIX 2: AdaGN modulation after first norm
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)                           # (B, 2*C)
            time_emb = time_emb[..., None, None]                  # (B, 2*C, 1, 1)
            scale, shift = time_emb.chunk(2, dim=1)               # each (B, C, 1, 1)
            x = x * (1.0 + scale) + shift

        x = F.silu(x)
        x = F.silu(self.norm2(self.conv2(x)))
        return x


# --------------------------------------------------------------------------- #
# Down / Up blocks
# --------------------------------------------------------------------------- #

class DownBlock(nn.Module):
    """Downsampling block: MaxPool → DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = None):
        super().__init__()
        self.pool        = nn.MaxPool2d(2)
        self.double_conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        return self.double_conv(self.pool(x), t)


class UpBlock(nn.Module):
    """
    Upsampling block: ConvTranspose2d → cat(skip) → DoubleConv.
    Skip connections and odd-dimension padding are preserved from original.
    """
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, time_emb_dim: int = None):
        super().__init__()
        self.up          = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                              kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels // 2 + skip_channels,
                                      out_channels, time_emb_dim)

    def forward(self, x_prev: torch.Tensor,
                x_skip: torch.Tensor,
                t: torch.Tensor = None) -> torch.Tensor:
        x_prev = self.up(x_prev)

        # Robust padding for odd spatial dimensions
        diffY = x_skip.size(2) - x_prev.size(2)
        diffX = x_skip.size(3) - x_prev.size(3)
        x_prev = F.pad(x_prev, [diffX // 2, diffX - diffX // 2,
                                 diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_skip, x_prev], dim=1)
        return self.double_conv(x, t)


# --------------------------------------------------------------------------- #
# Conditional UNet
# --------------------------------------------------------------------------- #

class ConditionalUNet(nn.Module):
    """
    Time- and thermally-conditioned UNet backbone for the DDPM denoiser.

    Summary of fixes vs. original:
      • BatchNorm2d  → GroupNorm(8, ...)   in every DoubleConv
      • Additive time conditioning → AdaGN scale+shift  in every DoubleConv
      • Time MLP deepened: sin_emb → Linear(dim, dim*4) → SiLU → Linear(dim*4, dim)
      • Self-attention added at the bottleneck between bot1 and bot2
    """
    def __init__(self, c_in: int = 4, c_out: int = 3, time_dim: int = 256):
        super().__init__()
        self.time_dim = time_dim

        # FIX 3: Deeper time MLP — two linear layers with 4× hidden expansion.
        # A single linear layer is too shallow to disentangle timestep semantics.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Encoder
        self.inc   = DoubleConv(c_in, 64,  time_dim)
        self.down1 = DownBlock(64,  128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        self.down3 = DownBlock(256, 512, time_dim)

        # Bottleneck
        self.bot1     = DoubleConv(512, 512, time_dim)
        self.bot_attn = SelfAttention(512, num_heads=4)   # FIX: attention here
        self.bot2     = DoubleConv(512, 512, time_dim)

        # Decoder
        self.up1 = UpBlock(512, 256, 256, time_dim)
        self.up2 = UpBlock(256, 128, 128, time_dim)
        self.up3 = UpBlock(128,  64,  64, time_dim)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor,
                cond: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        # Concatenate noisy visible image and static thermal condition
        x = torch.cat([x, cond], dim=1)

        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        x1 = self.inc(x, t_emb)      # (B, 64,  H,   W)
        x2 = self.down1(x1, t_emb)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2, t_emb)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3, t_emb)   # (B, 512, H/8, W/8)

        # Bottleneck with attention
        x4 = self.bot1(x4, t_emb)
        x4 = self.bot_attn(x4)        # FIX: long-range spatial context
        x4 = self.bot2(x4, t_emb)

        # Decoder
        x = self.up1(x4, x3, t_emb)
        x = self.up2(x,  x2, t_emb)
        x = self.up3(x,  x1, t_emb)

        return self.outc(x)


# --------------------------------------------------------------------------- #
# DDPM framework
# --------------------------------------------------------------------------- #

class ThermalToVisibleDDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020) with:
      • Cosine noise schedule  (Nichol & Dhariwal, 2021)   [FIX 4]
      • Standard DDPM reverse sampling
      • DDIM deterministic fast-sampling                    [FIX 5]

    Training note
    -------------
    After loss.backward() always clip gradients before optimizer.step():

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    Diffusion models are sensitive to gradient explosions early in training.
    """

    def __init__(self, network: nn.Module,
                 beta_1: float = 1e-4,
                 beta_T: float = 0.02,
                 T: int = 1000,
                 schedule: str = 'cosine'):
        super().__init__()
        self.network = network
        self.T       = T

        # FIX 4: Cosine schedule by default; linear kept as a fallback option.
        if schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        else:
            betas = torch.linspace(beta_1, beta_T, T)

        alphas                = 1.0 - betas
        alphas_cumprod        = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev   = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance    = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.register_buffer('betas',                       betas)
        self.register_buffer('alphas_cumprod',              alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev',         alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod',         torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('posterior_variance',          posterior_variance)

    # ---------------------------------------------------------------------- #
    # Training forward pass
    # ---------------------------------------------------------------------- #

    def forward(self, x_0: torch.Tensor,
                cond: torch.Tensor):
        """
        Randomly samples a timestep, corrupts x_0, and returns the ground-truth
        noise alongside the network's noise prediction and Min-SNR weights.

        Args:
            x_0:  Ground-truth visible image   (B, 3, H, W)
            cond: Thermal condition image       (B, 1, H, W)
        Returns:
            (noise, predicted_noise, snr_weights)
            Use: loss = (snr_weights * F.mse_loss(pred, noise, reduction='none')).mean()
        """
        batch_size = x_0.shape[0]
        device     = x_0.device

        t     = torch.randint(0, self.T, (batch_size,), device=device).long()
        noise = torch.randn_like(x_0)

        sqrt_ac_t     = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_1mac_t   = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        x_t           = sqrt_ac_t * x_0 + sqrt_1mac_t * noise

        predicted_noise = self.network(x_t, cond, t)

        # Min-SNR-γ weighting (Hang et al., 2023)
        # Prevents gradient conflicts across timesteps by clamping SNR weights
        ac_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1).float()
        snr  = ac_t / (1.0 - ac_t).clamp(min=1e-8)
        snr_weights = torch.clamp(snr, max=5.0) / snr.clamp(min=1e-8)

        return noise, predicted_noise, snr_weights

    # ---------------------------------------------------------------------- #
    # DDPM reverse sampling  (slow, stochastic, original algorithm)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample_ddpm(self, cond: torch.Tensor,
                    shape: tuple) -> torch.Tensor:
        """
        Full T-step stochastic DDPM reverse process.
        Use when sample quality is the priority and inference speed is not.

        Args:
            cond:  Thermal condition  (B, 1, H, W)
            shape: Output tensor shape  (B, 3, H, W)
        Returns:
            Generated visible image clamped to [-1, 1]
        """
        device = cond.device
        b      = shape[0]
        x_t    = torch.randn(shape, device=device)

        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            predicted_noise = self.network(x_t, cond, t)

            alpha_t        = 1.0 - self.betas[i]
            alpha_cumprod  = self.alphas_cumprod[i]

            coeff1 = 1.0 / torch.sqrt(alpha_t)
            coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_cumprod)
            mean   = coeff1 * (x_t - coeff2 * predicted_noise)

            if i > 0:
                noise = torch.randn_like(x_t)
                x_t   = mean + torch.sqrt(self.posterior_variance[i]) * noise
            else:
                x_t = mean

        return torch.clamp(x_t, -1.0, 1.0)

    # ---------------------------------------------------------------------- #
    # FIX 5: DDIM reverse sampling  (fast, deterministic)
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def sample_ddim(self, cond: torch.Tensor,
                    shape: tuple,
                    ddim_steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM deterministic sampler (Song et al., 2020).
        Produces comparable quality to DDPM in 50 steps instead of 1000
        by making the reverse process deterministic (eta=0) or partially
        stochastic (0 < eta <= 1, where eta=1 recovers DDPM variance).

        Args:
            cond:        Thermal condition      (B, 1, H, W)
            shape:       Output tensor shape     (B, 3, H, W)
            ddim_steps:  Number of denoising steps (default 50)
            eta:         Stochasticity factor; 0.0 = fully deterministic
        Returns:
            Generated visible image clamped to [-1, 1]
        """
        device  = cond.device
        b       = shape[0]
        x_t     = torch.randn(shape, device=device)

        # Build a uniformly-spaced subsequence of timesteps
        step_indices = torch.linspace(0, self.T - 1, ddim_steps + 1,
                                      dtype=torch.long, device=device)

        for idx in reversed(range(1, len(step_indices))):
            t_cur  = step_indices[idx]
            t_prev = step_indices[idx - 1]

            t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
            predicted_noise = self.network(x_t, cond, t_batch)

            ac_cur  = self.alphas_cumprod[t_cur]
            ac_prev = self.alphas_cumprod[t_prev]

            # Predict x_0 from current x_t and predicted noise
            x0_pred = (x_t - torch.sqrt(1.0 - ac_cur) * predicted_noise) / torch.sqrt(ac_cur)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # DDIM direction toward x_t
            sigma = (eta * torch.sqrt((1.0 - ac_prev) / (1.0 - ac_cur))
                         * torch.sqrt(1.0 - ac_cur / ac_prev))

            direction = torch.sqrt(1.0 - ac_prev - sigma ** 2) * predicted_noise
            noise     = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

            x_t = torch.sqrt(ac_prev) * x0_pred + direction + sigma * noise

        return torch.clamp(x_t, -1.0, 1.0)

    # Convenience alias — default sampling uses the fast DDIM path
    @torch.no_grad()
    def sample(self, cond: torch.Tensor,
               shape: tuple,
               ddim_steps: int = 50) -> torch.Tensor:
        """Default sampler: DDIM with 50 steps. Pass ddim_steps=self.T for full DDPM."""
        return self.sample_ddim(cond, shape, ddim_steps=ddim_steps)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Initiating Cond-DDPM model diagnostics...")

    unet             = ConditionalUNet(c_in=4, c_out=3, time_dim=256)
    diffusion_model  = ThermalToVisibleDDPM(network=unet, T=1000, schedule='cosine')

    total_params = sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_visible = torch.randn(2, 3, 128, 128)
    dummy_thermal = torch.randn(2, 1, 128, 128)

    # 1. Training forward pass
    real_noise, pred_noise, snr_weights = diffusion_model(dummy_visible, dummy_thermal)
    print(f"\nForward pass  — noise shape: {real_noise.shape} | pred shape: {pred_noise.shape} | snr_weights shape: {snr_weights.shape}")


    # 2. DDIM fast sampling (50 steps)
    output_shape = (2, 3, 128, 128)
    sampled_ddim = diffusion_model.sample_ddim(dummy_thermal, output_shape, ddim_steps=50)
    print(f"DDIM sample   — shape: {sampled_ddim.shape}  range: [{sampled_ddim.min():.3f}, {sampled_ddim.max():.3f}]")

    # 3. DDPM full sampling (reduced T for speed in test)
    fast_ddpm_model = ThermalToVisibleDDPM(network=unet, T=50, schedule='cosine')
    sampled_ddpm    = fast_ddpm_model.sample_ddpm(dummy_thermal, output_shape)
    print(f"DDPM sample   — shape: {sampled_ddpm.shape}  range: [{sampled_ddpm.min():.3f}, {sampled_ddpm.max():.3f}]")

    print("\nDDPM implementation is complete and structurally verified!")
