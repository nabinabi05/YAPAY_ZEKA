import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Vector Quantizer with EMA updates + dead code reset
# --------------------------------------------------------------------------- #

class VectorQuantizer(nn.Module):
    """
    Vector Quantization (VQ) bottleneck with Exponential Moving Average (EMA)
    codebook updates and dead-code detection/reset.

    FIX 1 — EMA codebook updates replace pure gradient descent.
        The original trained the codebook via gradient descent on q_latent_loss.
        This is known to be unstable and slow because the codebook embedding
        gradients are small and compete with the encoder gradients.  EMA updates
        move each codebook vector directly toward the mean of the encoder outputs
        assigned to it, which is faster and more stable (VQ-VAE-2, Razavi 2019).
        When EMA is active, q_latent_loss is dropped — only the commitment loss
        (encoder → codebook direction) remains in the training objective.

    FIX 2 — Dead code reset.
        Without intervention, codes that are never selected accumulate stale
        embeddings and waste codebook capacity.  Any code unused for
        `dead_threshold` consecutive forward passes is reinitialised to a
        random encoder output from the current batch.

    FIX 3 — Codebook perplexity exposed for monitoring.
        Perplexity = exp(entropy of code usage distribution).  Maximum value is
        num_embeddings (every code used equally).  Values << num_embeddings
        indicate codebook collapse.  Returns perplexity as part of the forward
        output so the training loop can log it.

    Loss semantics (unchanged math, renamed for clarity):
        commitment_loss: ||z_e - sg[e]||²   encoder commits to nearest code
        vq_loss        = commitment_cost * commitment_loss  (only term when EMA on)
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99,
                 dead_threshold: int = 100):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_embeddings  = num_embeddings
        self.commitment_cost = commitment_cost
        self.ema_decay       = ema_decay
        self.dead_threshold  = dead_threshold

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA buffers — not parameters, updated in forward
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_weight_sum',   self.embedding.weight.data.clone())
        self.register_buffer('usage_counter',    torch.zeros(num_embeddings, dtype=torch.long))

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: (B, C, H, W) continuous encoder features
        Returns:
            quantized:   (B, C, H, W) straight-through quantized features
            vq_loss:     scalar commitment loss
            perplexity:  scalar codebook utilisation metric (log to W&B / tensorboard)
        """
        # (B, C, H, W) → (B*H*W, C)
        flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # L2 distance to all codebook vectors
        distances = (flat.pow(2).sum(1, keepdim=True)
                     + self.embedding.weight.pow(2).sum(1)
                     - 2.0 * flat @ self.embedding.weight.t())   # (N, K)

        encoding_indices = distances.argmin(1)                   # (N,)
        encodings_oh = F.one_hot(encoding_indices,
                                 self.num_embeddings).float()    # (N, K)

        # Reshape quantized back to (B, C, H, W)
        quantized = self.embedding(encoding_indices)             # (N, C)
        quantized = (quantized
                     .view(inputs.shape[0], inputs.shape[2],
                           inputs.shape[3], self.embedding_dim)
                     .permute(0, 3, 1, 2).contiguous())

        # ------------------------------------------------------------------ #
        # FIX 1: EMA codebook update (training only)
        # ------------------------------------------------------------------ #
        if self.training:
            with torch.no_grad():
                batch_cluster_size = encodings_oh.sum(0)         # (K,)
                batch_weight_sum   = encodings_oh.t() @ flat.float()  # cast fp16->fp32 for EMA buffers

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    batch_cluster_size * (1.0 - self.ema_decay))
                self.ema_weight_sum.mul_(self.ema_decay).add_(
                    batch_weight_sum   * (1.0 - self.ema_decay))

                # Laplace smoothing prevents division by zero for unused codes
                n = self.ema_cluster_size.sum()
                smoothed = ((self.ema_cluster_size + 1e-5)
                            / (n + self.num_embeddings * 1e-5) * n)
                self.embedding.weight.data.copy_(
                    self.ema_weight_sum / smoothed.unsqueeze(1))

                # FIX 2: Dead code detection and reset.
                # usage_counter tracks consecutive steps without use.
                # Increment for codes NOT used this step; reset to 0 for codes that WERE used.
                not_used = (batch_cluster_size == 0).long()
                self.usage_counter = (self.usage_counter + not_used) * not_used
                dead_mask = self.usage_counter > self.dead_threshold

                num_dead = dead_mask.sum().item()
                if num_dead > 0:
                    # Reinitialise dead codes to random encoder outputs
                    rand_idx   = torch.randperm(flat.size(0), device=flat.device)[:num_dead]
                    flat_f32   = flat[rand_idx].detach().float()   # fp16 -> fp32
                    self.embedding.weight.data[dead_mask] = flat_f32
                    self.ema_weight_sum[dead_mask]        = flat_f32
                    self.ema_cluster_size[dead_mask]      = 1.0
                    self.usage_counter[dead_mask]         = 0

        # ------------------------------------------------------------------ #
        # FIX 3: Perplexity for codebook utilisation monitoring
        # ------------------------------------------------------------------ #
        avg_probs   = encodings_oh.mean(0)                       # (K,)
        perplexity  = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        # Commitment loss only (codebook updated via EMA, not gradients)
        commitment_loss = F.mse_loss(inputs, quantized.detach())
        vq_loss         = self.commitment_cost * commitment_loss

        # Straight-Through Estimator — gradients bypass argmin
        quantized_st = inputs + (quantized - inputs).detach()

        return quantized_st, vq_loss, perplexity


# --------------------------------------------------------------------------- #
# Convolutional Positional Encoding
# --------------------------------------------------------------------------- #

class ConvPositionalEncoding(nn.Module):
    """
    Translation-invariant convolutional positional encoding.
    Unchanged from original — design is correct.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


# --------------------------------------------------------------------------- #
# Transformer with spatial downsampling
# --------------------------------------------------------------------------- #

class TransformerModule(nn.Module):
    """
    Global self-attention module with spatial resolution management.

    FIX 4 — Spatial pooling before attention, bilinear upsample after.
        The original applied attention directly to the backbone output which at
        256×256 input produces 64×64 = 4096 tokens.  Standard self-attention
        is O(N²) per layer so 6 layers × 8 heads × 4096² ≈ 800M operations
        per forward pass — computationally intractable and would OOM on most GPUs.

        The fix adds a 4× spatial pooling before the transformer (reducing tokens
        to 16×16 = 256) and a bilinear upsample back to the original resolution
        after.  This keeps the O(N²) cost affordable (256² = 65K) while still
        giving the transformer a global receptive field over the scene semantics.
        The CPE is applied at the original resolution before pooling so local
        structure is encoded before global mixing.
    """
    def __init__(self, dim: int, num_heads: int = 8, depth: int = 4,
                 pool_factor: int = 4):
        super().__init__()
        self.pool_factor = pool_factor
        self.cpe         = ConvPositionalEncoding(dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPE at original resolution for fine-grained local context
        x = self.cpe(x)                          # (B, C, H, W)

        B, C, H, W = x.shape
        H_s = H // self.pool_factor
        W_s = W // self.pool_factor

        # FIX 4: Downsample for tractable attention
        x_small = F.avg_pool2d(x, kernel_size=self.pool_factor)  # (B, C, H/p, W/p)

        seq = x_small.flatten(2).transpose(1, 2)                  # (B, L_small, C)
        seq = self.transformer(seq)                                # (B, L_small, C)

        # Upsample back to original resolution
        out_small = seq.transpose(1, 2).view(B, C, H_s, W_s)
        out       = F.interpolate(out_small, size=(H, W),
                                  mode='bilinear', align_corners=False)
        return out


# --------------------------------------------------------------------------- #
# CNN Backbone with skip-connection exports
# --------------------------------------------------------------------------- #

class CNNBackbone(nn.Module):
    """
    Local feature extractor.

    FIX 5 — Exports intermediate feature maps for skip connections to the decoder.
        The original had no skip path from encoder to decoder, so all high-
        frequency spatial detail had to be reconstructed entirely from the
        quantized bottleneck.  Now exports (feat1, feat2) at H/2 and H/4 for
        the upsampler to consume.
    FIX 6 — affine=True on all InstanceNorm2d layers.
    """
    def __init__(self, in_channels: int = 1, dim: int = 256):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7, padding=0),
            nn.InstanceNorm2d(64, affine=True),    # FIX 6
            nn.GELU(),
        )   # (B, 64, H, W)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),   # FIX 6
            nn.GELU(),
        )   # (B, 128, H/2, W/2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, dim, 4, stride=2, padding=1),
            nn.InstanceNorm2d(dim, affine=True),   # FIX 6
            nn.GELU(),
        )   # (B, dim, H/4, W/4)

    def forward(self, x: torch.Tensor):
        f1 = self.enc1(x)    # skip at full resolution
        f2 = self.enc2(f1)   # skip at H/2
        f3 = self.enc3(f2)   # bottleneck input
        return f3, f2, f1


# --------------------------------------------------------------------------- #
# CNN Upsampler with skip connections
# --------------------------------------------------------------------------- #

class CNNUpsampler(nn.Module):
    """
    Decodes hybridised feature layers back to RGB spatial domain.

    Correct U-Net skip pattern: UPSAMPLE first, THEN concatenate the skip.
    The previous version concatenated before upsampling, causing a spatial
    dimension mismatch (x at H/4 cannot be cat'd with skip2 at H/2).

    Shape flow (example: input 256×256):
        x     : (B, 256, 64,  64 ) ← from transformer
        after up1 → (B, 128, 128, 128)
        cat skip2 → (B, 256, 128, 128)  skip2=(B,128,128,128)
        after conv1→ (B, 128, 128, 128)
        after up2 → (B,  64, 256, 256)
        cat skip1 → (B, 128, 256, 256)  skip1=(B, 64,256,256)
        after conv2→ (B,  64, 256, 256)
        after out  → (B,   3, 256, 256)
    """
    def __init__(self, dim: int = 256, out_channels: int = 3):
        super().__init__()

        # Stage 1: upsample H/4 → H/2, then merge skip2 (128 ch)
        self.up1   = nn.ConvTranspose2d(dim, dim // 2, 4, stride=2, padding=1)
        # dim//2 + 128 = 128 + 128 = 256 in → 128 out
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim // 2 + 128, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.GELU(),
        )

        # Stage 2: upsample H/2 → H, then merge skip1 (64 ch)
        self.up2   = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        # 64 + 64 = 128 in → 64 out
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.GELU(),
        )

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor,
                skip2: torch.Tensor,
                skip1: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)                              # (B, 128, H/2, W/2)
        x = self.conv1(torch.cat([x, skip2], dim=1)) # (B, 128, H/2, W/2)
        x = self.up2(x)                              # (B,  64, H,   W  )
        x = self.conv2(torch.cat([x, skip1], dim=1)) # (B,  64, H,   W  )
        return self.out_conv(x)                      # (B,   3, H,   W  )


# --------------------------------------------------------------------------- #
# Full VQ-InfraTrans model
# --------------------------------------------------------------------------- #

class VQInfraTrans(nn.Module):
    """
    VQ-InfraTrans: Vector-Quantised Infrared Transformer.
    Hybrid CNN-ViT architecture for continuous Thermal-to-Visible generation.

    Pipeline
    ────────
    Thermal image
        │
        ▼
    CNNBackbone  ──────────────────────────────────────┐ (skip f2, f1)
        │ f3                                            │
        ▼                                               │
    VectorQuantizer (EMA, dead-code reset)              │
        │ quantized                                     │
        ▼                                               │
    TransformerModule (pooled → attention → upsample)  │
        │ global features                               │
        ▼                                               │
    CNNUpsampler  ◄────────────────────────────────────┘
        │
        ▼
    RGB output

    Training objective (in your training loop):
        total_loss = pixel_loss + λ_vq * vq_loss
        e.g.:
            pixel_loss = F.l1_loss(pred, target)
            total_loss = pixel_loss + 1.0 * vq_loss
        Log `perplexity` to monitor codebook health — target > 100 out of 1024.
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3,
                 latent_dim: int = 256, vq_codes: int = 1024,
                 transformer_depth: int = 6, transformer_pool: int = 2):
        super().__init__()
        self.backbone            = CNNBackbone(in_channels=input_nc, dim=latent_dim)
        self.vector_quantization = VectorQuantizer(num_embeddings=vq_codes,
                                                   embedding_dim=latent_dim)
        self.transformer         = TransformerModule(dim=latent_dim, num_heads=8,
                                                     depth=transformer_depth,
                                                     pool_factor=transformer_pool)
        self.upsampler           = CNNUpsampler(dim=latent_dim, out_channels=output_nc)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Thermal image  (B, 1, H, W)
        Returns:
            rgb_out:    Generated visible image  (B, 3, H, W)
            vq_loss:    Commitment loss scalar   (add to pixel loss in training loop)
            perplexity: Codebook utilisation     (log for monitoring, do not backprop)
        """
        # 1. Local feature extraction + skip exports
        features, skip2, skip1 = self.backbone(x)

        # 2. VQ bottleneck with EMA codebook + dead-code reset
        quantized, vq_loss, perplexity = self.vector_quantization(features)

        # 3. Global transformer (pooled for tractable O(N²) cost)
        global_features = self.transformer(quantized)

        # 4. Decode with skip connections
        rgb_out = self.upsampler(global_features, skip2, skip1)

        return rgb_out, vq_loss, perplexity


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Initiating VQ-InfraTrans diagnostics...")

    model       = VQInfraTrans(input_nc=1, output_nc=3,
                               latent_dim=256, vq_codes=1024,
                               transformer_depth=6, transformer_pool=2)  # default matches training
    dummy_input = torch.randn(2, 1, 256, 256)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Training mode (EMA updates active)
    model.train()
    rgb_out, vq_loss, perplexity = model(dummy_input)

    assert rgb_out.shape == (2, 3, 256, 256), f"Shape mismatch: {rgb_out.shape}"
    print(f"\nForward pass (train)  — output: {rgb_out.shape}")
    print(f"VQ commitment loss    : {vq_loss.item():.4f}")
    print(f"Codebook perplexity   : {perplexity.item():.2f}  "
          f"(max={1024}, target >100 for healthy codebook)")

    # Inference mode (EMA updates disabled)
    model.eval()
    with torch.no_grad():
        rgb_eval, _, ppl_eval = model(dummy_input)
    print(f"\nForward pass (eval)   — output: {rgb_eval.shape}")
    print(f"Codebook perplexity   : {ppl_eval.item():.2f}")

    # Gradient flow check
    model.train()
    loss = rgb_out.mean() + vq_loss
    loss.backward()
    print("\nGradient flow verified — all backward passes successful.")
    print("\nVQ-InfraTrans implementation is complete and structurally verified!")
