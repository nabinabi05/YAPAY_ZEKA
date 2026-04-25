import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Loss Modules
# --------------------------------------------------------------------------- #

class WeberLocalContrastLoss(nn.Module):
    """
    Mathematical approximation of Human Visual Perception based on Weber's Law.
    Weber's Law: ΔI / I = k  (just-noticeable difference ∝ background intensity).

    FIX: Original computed delta_I from the full RGB tensor (pred - target) then
    compared it against a background intensity derived from the luminance tensor.
    These two quantities live in different spaces, making the ratio meaningless.
    Both delta_I and background_I are now derived consistently from luminance.
    """
    def __init__(self, window_size: int = 5, eps: float = 0.01):
        super().__init__()
        self.eps      = eps
        self.avg_pool = nn.AvgPool2d(kernel_size=window_size, stride=1,
                                     padding=window_size // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Shift [-1, 1] → [0, 1] to prevent negative denominators
        pred   = (pred   + 1.0) / 2.0
        target = (target + 1.0) / 2.0

        # Rec. 601 luma coefficients
        if pred.size(1) == 3:
            pred_lum   = 0.299 * pred[:,0:1]   + 0.587 * pred[:,1:2]   + 0.114 * pred[:,2:3]
            target_lum = 0.299 * target[:,0:1] + 0.587 * target[:,1:2] + 0.114 * target[:,2:3]
        else:
            pred_lum   = pred
            target_lum = target

        # FIX: delta_I now uses luminance maps, consistent with background_I
        delta_I      = torch.abs(pred_lum - target_lum)           # (B, 1, H, W)
        background_I = self.avg_pool(target_lum)                   # (B, 1, H, W)

        weber_contrast = delta_I / (background_I + self.eps)
        return torch.mean(weber_contrast)


class TemporalConsistencyLoss(nn.Module):
    """
    Enforces frame-to-frame temporal consistency.

    FIX: Added motion-mask weighting.
    Plain frame subtraction penalises legitimate object motion the same as
    flickering artifacts.  We compute a motion magnitude mask from the target
    sequence and down-weight regions where real motion is expected, focusing
    the penalty on static-background flickering.
    """
    def __init__(self, motion_threshold: float = 0.05):
        super().__init__()
        self.motion_threshold = motion_threshold

    def forward(self, pred_t:     torch.Tensor,
                      pred_prev:  torch.Tensor,
                      target_t:   torch.Tensor,
                      target_prev: torch.Tensor) -> torch.Tensor:

        pred_motion   = pred_t   - pred_prev
        target_motion = target_t - target_prev

        # Regions with large ground-truth motion are de-emphasised
        motion_magnitude = torch.abs(target_motion).mean(dim=1, keepdim=True)
        static_mask      = (motion_magnitude < self.motion_threshold).float()

        raw_loss    = F.smooth_l1_loss(pred_motion, target_motion, reduction='none')
        masked_loss = raw_loss * static_mask
        # Normalise by the number of static pixels to keep the scale stable
        denom = static_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / denom


# --------------------------------------------------------------------------- #
# Generator building blocks
# --------------------------------------------------------------------------- #

class ResidualBlock(nn.Module):
    """Proper residual block with affine InstanceNorm."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# --------------------------------------------------------------------------- #
# Video Generator
# --------------------------------------------------------------------------- #

class VideoGenerator(nn.Module):
    """
    Temporal Recurrent Generator conditioned on:
      • Current thermal frame   x_t      (input_nc channels)
      • Previous thermal frame  x_{t-1}  (input_nc channels)
      • Previous visible output y_{t-1}  (output_nc channels)

    FIX 1 — Encoder-decoder with skip connections (was a flat sequential).
        The original had only one downsampling step and no skip connections,
        so all high-frequency spatial detail was discarded before decoding.
        Now uses a 3-level encoder-decoder with U-Net skip connections.

    FIX 2 — Real residual blocks in the bottleneck (was pseudo-residual).
        The original "bottleneck" was two plain conv layers with no identity
        shortcut — just a pair of convolutions labelled a "proxy".

    FIX 3 — affine=True on all InstanceNorm2d layers.
    """
    def __init__(self, input_nc: int = 1, output_nc: int = 3, ngf: int = 64,
                 n_res_blocks: int = 6):
        super().__init__()
        in_ch = input_nc * 2 + output_nc   # x_t + x_prev + y_prev

        # ------------------------------------------------------------------ #
        # Encoder
        # ------------------------------------------------------------------ #
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )  # (B, ngf, H, W)

        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),
        )  # (B, ngf*2, H/2, W/2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.ReLU(inplace=True),
        )  # (B, ngf*4, H/4, W/4)

        # ------------------------------------------------------------------ #
        # Bottleneck — real residual blocks
        # ------------------------------------------------------------------ #
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(n_res_blocks)]
        )

        # ------------------------------------------------------------------ #
        # Decoder with skip connections
        # FIX 1: input channels doubled at each stage to accommodate skip
        # ------------------------------------------------------------------ #
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + ngf * 4, ngf * 2,   # skip from enc3
                               4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True),
        )  # (B, ngf*2, H/2, W/2)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + ngf * 2, ngf,        # skip from enc2
                               4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        )  # (B, ngf, H, W)

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf + ngf, output_nc, 7),                # skip from enc1
            nn.Tanh(),
        )  # (B, output_nc, H, W)

    def forward(self, x_t:    torch.Tensor,
                      x_prev: torch.Tensor = None,
                      y_prev: torch.Tensor = None) -> torch.Tensor:
        if x_prev is None:
            x_prev = torch.zeros_like(x_t)
        if y_prev is None:
            y_prev = torch.zeros(x_t.size(0), 3,
                                 x_t.size(2), x_t.size(3), device=x_t.device)

        inp = torch.cat([x_t, x_prev, y_prev], dim=1)

        # Encoder
        e1 = self.enc1(inp)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder + skip connections
        d1  = self.dec1(torch.cat([b,  e3], dim=1))
        d2  = self.dec2(torch.cat([d1, e2], dim=1))
        out = self.out_conv(torch.cat([d2, e1], dim=1))
        return out


# --------------------------------------------------------------------------- #
# Discriminator
# --------------------------------------------------------------------------- #

class SequencePatchDiscriminator(nn.Module):
    """
    Temporal-aware PatchGAN discriminator.
    Evaluates the authenticity of a frame *transition* (t-1 → t) rather than
    a single static frame, natively penalising jittery fake transitions.

    FIX: Added spectral normalisation to all conv layers.
        Without it, discriminator gradients can explode and destabilise GAN
        training.  Spectral norm constrains the Lipschitz constant of each
        layer with zero extra hyperparameters.
    """
    def __init__(self, input_nc: int = 1, output_nc: int = 3, ndf: int = 64):
        super().__init__()
        in_ch = (input_nc + output_nc) * 2   # (x_t, y_t, x_{t-1}, y_{t-1})

        SN = nn.utils.spectral_norm             # FIX: spectral norm alias

        self.net = nn.Sequential(
            # No norm on first layer (standard PatchGAN practice)
            SN(nn.Conv2d(in_ch,    ndf,     4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(ndf,      ndf * 2, 4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(ndf * 2,  ndf * 4, 4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            SN(nn.Conv2d(ndf * 4,  1,       4, stride=1, padding=1)),
            # PatchGAN: outputs a spatial validity map, not a single logit
        )

    def forward(self, x_t:   torch.Tensor,
                      y_t:   torch.Tensor,
                      x_prev: torch.Tensor,
                      y_prev: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_t, y_t, x_prev, y_prev], dim=1)
        return self.net(inp)


# --------------------------------------------------------------------------- #
# Full FWGAN model
# --------------------------------------------------------------------------- #

class FWGANArchive(nn.Module):
    """
    FWGAN (Flicker-free Weber GAN)
    Thermal video sequence → Visible domain sequence translation with:
      • Weber perceptual contrast loss  (human luminance perception)
      • Motion-masked temporal consistency loss  (anti-flicker)
      • LSGAN adversarial loss  (Least-Squares GAN objective)
      • Spectral-normalised discriminator  (training stability)
      • U-Net generator with real residual bottleneck  (image quality)

    Training note
    -------------
    Always clip gradients after loss.backward():

        torch.nn.utils.clip_grad_norm_(model.generator.parameters(),     1.0)
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 3,
                 lambda_l1: float = 10.0,
                 lambda_weber: float = 10.0,
                 lambda_temp: float = 5.0):
        super().__init__()
        self.generator     = VideoGenerator(input_nc=input_nc, output_nc=output_nc)
        self.discriminator = SequencePatchDiscriminator(input_nc=input_nc,
                                                        output_nc=output_nc)
        # FIX: lambda_l1 is now a proper parameter instead of a hardcoded *10
        self.lambda_l1    = lambda_l1
        self.lambda_weber = lambda_weber
        self.lambda_temp  = lambda_temp

        self.weber_criterion   = WeberLocalContrastLoss(window_size=5, eps=0.01)
        self.temporal_criterion = TemporalConsistencyLoss(motion_threshold=0.05)

    # ---------------------------------------------------------------------- #
    # Generation
    # ---------------------------------------------------------------------- #

    def forward_generate(self, x_t:    torch.Tensor,
                               x_prev: torch.Tensor = None,
                               y_prev: torch.Tensor = None) -> torch.Tensor:
        return self.generator(x_t, x_prev, y_prev)

    # ---------------------------------------------------------------------- #
    # Generator losses
    # ---------------------------------------------------------------------- #

    def compute_generator_losses(self,
                                  pred_t:         torch.Tensor,
                                  target_t:       torch.Tensor,
                                  pred_prev:      torch.Tensor,
                                  target_prev:    torch.Tensor,
                                  disc_pred_fake: torch.Tensor):
        """
        Aggregated generator loss:
          L_G = L_adv  +  λ_l1 * L_L1  +  λ_weber * L_weber  +  λ_temp * L_temp

        Args:
            pred_t:         Generated frame at t
            target_t:       Ground-truth visible frame at t
            pred_prev:      Generated frame at t-1
            target_prev:    Ground-truth visible frame at t-1
            disc_pred_fake: Discriminator output on the fake transition
        Returns:
            (total_loss, loss_dict)
        """
        # LSGAN adversarial loss (generator wants D to output 1)
        gan_loss   = F.mse_loss(disc_pred_fake, torch.ones_like(disc_pred_fake))

        # Pixel-level reconstruction
        l1_loss    = F.l1_loss(pred_t, target_t) * self.lambda_l1

        # Perceptual Weber contrast
        weber_loss = self.weber_criterion(pred_t, target_t) * self.lambda_weber

        # Temporal anti-flicker (motion-masked)
        if pred_prev is not None and target_prev is not None:
            temp_loss = self.temporal_criterion(pred_t, pred_prev,
                                                target_t, target_prev) * self.lambda_temp
        else:
            temp_loss = torch.tensor(0.0, device=pred_t.device)

        total = gan_loss + l1_loss + weber_loss + temp_loss

        loss_dict = {
            "gan":   gan_loss.item(),
            "l1":    l1_loss.item(),
            "weber": weber_loss.item(),
            "temp":  temp_loss.item(),
            "total": total.item(),
        }
        return total, loss_dict

    # ---------------------------------------------------------------------- #
    # FIX: Discriminator losses (was entirely missing)
    # ---------------------------------------------------------------------- #

    def compute_discriminator_losses(self,
                                      disc_pred_real: torch.Tensor,
                                      disc_pred_fake: torch.Tensor):
        """
        LSGAN discriminator loss.
        The discriminator is trained separately from the generator.
        disc_pred_fake must be computed on a detached generator output so that
        gradients do not flow back through the generator.

        Typical usage in the training loop:
            pred_t_detached = pred_t.detach()
            disc_real = model.discriminator(x_t, target_t, x_prev, target_prev)
            disc_fake = model.discriminator(x_t, pred_t_detached, x_prev, pred_prev.detach())
            d_loss, d_log = model.compute_discriminator_losses(disc_real, disc_fake)

        Args:
            disc_pred_real: Discriminator output on the real transition
            disc_pred_fake: Discriminator output on the fake (detached) transition
        Returns:
            (total_d_loss, loss_dict)
        """
        real_loss = F.mse_loss(disc_pred_real, torch.ones_like(disc_pred_real))
        fake_loss = F.mse_loss(disc_pred_fake, torch.zeros_like(disc_pred_fake))
        total_d   = (real_loss + fake_loss) * 0.5

        loss_dict = {
            "d_real":  real_loss.item(),
            "d_fake":  fake_loss.item(),
            "d_total": total_d.item(),
        }
        return total_d, loss_dict


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Initializing FWGAN Architecture...")
    fwgan = FWGANArchive(input_nc=1, output_nc=3)

    total_params = sum(p.numel() for p in fwgan.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_x_t    = torch.randn(1, 1, 128, 128)
    dummy_x_prev = torch.randn(1, 1, 128, 128)
    target_t     = torch.randn(1, 3, 128, 128)
    target_prev  = torch.randn(1, 3, 128, 128)

    # t-1 frame
    print("\nSynthesizing base frame (t-1)...")
    pred_prev = fwgan.forward_generate(dummy_x_prev)
    print(f"  Base output shape : {pred_prev.shape}")

    # t frame
    print("Synthesizing sequential frame (t)...")
    pred_t = fwgan.forward_generate(dummy_x_t, dummy_x_prev, pred_prev)
    print(f"  Frame output shape: {pred_t.shape}")

    # Discriminator on fake transition
    disc_fake = fwgan.discriminator(dummy_x_t, pred_t, dummy_x_prev, pred_prev)
    print(f"  Discriminator patch map shape: {disc_fake.shape}")

    # Discriminator on real transition
    disc_real = fwgan.discriminator(dummy_x_t, target_t, dummy_x_prev, target_prev)

    # Generator losses
    g_loss, g_log = fwgan.compute_generator_losses(
        pred_t, target_t, pred_prev, target_prev, disc_fake
    )
    print(f"\nGenerator losses : {g_log}")

    # Discriminator losses  (on detached fakes)
    disc_fake_detached = fwgan.discriminator(
        dummy_x_t, pred_t.detach(), dummy_x_prev, pred_prev.detach()
    )
    d_loss, d_log = fwgan.compute_discriminator_losses(disc_real, disc_fake_detached)
    print(f"Discriminator losses : {d_log}")

    print("\nFWGAN implementation is complete and structurally verified!")
