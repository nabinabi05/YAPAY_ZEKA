import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualBlock(nn.Module):
    """Standard residual block with reflection padding and affine InstanceNorm.

    Optional dropout for regularization. It is appended at the END of the block
    so it adds no parameters and never shifts the conv/norm state_dict indices —
    a dropout>0 model and a dropout=0 model share identical keys, so existing
    checkpoints still load either way.
    """
    def __init__(self, channels, dropout=0.0):
        super(ResidualBlock, self).__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),   # affine=True: learnable scale/shift
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based generator backbone for Thermal-to-Visible image translation.
    
    Key fix: U-Net style skip connections are now used from encoder to decoder,
    ensuring that high-resolution spatial details are preserved through the network
    and not lost in the bottleneck. Encoder/decoder channel sizes are adjusted
    accordingly to account for concatenated skip tensors.
    """
    def __init__(self, input_nc=1, output_nc=3, ngf=64, n_blocks=9, res_dropout=0.0):
        super(Generator, self).__init__()

        # ------------------------------------------------------------------ #
        # 1. Encoding layers
        # ------------------------------------------------------------------ #
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),        # FIX: affine=True
            nn.ReLU(inplace=True)
        )  # output: (B, 64, H, W)

        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),    # FIX: affine=True
            nn.ReLU(inplace=True)
        )  # output: (B, 128, H/2, W/2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),    # FIX: affine=True
            nn.ReLU(inplace=True)
        )  # output: (B, 256, H/4, W/4)

        # ------------------------------------------------------------------ #
        # 2. ResNet bottleneck blocks
        # ------------------------------------------------------------------ #
        res_blocks = [ResidualBlock(ngf * 4, dropout=res_dropout) for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        # output: (B, 256, H/4, W/4)

        # ------------------------------------------------------------------ #
        # 3. Decoding / Upsampling layers
        #
        # FIX: Skip connections concatenate encoder features to decoder inputs.
        # dec1 receives:  res  (256) + enc3 (256) = 512 channels in
        # dec2 receives: dec1 (128) + enc2 (128) = 256 channels in
        # out_conv receives: dec2 (64) + enc1 (64) = 128 channels in
        # ------------------------------------------------------------------ #
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 + ngf * 4, ngf * 2,  # 512 -> 128
                               3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True)
        )  # output: (B, 128, H/2, W/2)

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 + ngf * 2, ngf,      # 256 -> 64
                               3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True)
        )  # output: (B, 64, H, W)

        self.out_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf + ngf, output_nc, 7),             # 128 -> output_nc
            nn.Tanh()
        )  # output: (B, output_nc, H, W)

    def forward(self, x, return_features=False):
        # Encoder
        feat1 = self.enc1(x)    # (B, 64,  H,   W)
        feat2 = self.enc2(feat1) # (B, 128, H/2, W/2)
        feat3 = self.enc3(feat2) # (B, 256, H/4, W/4)

        # Bottleneck
        res = self.res_blocks(feat3)  # (B, 256, H/4, W/4)

        # FIX: Decoder with skip connections via concatenation
        out1 = self.dec1(torch.cat([res, feat3], dim=1))    # (B, 128, H/2, W/2)
        out2 = self.dec2(torch.cat([out1, feat2], dim=1))   # (B, 64,  H,   W)
        out  = self.out_conv(torch.cat([out2, feat1], dim=1))  # (B, 3, H, W)

        if return_features:
            return out, [feat1, feat2, feat3, res]
        return out


class PatchContrastiveBranch(nn.Module):
    """
    Patch-wise Contrastive Guidance Branch.
    Projects intermediate backbone features into a shared latent space.
    
    FIX: forward() now accepts features from BOTH the source (thermal) and
    the generated image via return_features=True on the generator.
    The same MLP projectors are applied to both, enabling proper contrastive
    positive/negative pair construction in the training loop.
    """
    def __init__(self, in_channels_list=[64, 128, 256, 256], mlp_dim=256):
        super(PatchContrastiveBranch, self).__init__()
        self.mlps = nn.ModuleList()
        for in_channels in in_channels_list:
            mlp = nn.Sequential(
                nn.Conv2d(in_channels, mlp_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mlp_dim, mlp_dim, kernel_size=1)
            )
            self.mlps.append(mlp)

    def forward(self, features):
        """
        Args:
            features: list of feature maps from one domain
                      (either source thermal or generated image)
        Returns:
            list of L2-normalised projected feature maps
        """
        projected = []
        for feat, mlp in zip(features, self.mlps):
            proj = mlp(feat)
            proj = F.normalize(proj, dim=1)  # L2 normalise over channel dim
            projected.append(proj)
        return projected


class PerceptualContrastiveBranch(nn.Module):
    """
    Perceptual Contrastive Guidance Branch.
    Utilizes a frozen pre-trained VGG-19 network to extract hierarchical features.
    Provides guidance for high-level semantics, global structure, and style matching.
    
    FIX: torch.no_grad() is enforced inside forward() to avoid building an
    unnecessary computation graph through the frozen VGG layers.
    """
    def __init__(self, requires_grad=False):
        super(PerceptualContrastiveBranch, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential(*list(vgg.children())[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:18])  # relu3_2
        self.slice4 = nn.Sequential(*list(vgg.children())[18:27]) # relu4_2
        self.slice5 = nn.Sequential(*list(vgg.children())[27:36]) # relu5_2

        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: image tensor, either (B,1,H,W) thermal or (B,3,H,W) generated
        Returns:
            list of 5 VGG feature maps at increasing semantic depth

        NOTE: no torch.no_grad() here — VGG weights have requires_grad=False
        so they won't be updated, but the computation graph through them IS
        needed so the generator receives perceptual loss gradients via `pred`.
        Wrap target-side calls in torch.no_grad() externally for VRAM savings.
        """
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Shift [-1, 1] → [0, 1] then apply ImageNet normalisation
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)

        return [h1, h2, h3, h4, h5]


class NLayerDiscriminator(nn.Module):
    """Conditional 70x70 PatchGAN discriminator (pix2pix-style).

    Sees the INPUT and OUTPUT together: concat([thermal(1ch), visible(3ch)]) so
    it judges whether a visible image is a plausible translation OF THIS thermal
    frame, not just a plausible visible image. Output is a map of real/fake
    logits (one per ~70x70 receptive patch); use with an LSGAN (MSE) objective.
    """
    def __init__(self, input_nc=4, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw, padw = 4, 1
        layers = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
                  nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for n in range(1, n_layers):                       # progressively downsample
            nf_prev, nf_mult = nf_mult, min(2 ** n, 8)
            layers += [nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=False),
                       nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                       nn.LeakyReLU(0.2, inplace=True)]
        nf_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=False),
                   nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                   nn.LeakyReLU(0.2, inplace=True),
                   nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DCNet(nn.Module):
    """
    DC-Net (Dual-Branch Colorization Network)
    Translates Thermal/IR inputs to Visible outputs using:
      1. Patch-wise PatchNCE branch  — low-level texture coherence
      2. Perceptual VGG-19 branch    — high-level semantic guidance

    Training loop responsibilities (see NOTE below):
    ---------------------------------------------------
    For the perceptual loss to work the training loop must call
    perceptual_guidance on BOTH the generated image and the target,
    then compare the resulting feature lists, e.g.:

        perc_fake = model.perceptual_guidance(out_img)
        perc_real = model.perceptual_guidance(target_img)   # <-- required
        perceptual_loss = sum(
            F.l1_loss(f, r) for f, r in zip(perc_fake, perc_real)
        )

    For the PatchNCE loss the training loop must project features from
    BOTH the source (thermal) encoder pass and the generated output, e.g.:

        _, src_feats = model.generator(thermal_img, return_features=True)
        _, gen_feats = model.generator(out_img_detached, return_features=True)
        src_proj = model.patch_guidance(src_feats)
        gen_proj = model.patch_guidance(gen_feats)
        # then compute PatchNCE between src_proj and gen_proj
    """

    def __init__(self, input_nc=1, output_nc=3, res_dropout=0.0):
        super(DCNet, self).__init__()
        self.generator           = Generator(input_nc=input_nc, output_nc=output_nc,
                                             res_dropout=res_dropout)
        self.patch_guidance      = PatchContrastiveBranch()
        self.perceptual_guidance = PerceptualContrastiveBranch(requires_grad=False)

    def forward(self, x, extract_features=False):
        """
        Args:
            x:                Input thermal tensor  (B, input_nc, H, W)
            extract_features: Return auxiliary guidance features for training.

        Returns (inference):
            out_img

        Returns (training, extract_features=True):
            out_img, patch_feats, perc_feats
            where patch_feats are the *generated-side* projections and
            perc_feats are the VGG features of the generated image.
            The caller must separately compute the real-side counterparts
            (see class-level docstring above).
        """
        if not extract_features:
            return self.generator(x, return_features=False)

        # Training mode
        out_img, gen_features = self.generator(x, return_features=True)

        # Local branch: project generated-side spatial features
        patch_feats = self.patch_guidance(gen_features)

        # Global branch: embed generated output in VGG perceptual space
        perc_feats = self.perceptual_guidance(out_img)

        return out_img, patch_feats, perc_feats


# --------------------------------------------------------------------------- #
# Quick sanity-check / smoke-test
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    print("Initializing DC-Net...")
    model = DCNet(input_nc=1, output_nc=3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    dummy_input  = torch.randn(2, 1, 256, 256)
    dummy_target = torch.randn(2, 3, 256, 256)  # fake ground-truth visible image

    print("\n--- Inference Mode ---")
    out = model(dummy_input, extract_features=False)
    print(f"Generated image shape : {out.shape}")

    print("\n--- Training Mode ---")
    out_train, p_feats, v_feats = model(dummy_input, extract_features=True)
    print(f"Generated image shape : {out_train.shape}")
    print(f"Patch features        : {len(p_feats)} scales")
    print(f"Perceptual features   : {len(v_feats)} VGG layers")

    # Demonstrate correct perceptual loss computation
    print("\n--- Perceptual Loss Demo ---")
    perc_fake = model.perceptual_guidance(out_train)
    perc_real = model.perceptual_guidance(dummy_target)
    perceptual_loss = sum(F.l1_loss(f, r) for f, r in zip(perc_fake, perc_real))
    print(f"Perceptual loss (example): {perceptual_loss.item():.4f}")

    print("\nDC-Net architecture is ready!")
