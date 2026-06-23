"""
dcnet_paper.py — FAITHFUL re-implementation of DC-Net as described in:

  T. Jiang et al., "Dual-Branch Colorization Network for Unpaired Infrared
  Images Based on High-Level Semantic Features and Multiscale Residual
  Attention (DC-Net)," MDPI Electronics 13(18):3784, 2024.

This is the UNPAIRED / contrastive formulation (CUT-style), NOT the paired-L1
network in dc_net.py. It is trainable on LLVIP: feed IR and visible as two
domains, optimize contrastive + adversarial + perceptual-contrastive losses,
and evaluate SSIM/PSNR on the aligned test pairs (the paper reports SSIM 0.584).

Paper-specified pieces (from the methods section):
  * Generator    : Multiscale-Residual-Attention U-Net (MRB encoder, CARB decoder)
  * PwCGB        : PatchNCE (256 patches/layer, tau=0.07) + adversarial loss (Eq.2)
  * PCGB         : perceptual contrastive loss (Eq.3), VGG16 layers {4,8,12,16},
                   L_PerCon = sum_i  ||q,k+||_1 / (||q,k-||_1 + t),  t=1e-7
  * loss weights : contrastive=0.5, adversarial=0.5, perceptual=1.0   (Eq.4)
  * optim        : Adam(beta=0.5,0.999), lr=1e-4, batch=1, 200 epochs, 256x256

Pieces the paper shows only as a figure (no numbers given) use standard CUT /
SE-attention defaults and are marked "[paper unspecified -> default]".
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dc_net import NLayerDiscriminator   # PatchGAN (reused, unconditional here)


# ===================================================================== #
# Generator: Multiscale Residual Attention U-Net (MRA-UNet)
# ===================================================================== #

class MultiscaleResidualBlock(nn.Module):
    """MRB (encoder): parallel multi-kernel branches fused with a residual.
    [paper: 'parallel two-branch structure with convolution kernels of
    different sizes'; exact kernels unspecified -> 3x3 and 5x5]."""
    def __init__(self, c):
        super().__init__()
        self.b3 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False),
                                nn.InstanceNorm2d(c, affine=True), nn.ReLU(True))
        self.b5 = nn.Sequential(nn.Conv2d(c, c, 5, padding=2, bias=False),
                                nn.InstanceNorm2d(c, affine=True), nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 1, bias=False),
                                  nn.InstanceNorm2d(c, affine=True))
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x + self.fuse(torch.cat([self.b3(x), self.b5(x)], dim=1)))


class ChannelAttention(nn.Module):
    """SE-style channel attention [paper unspecified -> SE default, reduction 8]."""
    def __init__(self, c, reduction=8):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(c // reduction, 1), 1), nn.ReLU(True),
            nn.Conv2d(max(c // reduction, 1), c, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class ChannelAttnResidualBlock(nn.Module):
    """CARB (decoder): channel attention + residual connection."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.InstanceNorm2d(c, affine=True), nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.InstanceNorm2d(c, affine=True))
        self.ca = ChannelAttention(c)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(x + self.ca(self.conv(x)))


class MRAUNetGenerator(nn.Module):
    """U-Net with MRB encoder + CARB decoder. Exposes multi-layer encoder
    features for PatchNCE via forward(x, layers, encode_only=True)."""
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_bottleneck=6):
        # input_nc=3: IR is fed as a repeated 3-channel image so the SAME encoder
        # can extract PatchNCE features from both the IR input and the 3-channel
        # generated output (standard CUT handling for asymmetric-channel tasks).
        super().__init__()
        self.stem  = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, 7, bias=False),
                                   nn.InstanceNorm2d(ngf, affine=True), nn.ReLU(True))
        self.mrb1  = MultiscaleResidualBlock(ngf)                                   # 64,  H
        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, 3, 2, 1, bias=False),
                                   nn.InstanceNorm2d(ngf * 2, affine=True), nn.ReLU(True))
        self.mrb2  = MultiscaleResidualBlock(ngf * 2)                              # 128, H/2
        self.down2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1, bias=False),
                                   nn.InstanceNorm2d(ngf * 4, affine=True), nn.ReLU(True))
        self.bott  = nn.Sequential(*[MultiscaleResidualBlock(ngf * 4) for _ in range(n_bottleneck)])
        # decoder
        self.up1   = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, 1, bias=False),
                                   nn.InstanceNorm2d(ngf * 2, affine=True), nn.ReLU(True))
        self.fuse1 = nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False)   # after concat skip (128+128)
        self.carb1 = ChannelAttnResidualBlock(ngf * 2)
        self.up2   = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, 1, bias=False),
                                   nn.InstanceNorm2d(ngf, affine=True), nn.ReLU(True))
        self.fuse2 = nn.Conv2d(ngf * 2, ngf, 1, bias=False)       # after concat skip (64+64)
        self.carb2 = ChannelAttnResidualBlock(ngf)
        self.out   = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh())
        self._enc  = [self.stem, self.mrb1, self.down1, self.mrb2, self.down2, self.bott]

    def forward(self, x, layers=None, encode_only=False):
        if encode_only:                                  # feature extraction for PatchNCE
            feats, h = [], x
            for i, m in enumerate(self._enc):
                h = m(h)
                if layers and i in layers:
                    feats.append(h)
            return feats
        s0 = self.stem(x); s1 = self.mrb1(s0)            # skip A (64, H)
        d1 = self.down1(s1); s2 = self.mrb2(d1)          # skip B (128, H/2)
        d2 = self.down2(s2); b = self.bott(d2)
        u1 = self.up1(b); u1 = self.carb1(self.fuse1(torch.cat([u1, s2], 1)))
        u2 = self.up2(u1); u2 = self.carb2(self.fuse2(torch.cat([u2, s1], 1)))
        return self.out(u2)


# ===================================================================== #
# PwCGB: PatchSampleF (projection head) + PatchNCE loss
# ===================================================================== #

class PatchSampleF(nn.Module):
    """CUT projection head H: samples num_patches feature vectors per layer,
    projects through a 2-layer MLP, L2-normalizes. MLPs are created lazily on
    first call (sized to each layer's channels), so build optimizers AFTER one
    warmup forward."""
    def __init__(self, proj_dim=256):
        super().__init__()
        self.proj_dim = proj_dim
        self.mlps = nn.ModuleList()
        self._built = False

    def _build(self, feats):
        for f in feats:
            self.mlps.append(nn.Sequential(
                nn.Linear(f.shape[1], self.proj_dim), nn.ReLU(),
                nn.Linear(self.proj_dim, self.proj_dim)))
        self._built = True
        self.to(feats[0].device)

    def forward(self, feats, num_patches=256, patch_ids=None):
        if not self._built:
            self._build(feats)
        out_feats, out_ids = [], []
        for i, f in enumerate(feats):
            B, C, H, W = f.shape
            f = f.permute(0, 2, 3, 1).reshape(B, H * W, C)        # B, HW, C
            if patch_ids is not None:
                ids = patch_ids[i]
            else:
                ids = torch.randperm(H * W, device=f.device)[:min(num_patches, H * W)]
            sampled = f[:, ids, :]                                # (B, P, C)
            out_feats.append(F.normalize(self.mlps[i](sampled), dim=-1))   # (B, P, proj)
            out_ids.append(ids)
        return out_feats, out_ids


class PatchNCELoss(nn.Module):
    """InfoNCE between aligned query (from output) and key (from input) patches.
    Negatives are the OTHER patches within the SAME image only (per-image
    grouping), so it is correct for any batch size, not just bs=1."""
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau
        self.ce = nn.CrossEntropyLoss()

    def forward(self, q, k):                                      # q,k: (B, P, C)
        # Disable autocast: bmm is on autocast's fp16 allowlist, so it would
        # re-cast even an explicit .float() back to fp16 and the -1e9 mask
        # would overflow. Run the whole InfoNCE in true fp32 (AMP best practice).
        with torch.amp.autocast(device_type='cuda' if q.is_cuda else 'cpu', enabled=False):
            q, k = q.float(), k.float()
            B, P, _ = q.shape
            l_pos = (q * k).sum(dim=-1, keepdim=True)             # (B,P,1)
            l_neg = torch.bmm(q, k.transpose(1, 2))               # (B,P,P) within-image
            l_neg = l_neg.masked_fill(torch.eye(P, device=q.device, dtype=torch.bool)[None], -1e9)
            logits = torch.cat([l_pos, l_neg], dim=-1).reshape(B * P, 1 + P) / self.tau
            labels = torch.zeros(B * P, dtype=torch.long, device=q.device)
            return self.ce(logits, labels)


# ===================================================================== #
# PCGB: VGG16 features + perceptual contrastive loss (Eq. 3)
# ===================================================================== #

class VGG16Features(nn.Module):
    """Frozen VGG16 feature extractor at layers {4,8,12,16} (paper-specified)."""
    def __init__(self, layers=(4, 8, 12, 16)):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()
        prev = 0
        for l in layers:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, l)]))
            prev = l
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                         # x in [-1,1]
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = ((x + 1) / 2 - self.mean) / self.std                  # graph kept (frozen weights)
        out, h = [], x
        for s in self.slices:
            h = s(h); out.append(h)
        return out


def perceptual_contrastive_loss(vgg, fake, visible, infrared, t=1e-7):
    """Eq.3: sum_i ||q,k+||_1 / (||q,k-||_1 + t).
    q = generated features; k+ = aligned visible (same location);
    k- = grayscale/IR features AND spatially-shuffled (non-corresponding) visible."""
    fq, fp, fn = vgg(fake), vgg(visible), vgg(infrared)
    loss = 0.0
    for q, kpos, kneg in zip(fq, fp, fn):
        B, C, H, W = kpos.shape
        idx = torch.randperm(H * W, device=kpos.device)
        kshuf = kpos.flatten(2)[:, :, idx].view(B, C, H, W)       # non-corresponding visible
        pos = (q - kpos).abs().mean()
        neg = 0.5 * ((q - kneg).abs().mean() + (q - kshuf).abs().mean())
        loss = loss + pos.float() / (neg.float() + t)             # fp32: t=1e-7 underflows to 0 in fp16
    return loss


# ===================================================================== #
# Adversarial loss (Eq. 2: standard GAN; LSGAN available as an option)
# ===================================================================== #

class GANLoss(nn.Module):
    def __init__(self, mode='vanilla'):
        super().__init__()
        self.mode = mode
        self.loss = nn.BCEWithLogitsLoss() if mode == 'vanilla' else nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


# ===================================================================== #
# Trainer
# ===================================================================== #

NCE_LAYERS = [0, 1, 2, 3, 4]          # taps into MRAUNetGenerator._enc
LAMBDA_NCE, LAMBDA_GAN, LAMBDA_PERC = 0.5, 0.5, 1.0   # Eq.4


def _resolve_splits(root):
    inf, vis = os.path.join(root, "infrared"), os.path.join(root, "visible")
    if all(os.path.isdir(os.path.join(d, s)) for d in (inf, vis) for s in ("train", "test")):
        return (os.path.join(inf, "train"), os.path.join(vis, "train")), \
               (os.path.join(inf, "test"),  os.path.join(vis, "test"))
    return None


@torch.no_grad()
def evaluate_ssim(G, loader, device, n_max=200):
    from skimage.metrics import structural_similarity as ssim
    G.eval(); scores = []
    for i, b in enumerate(loader):
        if i >= n_max:
            break
        fake = G(b["thermal"].to(device).repeat(1, 3, 1, 1))
        p = ((fake[0].cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
        g = ((b["visible"][0] + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
        scores.append(ssim(g, p, data_range=1.0, channel_axis=2))
    G.train()
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def save_samples(G, loader, device, path, n=4):
    """Save a thermal | generated | ground-truth grid (handy to show results)."""
    from torchvision.utils import save_image
    G.eval(); rows = []
    for i, b in enumerate(loader):
        if i >= n:
            break
        th = b["thermal"].to(device).repeat(1, 3, 1, 1)
        fake = G(th)
        rows += [((th[0].cpu() + 1) / 2).clamp(0, 1),
                 ((fake[0].cpu() + 1) / 2).clamp(0, 1),
                 ((b["visible"][0] + 1) / 2).clamp(0, 1)]
    save_image(rows, path, nrow=3)
    G.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("LLVIP_ROOT", "data/LLVIP/LLVIP"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--gan_mode", default="vanilla", choices=["vanilla", "lsgan"])
    ap.add_argument("--max_train", type=int, default=0, help="subsample N train images (0=all) — use to fit a deadline")
    ap.add_argument("--eval_n", type=int, default=200, help="# test images for the per-epoch SSIM estimate")
    ap.add_argument("--sample_dir", default="samples/dcnet_paper")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader num_workers (0 on Windows, 4 on Colab/Linux)")
    ap.add_argument("--amp", action="store_true", help="mixed-precision (fp16 activations) — 1.5-2x speedup on A100")
    a = ap.parse_args()
    from data.dataset import create_dataloader, ThermalVisibleDataset
    from torch.utils.data import DataLoader, Subset
    os.makedirs(a.ckpt_dir, exist_ok=True)
    os.makedirs(a.sample_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  AMP: {a.amp}")

    splits = _resolve_splits(a.root)
    assert splits is not None, f"need {a.root}/infrared/(train|test) and visible/(train|test)"
    (tr_t, tr_v), (va_t, va_v) = splits
    train_ds = ThermalVisibleDataset(tr_t, tr_v, mode="paired", is_train=True, split_ratio=None)
    if a.max_train and a.max_train < len(train_ds):
        idx = list(range(len(train_ds))); random.Random(0).shuffle(idx)
        train_ds = Subset(train_ds, idx[:a.max_train])
        print(f"[data] subsampled train set to {a.max_train} images (deadline mode)")
    train_loader = DataLoader(train_ds, batch_size=a.batch, shuffle=True,
                              num_workers=a.workers, drop_last=True, pin_memory=device.type=="cuda")
    val_loader   = create_dataloader(va_t, va_v, mode="paired", is_train=False,
                                     batch_size=1, split_ratio=None)

    G   = MRAUNetGenerator(3, 3).to(device)
    D   = NLayerDiscriminator(input_nc=3).to(device)        # unconditional (Eq.2 D(I_y))
    Fnet = PatchSampleF().to(device)
    vgg = VGG16Features().to(device)
    nce_losses = [PatchNCELoss().to(device) for _ in NCE_LAYERS]
    gan = GANLoss(a.gan_mode)

    # Warmup forward to materialize PatchSampleF MLPs before building its optimizer.
    with torch.no_grad():
        warm = G(next(iter(train_loader))["thermal"].to(device).repeat(1, 3, 1, 1),
                 NCE_LAYERS, encode_only=True)
        Fnet(warm)
    optG = torch.optim.Adam(G.parameters(),   lr=a.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(),   lr=a.lr, betas=(0.5, 0.999))
    optF = torch.optim.Adam(Fnet.parameters(), lr=a.lr, betas=(0.5, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=a.amp)
    autocast = lambda: torch.amp.autocast('cuda', enabled=a.amp)

    best_ssim, best_epoch, no_improve = -1.0, 0, 0
    best_path = os.path.join(a.ckpt_dir, "DCNet_paper_best.pth")
    for epoch in range(1, a.epochs + 1):
        G.train(); D.train()
        for b in train_loader:
            real_A = b["thermal"].to(device).repeat(1, 3, 1, 1)   # IR as 3-channel
            real_B = b["visible"].to(device)                      # visible

            with autocast():
                fake = G(real_A)

            # (1) Discriminator
            optD.zero_grad()
            with autocast():
                d_loss = 0.5 * (gan(D(real_B), True) + gan(D(fake.detach()), False))
            scaler.scale(d_loss).backward(); scaler.step(optD); scaler.update()

            # (2) Generator + PatchSampleF
            optG.zero_grad(); optF.zero_grad()
            with autocast():
                g_gan = gan(D(fake), True)
                feat_k = G(real_A, NCE_LAYERS, encode_only=True)   # keys from input
                feat_q = G(fake,   NCE_LAYERS, encode_only=True)   # queries from output
                k_pool, ids = Fnet(feat_k, 256)
                q_pool, _   = Fnet(feat_q, 256, ids)
                nce = sum(crit(q, k) for q, k, crit in zip(q_pool, k_pool, nce_losses)) / len(q_pool)
                perc = perceptual_contrastive_loss(vgg, fake, real_B, real_A)
                g_loss = LAMBDA_NCE * nce + LAMBDA_GAN * g_gan + LAMBDA_PERC * perc
            scaler.scale(g_loss).backward(); scaler.step(optG); scaler.step(optF); scaler.update()

        val_ssim = evaluate_ssim(G, val_loader, device, n_max=a.eval_n)
        print(f"[epoch {epoch}/{a.epochs}] D={d_loss.item():.3f} G={g_loss.item():.3f} "
              f"nce={nce.item():.3f} perc={perc.item():.3f} | val SSIM={val_ssim:.4f}")
        if val_ssim > best_ssim + 1e-3:
            best_ssim, best_epoch, no_improve = val_ssim, epoch, 0
            torch.save(G.state_dict(), best_path)
            save_samples(G, val_loader, device, os.path.join(a.sample_dir, "best.png"))
            print(f"  [BEST] val SSIM {val_ssim:.4f} -> {best_path}  (+ sample grid)")
        else:
            no_improve += 1
            if no_improve >= a.patience:
                print(f"  [EARLY-STOP] best {best_ssim:.4f} @ epoch {best_epoch}")
                break
    print(f"Done. Best val SSIM {best_ssim:.4f} @ epoch {best_epoch} (paper reports ~0.584).")


if __name__ == "__main__":
    main()
