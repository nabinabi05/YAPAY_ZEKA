"""
generate_samples.py
-------------------
Loads each model checkpoint, picks N validation image pairs,
runs inference, and saves a single side-by-side comparison figure.

Layout per row:
  [Thermal Input] | [DCNet] | [FWGAN] | [VQ-InfraTrans] | [Inter-Mamba] | [Cond-DDPM] | [GT Visible]
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.dc_net        import DCNet
from models.diffusion_model import ThermalToVisibleDDPM, ConditionalUNet
from models.fwgan          import FWGANArchive
from models.vq_infratrans  import VQInfraTrans
from models.mamba_fusion   import InterMambaBlock
from data.dataset          import get_image_paths


# ── MambaTranslatorProxy (copied from train_and_eval.py) ────────────────────
class MambaTranslatorProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True), nn.SiLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True), nn.SiLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True), nn.SiLU())
        self.mamba = InterMambaBlock(dim=256)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True), nn.SiLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True), nn.SiLU())
        self.out = nn.Sequential(nn.Conv2d(128, 3, 3, padding=1), nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m  = self.mamba(visible=e3, thermal=e3)
        d1 = self.dec1(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        return self.out(torch.cat([d2, e1], dim=1))


# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR     = os.path.dirname(__file__)
THERMAL_DIR  = os.path.join("data", "LLVIP", "LLVIP", "infrared")
VISIBLE_DIR  = os.path.join("data", "LLVIP", "LLVIP", "visible")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "sample_comparison.png")
N_SAMPLES    = 6          # number of image rows
IMG_SIZE     = (256, 256)
SPLIT_RATIO  = 0.8        # same as training
DDIM_STEPS   = 50
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] CHW tensor → HWC uint8 numpy array."""
    arr = ((t.float().cpu().clamp(-1, 1) + 1.0) / 2.0).numpy()
    arr = np.transpose(arr, (1, 2, 0))           # HWC
    return (arr * 255).clip(0, 255).astype(np.uint8)


def load_image_pair(th_path: str, vi_path: str):
    """Load one (thermal, visible) pair → normalised tensors."""
    th = Image.open(th_path).convert("L")
    vi = Image.open(vi_path).convert("RGB")
    th = TF.resize(th, IMG_SIZE, interpolation=TF.InterpolationMode.BICUBIC)
    vi = TF.resize(vi, IMG_SIZE, interpolation=TF.InterpolationMode.BICUBIC)

    th_t = TF.to_tensor(th)                        # [0,1]
    th_t = TF.normalize(th_t, (0.5,), (0.5,))      # [-1,1]

    vi_t = TF.to_tensor(vi)
    vi_t = TF.normalize(vi_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return th_t.unsqueeze(0), vi_t.unsqueeze(0)    # (1,1,H,W), (1,3,H,W)


# ── Model loading ──────────────────────────────────────────────────────────────
def build_models() -> dict:
    models = {}

    # DCNet
    m = DCNet(input_nc=1, output_nc=3)
    ckpt = os.path.join(CKPT_DIR, "DCNet_final.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval().to(DEVICE)
    models["DCNet"] = m
    print(f"  [✓] DCNet loaded  ({sum(p.numel() for p in m.parameters()):,} params)")

    # FWGAN
    m = FWGANArchive(input_nc=1, output_nc=3)
    ckpt = os.path.join(CKPT_DIR, "FWGAN_final.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval().to(DEVICE)
    models["FWGAN"] = m
    print(f"  [✓] FWGAN loaded  ({sum(p.numel() for p in m.parameters()):,} params)")

    # VQ-InfraTrans
    m = VQInfraTrans(input_nc=1, output_nc=3)
    ckpt = os.path.join(CKPT_DIR, "VQ-InfraTrans_final.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval().to(DEVICE)
    models["VQ-InfraTrans"] = m
    print(f"  [✓] VQ-InfraTrans loaded  ({sum(p.numel() for p in m.parameters()):,} params)")

    # Inter-Mamba
    m = MambaTranslatorProxy()
    ckpt = os.path.join(CKPT_DIR, "Inter-Mamba_final.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval().to(DEVICE)
    models["Inter-Mamba"] = m
    print(f"  [✓] Inter-Mamba loaded  ({sum(p.numel() for p in m.parameters()):,} params)")

    # Cond-DDPM
    unet = ConditionalUNet(c_in=4, c_out=3)
    m    = ThermalToVisibleDDPM(network=unet, T=1000)
    ckpt = os.path.join(CKPT_DIR, "Cond-DDPM_final.pth")
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.eval().to(DEVICE)
    models["Cond-DDPM"] = m
    print(f"  [✓] Cond-DDPM loaded  ({sum(p.numel() for p in m.parameters()):,} params)")

    return models


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(models: dict, thermal: torch.Tensor, visible: torch.Tensor) -> dict:
    th = thermal.to(DEVICE)
    vi = visible.to(DEVICE)
    preds = {}

    preds["DCNet"]         = models["DCNet"](th, extract_features=False)
    preds["FWGAN"]         = models["FWGAN"].forward_generate(th)
    preds["VQ-InfraTrans"] = models["VQ-InfraTrans"](th)[0]
    preds["Inter-Mamba"]   = models["Inter-Mamba"](th)
    preds["Cond-DDPM"]     = models["Cond-DDPM"].sample(
                                 th,
                                 shape=(th.shape[0], 3, IMG_SIZE[0], IMG_SIZE[1]),
                                 ddim_steps=DDIM_STEPS)
    return preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # ── Dataset paths ────────────────────────────────────────────────────────
    all_thermal = get_image_paths(THERMAL_DIR)
    all_visible = get_image_paths(VISIBLE_DIR)
    split_idx   = int(len(all_thermal) * SPLIT_RATIO)
    val_thermal = all_thermal[split_idx:]
    val_visible = all_visible[split_idx:]
    print(f"Validation set: {len(val_thermal)} pairs")

    # Randomly pick N_SAMPLES pairs
    indices = random.sample(range(len(val_thermal)), min(N_SAMPLES, len(val_thermal)))

    # ── Load models ──────────────────────────────────────────────────────────
    print("\nLoading checkpoints...")
    models = build_models()

    col_names  = ["Thermal\n(Input)", "DCNet", "FWGAN",
                  "VQ-InfraTrans", "Inter-Mamba", "Cond-DDPM", "Ground\nTruth"]
    n_cols     = len(col_names)
    n_rows     = len(indices)
    fig_w      = n_cols * 3.0
    fig_h      = n_rows * 3.0 + 0.6   # extra for column headers

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.patch.set_facecolor("#111111")

    # Column headers
    header_colors = {
        "Thermal\n(Input)":  "#4a9eff",
        "DCNet":             "#2ecc71",
        "FWGAN":             "#e74c3c",
        "VQ-InfraTrans":     "#f39c12",
        "Inter-Mamba":       "#9b59b6",
        "Cond-DDPM":         "#1abc9c",
        "Ground\nTruth":     "#ecf0f1",
    }

    print("\nGenerating images...")
    for row_idx, img_idx in enumerate(indices):
        th_t, vi_t = load_image_pair(val_thermal[img_idx], val_visible[img_idx])
        preds       = run_inference(models, th_t, vi_t)

        # Build column images
        col_imgs = [
            tensor_to_numpy(th_t[0].repeat(3, 1, 1)),   # thermal → greyscale RGB
            tensor_to_numpy(preds["DCNet"][0]),
            tensor_to_numpy(preds["FWGAN"][0]),
            tensor_to_numpy(preds["VQ-InfraTrans"][0]),
            tensor_to_numpy(preds["Inter-Mamba"][0]),
            tensor_to_numpy(preds["Cond-DDPM"][0]),
            tensor_to_numpy(vi_t[0]),                    # ground truth
        ]

        for col_idx, (img, cname) in enumerate(zip(col_imgs, col_names)):
            ax = axes[row_idx][col_idx]
            ax.imshow(img)
            ax.axis("off")

            # Top border colour per column
            for spine in ax.spines.values():
                spine.set_edgecolor(header_colors[cname])
                spine.set_linewidth(2)
                spine.set_visible(True)

            # Column labels only on first row
            if row_idx == 0:
                ax.set_title(cname,
                             color=header_colors[cname],
                             fontsize=10, fontweight="bold",
                             pad=4)

        print(f"  Row {row_idx+1}/{n_rows} done — {os.path.basename(val_thermal[img_idx])}")

    plt.tight_layout(pad=0.4)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n✓ Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
