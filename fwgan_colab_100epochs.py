"""
FWGAN 100-Epoch Training Script for Google Colab
=================================================
Run via the COLAB_NOTEBOOK_CELLS.md setup cell which handles:
  - Drive mount, GitHub clone, dataset extraction
  - All outputs saved to MyDrive/FWGAN_100ep/

What it does:
  1. Finds LLVIP dataset (auto-detects from Drive zip)
  2. Trains FWGAN for 100 epochs
  3. Saves sample images every 5 epochs (train | val comparison)
  4. Logs PSNR, SSIM, MAE, RMSE every epoch
  5. Plots metric curves at the end
  6. Saves checkpoints every 25 epochs + final
"""

# ============================================================================ #
# CELL 1 — Setup & Dependencies
# ============================================================================ #
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("scikit-image")

# REPO_DIR is wherever this script lives — set by the notebook cell via os.chdir()
REPO_DIR = os.getcwd()
print(f"Working directory: {REPO_DIR}")

# ============================================================================ #
# CELL 2 — Dataset Setup (Drive → /content SSD)
# ============================================================================ #
# Strategy:
#   1. Dataset zip lives in Google Drive (slow network storage)
#   2. Unzip once to /content (fast local SSD) each session
#   3. Train from /content — much faster I/O than Drive directly
#   4. Outputs (checkpoints, samples) go back to Drive permanently
# ============================================================================ #

DATASET_LOCAL = "/content/LLVIP"          # fast SSD target
DRIVE_MYDIR   = "/content/drive/MyDrive"
DRIVE_DATASET = os.path.join(DRIVE_MYDIR, "LLVIP")  # pre-extracted in Drive (optional)

def find_llvip_dirs(search_root):
    """
    Recursively find infrared/ and visible/ dirs.
    Accepts the dir even if images live in subdirs (e.g. infrared/train/).
    """
    def has_images_recursive(path):
        for r, _, fs in os.walk(path):
            if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in fs):
                return True
        return False

    thermal_dir = visible_dir = None
    for root, dirs, files in os.walk(search_root):
        basename = os.path.basename(root)
        if basename == "infrared" and has_images_recursive(root):
            thermal_dir = root
        elif basename == "visible" and has_images_recursive(root):
            visible_dir = root
        if thermal_dir and visible_dir:
            break
    return thermal_dir, visible_dir

def find_zip_in_drive(drive_root):
    """Find LLVIP zip anywhere in Drive."""
    for root, dirs, files in os.walk(drive_root):
        for f in files:
            if "llvip" in f.lower() and f.lower().endswith(".zip"):
                return os.path.join(root, f)
    return None

THERMAL_DIR = VISIBLE_DIR = None

# ── Step 1: Check if already unzipped to /content this session ───────────────
print("Looking for LLVIP dataset...")
THERMAL_DIR, VISIBLE_DIR = find_llvip_dirs(DATASET_LOCAL)
if THERMAL_DIR and VISIBLE_DIR:
    print(f"✅ Dataset already in /content (fast SSD)")

# ── Step 2: Try pre-extracted copy in Drive ───────────────────────────────────
if not THERMAL_DIR and os.path.isdir(DRIVE_DATASET):
    print("Found extracted LLVIP in Drive — copying to /content SSD...")
    import shutil
    shutil.copytree(DRIVE_DATASET, DATASET_LOCAL, dirs_exist_ok=True)
    THERMAL_DIR, VISIBLE_DIR = find_llvip_dirs(DATASET_LOCAL)
    if THERMAL_DIR:
        print("✅ Copied from Drive to /content")

# ── Step 3: Find zip in Drive and extract ─────────────────────────────────────
if not THERMAL_DIR and os.path.isdir(DRIVE_MYDIR):
    print("Searching Drive for LLVIP zip...")
    zip_path = find_zip_in_drive(DRIVE_MYDIR)
    if zip_path:
        print(f"Found zip: {zip_path}")
        # Remove previous partial extraction (causes "overwrite?" prompts in unzip)
        import shutil as _shutil
        if os.path.exists(DATASET_LOCAL):
            print("Removing previous partial extraction...")
            _shutil.rmtree(DATASET_LOCAL)
        os.makedirs(DATASET_LOCAL, exist_ok=True)
        print("Extracting to /content/LLVIP (this takes ~2 min, only once per session)...")
        ret = subprocess.run(
            ["unzip", "-o", "-q", zip_path, "-d", DATASET_LOCAL])  # -o = overwrite without asking
        if ret.returncode == 0:
            # Search broadly after extraction — zip may create nested dirs
            THERMAL_DIR, VISIBLE_DIR = find_llvip_dirs(DATASET_LOCAL)
            if not THERMAL_DIR:
                THERMAL_DIR, VISIBLE_DIR = find_llvip_dirs("/content")
            if THERMAL_DIR:
                print(f"✅ Extracted successfully. Found at: {THERMAL_DIR}")
            else:
                # Show what was actually extracted to help debug
                print("⚠️  Unzip done but infrared/visible dirs not found. Extracted contents:")
                subprocess.run(["find", DATASET_LOCAL, "-type", "d", "-maxdepth", "5"])
        else:
            print(f"❌ unzip failed with code {ret.returncode}")
    else:
        print("No LLVIP zip found in Drive.")

# ── Step 4: Search all of /content as fallback ────────────────────────────────
if not THERMAL_DIR:
    print("Scanning /content for any existing dataset...")
    THERMAL_DIR, VISIBLE_DIR = find_llvip_dirs("/content")

# ── Final check ───────────────────────────────────────────────────────────────
if not THERMAL_DIR or not VISIBLE_DIR:
    print("\n" + "="*60)
    print("  DATASET NOT FOUND")
    print("="*60)
    print("\nUpload LLVIP zip to Google Drive, then re-run.")
    print("Expected location in Drive:  MyDrive/LLVIP.zip  (any subfolder works)")
    print("\nOR run this in a new cell to manually specify:")
    print("  ZIP_PATH = '/content/drive/MyDrive/your/path/LLVIP.zip'")
    print("  !unzip -q {ZIP_PATH} -d /content/LLVIP")
    print("="*60)
    raise FileNotFoundError("LLVIP dataset not found. Upload zip to Drive.")

from data.dataset import get_image_paths
n_thermal = len(get_image_paths(THERMAL_DIR))
n_visible = len(get_image_paths(VISIBLE_DIR))
print(f"\nDataset ready: {n_thermal} thermal, {n_visible} visible images")
print(f"  Thermal: {THERMAL_DIR}")
print(f"  Visible: {VISIBLE_DIR}")


# ============================================================================ #
# CELL 3 — Imports & Config
# ============================================================================ #
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid, save_image

from data.dataset import create_dataloader
from models.fwgan import FWGANArchive

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE       = 128      # Matches train_and_eval.py config
EPOCHS           = 100
GEN_LR           = 2e-4
DISC_LR          = 1e-4     # TTUR: disc learns slower for stability
BETAS            = (0.5, 0.999)
GRAD_CLIP        = 1.0
SAMPLE_INTERVAL  = 5        # Save images every N epochs
REPLAY_BUF_SIZE  = 50
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP          = DEVICE.type == "cuda"

print(f"Device: {DEVICE}")
print(f"AMP enabled: {USE_AMP}")
print(f"Config: batch={BATCH_SIZE}, epochs={EPOCHS}, gen_lr={GEN_LR}, disc_lr={DISC_LR}")

# ── Output dirs — use Drive if already mounted, otherwise local ───────────────
# NOTE: Mount Drive in a SEPARATE cell BEFORE running this script:
#   from google.colab import drive; drive.mount('/content/drive')
DRIVE_MYDIR = "/content/drive/MyDrive"
if os.path.isdir(DRIVE_MYDIR):
    DRIVE_DIR  = os.path.join(DRIVE_MYDIR, "FWGAN_100ep")
    SAMPLE_DIR = os.path.join(DRIVE_DIR, "samples")
    CKPT_DIR   = os.path.join(DRIVE_DIR, "checkpoints")
    print(f"✅ Google Drive detected. Saving to: {DRIVE_DIR}")
else:
    print("⚠️  Drive not mounted — saving locally (files lost on session end).")
    SAMPLE_DIR = os.path.join(REPO_DIR, "fwgan_samples")
    CKPT_DIR   = os.path.join(REPO_DIR, "checkpoints")

os.makedirs(SAMPLE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)
print(f"  Samples     → {SAMPLE_DIR}")
print(f"  Checkpoints → {CKPT_DIR}")

# ============================================================================ #
# CELL 4 — Replay Buffer & Metrics
# ============================================================================ #

class ReplayBuffer:
    """Stores recent fakes and randomly replays them for discriminator training."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, batch):
        result = []
        for elem in batch:
            if len(self.data) < self.max_size:
                self.data.append(elem)
                result.append(elem)
            elif np.random.rand() > 0.5:
                idx = np.random.randint(0, self.max_size)
                result.append(self.data[idx].clone())
                self.data[idx] = elem
            else:
                result.append(elem)
        return torch.stack(result)


def evaluate_batch_metrics(pred_tensor, target_tensor):
    """Computes PSNR, SSIM, MAE, MSE on a batch of [-1, 1] tensors."""
    pred   = ((pred_tensor   + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    target = ((target_tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()

    psnr_list, ssim_list, mae_list, mse_list = [], [], [], []
    for i in range(pred.shape[0]):
        p_img = np.transpose(pred[i],   (1, 2, 0))
        t_img = np.transpose(target[i], (1, 2, 0))
        psnr_list.append(psnr(t_img, p_img, data_range=1.0))
        ssim_list.append(ssim(t_img, p_img, data_range=1.0, channel_axis=2))
        mae_list.append(np.mean(np.abs(t_img - p_img)))
        mse_list.append(np.mean((t_img - p_img) ** 2))

    return (np.mean(psnr_list), np.mean(ssim_list),
            np.mean(mae_list),  np.mean(mse_list))

# ============================================================================ #
# CELL 5 — Build Model, Optimizers, Schedulers
# ============================================================================ #
print("Initializing FWGAN...")
model = FWGANArchive(input_nc=1, output_nc=3).to(DEVICE)
model.lambda_temp = 0.0   # Disable temporal loss (not sequential video data)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
gen_params   = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
disc_params  = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
print(f"Total trainable params : {total_params:,}")
print(f"  Generator            : {gen_params:,}")
print(f"  Discriminator        : {disc_params:,}")

opt_gen  = optim.Adam(model.generator.parameters(),     lr=GEN_LR,  betas=BETAS)
opt_disc = optim.Adam(model.discriminator.parameters(), lr=DISC_LR, betas=BETAS)

scaler_gen  = GradScaler('cuda', enabled=USE_AMP)
scaler_disc = GradScaler('cuda', enabled=USE_AMP)

fake_buffer = ReplayBuffer(max_size=REPLAY_BUF_SIZE)

# DataLoaders
print("Creating data loaders...")
train_loader = create_dataloader(
    THERMAL_DIR, VISIBLE_DIR,
    mode="paired", is_train=True,
    batch_size=BATCH_SIZE, num_workers=2)
val_loader = create_dataloader(
    THERMAL_DIR, VISIBLE_DIR,
    mode="paired", is_train=False,
    batch_size=max(BATCH_SIZE, 1), num_workers=2)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# LR schedulers — linear decay in the second half of training
steps_per_epoch = len(train_loader)
total_steps     = EPOCHS * steps_per_epoch

def lr_lambda(step):
    decay_start = total_steps // 2
    if step < decay_start:
        return 1.0
    return max(0.0, 1.0 - (step - decay_start) / (total_steps - decay_start))

sched_gen  = optim.lr_scheduler.LambdaLR(opt_gen,  lr_lambda=lr_lambda)
sched_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lr_lambda)

# ============================================================================ #
# CELL 6 — Validation & Sample Saving Functions
# ============================================================================ #

@torch.no_grad()
def run_validation(model, val_loader, epoch, device, use_amp):
    """Full validation pass computing PSNR, SSIM, MAE, RMSE, FPS."""
    model.eval()
    psnr_list, ssim_list, mae_list, mse_list, times = [], [], [], [], []
    prev_pred_val    = None
    prev_thermal_val = None

    for batch in val_loader:
        thermal = batch['thermal'].to(device)
        visible = batch['visible'].to(device)
        B = thermal.shape[0]

        # Handle batch-size shrinkage on last batch
        if prev_thermal_val is not None and prev_thermal_val.shape[0] != B:
            prev_thermal_val = prev_thermal_val[:B]
        if prev_pred_val is not None and prev_pred_val.shape[0] != B:
            prev_pred_val = prev_pred_val[:B]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        with autocast('cuda', enabled=use_amp):
            pred = model.forward_generate(
                thermal,
                prev_thermal_val if prev_thermal_val is not None
                                 else torch.zeros_like(thermal),
                prev_pred_val    if prev_pred_val    is not None
                                 else torch.zeros_like(visible))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.time() - t0) / B)

        prev_pred_val    = pred.detach()
        prev_thermal_val = thermal.detach()

        b_psnr, b_ssim, b_mae, b_mse = evaluate_batch_metrics(pred, visible)
        psnr_list.append(b_psnr)
        ssim_list.append(b_ssim)
        mae_list.append(b_mae)
        mse_list.append(b_mse)

    final_psnr = np.mean(psnr_list)
    final_ssim = np.mean(ssim_list)
    final_mae  = np.mean(mae_list)
    final_rmse = np.sqrt(np.mean(mse_list))
    avg_time   = np.mean(times)
    fps        = 1.0 / avg_time if avg_time > 0 else 0.0

    print(f"\n{'─'*50}")
    print(f"  Epoch {epoch} — Validation Results")
    print(f"{'─'*50}")
    print(f"  PSNR  : {final_psnr:.4f} dB")
    print(f"  SSIM  : {final_ssim:.4f}")
    print(f"  MAE   : {final_mae:.4f}")
    print(f"  RMSE  : {final_rmse:.4f}")
    print(f"  Speed : {avg_time*1000:.2f} ms/img ({fps:.1f} FPS)")
    print(f"{'─'*50}\n")

    return {"psnr": final_psnr, "ssim": final_ssim,
            "mae": final_mae,   "rmse": final_rmse, "fps": fps}


@torch.no_grad()
def save_samples(model, val_loader, epoch, device, use_amp, save_dir,
                 n_samples=4, train_batch=None):
    """
    Saves a grid with TWO sections:
      TOP ROW(S)    — Training batch samples  (thermal | generated | target)
      BOTTOM ROW(S) — Validation samples      (thermal | generated | target)

    Both sections use prev-frame tracking so the generator has context.
    Grid layout:  [Thermal] [Generated] [Ground Truth]  (nrow=3)
    """
    model.eval()

    def make_triplets(thermal_b, visible_b):
        """Generate (thermal, pred, visible) triplet images for one batch."""
        with autocast('cuda', enabled=use_amp):
            pred = model.forward_generate(thermal_b, None, None)
        t_img = (thermal_b[0].cpu() + 1) / 2
        t_img = t_img.repeat(3, 1, 1)
        p_img = (pred[0].cpu().float() + 1) / 2
        v_img = (visible_b[0].cpu() + 1) / 2
        return [t_img.clamp(0,1), p_img.clamp(0,1), v_img.clamp(0,1)]

    all_images = []

    # ── Train samples (from last training batch) ──────────────────────────
    if train_batch is not None:
        thermal_tr = train_batch['thermal'].to(device)[:n_samples]
        visible_tr = train_batch['visible'].to(device)[:n_samples]
        triplets = make_triplets(thermal_tr, visible_tr)
        all_images.extend(triplets)
        # Add a blank separator row (all white) to visually divide train/val
        blank = torch.ones(3, thermal_tr.shape[2], thermal_tr.shape[3])
        all_images.extend([blank, blank, blank])

    # ── Validation samples ────────────────────────────────────────────────
    count          = 0

    for batch in val_loader:
        if count >= n_samples:
            break
        thermal = batch['thermal'].to(device)
        visible = batch['visible'].to(device)

        triplets = make_triplets(thermal, visible)
        all_images.extend(triplets)
        count += 1

    if all_images:
        # Label top-left corner is handled by naming: train rows come first
        grid = make_grid(all_images, nrow=3, padding=4, pad_value=0.8)
        path = os.path.join(save_dir, f"epoch_{epoch:03d}.png")
        save_image(grid, path)
        section = "[Train | ─── | Val]" if train_batch is not None else "[Val only]"
        print(f"  💾 Samples saved {section} → {path}")


# ============================================================================ #
# CELL 7 — Training Loop
# ============================================================================ #
print("\n" + "="*60)
print("  FWGAN Training — 100 Epochs")
print("="*60 + "\n")

history = {"epoch": [], "g_loss": [], "d_loss": [],
           "psnr": [], "ssim": [], "mae": [], "rmse": [], "fps": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch [{epoch:3d}/{EPOCHS}]",
                leave=True, ncols=100)

    prev_pred_f    = None
    prev_target_f  = None
    prev_thermal_f = None
    epoch_g_losses = []
    epoch_d_losses = []

    for batch in loop:
        thermal = batch['thermal'].to(DEVICE)
        visible = batch['visible'].to(DEVICE)

        # ── Step 1: Discriminator ─────────────────────────────────────────
        opt_disc.zero_grad()
        with autocast('cuda', enabled=USE_AMP):
            pred_t = model.forward_generate(
                thermal, prev_thermal_f,
                prev_pred_f.detach() if prev_pred_f is not None else None)

            pred_t_buffered = fake_buffer.push_and_pop(pred_t.detach())

            zeros_th  = torch.zeros_like(thermal)
            zeros_vis = torch.zeros_like(visible)
            if prev_thermal_f is None:
                disc_fake = model.discriminator(
                    thermal, pred_t_buffered, zeros_th, zeros_vis)
                disc_real = model.discriminator(
                    thermal, visible, zeros_th, zeros_vis)
            else:
                disc_fake = model.discriminator(
                    thermal, pred_t_buffered, prev_thermal_f, prev_pred_f.detach())
                disc_real = model.discriminator(
                    thermal, visible, prev_thermal_f, prev_target_f)

            d_loss, _ = model.compute_discriminator_losses(disc_real, disc_fake)

        scaler_disc.scale(d_loss).backward()
        scaler_disc.unscale_(opt_disc)
        torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), GRAD_CLIP)
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # ── Step 2: Generator ─────────────────────────────────────────────
        opt_gen.zero_grad()
        with autocast('cuda', enabled=USE_AMP):
            if prev_thermal_f is None:
                disc_for_gen = model.discriminator(
                    thermal, pred_t, zeros_th, zeros_vis)
            else:
                disc_for_gen = model.discriminator(
                    thermal, pred_t, prev_thermal_f, prev_pred_f.detach())

            g_loss, g_log = model.compute_generator_losses(
                pred_t, visible, prev_pred_f, prev_target_f, disc_for_gen)

        scaler_gen.scale(g_loss).backward()
        scaler_gen.unscale_(opt_gen)
        torch.nn.utils.clip_grad_norm_(model.generator.parameters(), GRAD_CLIP)
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # ── Update previous-frame state ───────────────────────────────────
        with torch.no_grad():
            prev_pred_f = model.forward_generate(
                thermal, prev_thermal_f,
                prev_pred_f.detach() if prev_pred_f is not None else None
            ).detach()
        prev_target_f  = visible.detach()
        prev_thermal_f = thermal.detach()

        epoch_g_losses.append(g_loss.item())
        epoch_d_losses.append(d_loss.item())

        # Step LR schedulers
        sched_gen.step()
        sched_disc.step()

        loop.set_postfix(G=f"{np.mean(epoch_g_losses[-10:]):.3f}",
                         D=f"{np.mean(epoch_d_losses[-10:]):.3f}")

    # ── End of epoch ──────────────────────────────────────────────────────
    avg_g = np.mean(epoch_g_losses)
    avg_d = np.mean(epoch_d_losses)
    print(f"  Epoch {epoch} — G_loss: {avg_g:.4f}, D_loss: {avg_d:.4f}")

    # Run validation
    metrics = run_validation(model, val_loader, epoch, DEVICE, USE_AMP)

    # Record history
    history["epoch"].append(epoch)
    history["g_loss"].append(float(avg_g))
    history["d_loss"].append(float(avg_d))
    for k in ["psnr", "ssim", "mae", "rmse", "fps"]:
        history[k].append(float(metrics[k]))

    # Save samples every SAMPLE_INTERVAL epochs (with last train batch for comparison)
    if epoch % SAMPLE_INTERVAL == 0 or epoch == 1:
        save_samples(model, val_loader, epoch, DEVICE, USE_AMP, SAMPLE_DIR,
                     train_batch=batch)  # 'batch' = last batch of this epoch

    # Save checkpoint every 25 epochs
    if epoch % 25 == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"fwgan_epoch_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  💾 Checkpoint saved → {ckpt_path}")

# Final checkpoint
final_ckpt = os.path.join(CKPT_DIR, "fwgan_final_100ep.pth")
torch.save(model.state_dict(), final_ckpt)
print(f"\n✅ Final model saved → {final_ckpt}")

# Save history as JSON
history_path = os.path.join(REPO_DIR, "fwgan_100ep_history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"📊 Training history saved → {history_path}")


# ============================================================================ #
# CELL 8 — Plot Results
# ============================================================================ #
print("\nGenerating metric plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("FWGAN Training Results — 100 Epochs", fontsize=16, fontweight="bold")

epochs = history["epoch"]

# G/D Loss
ax = axes[0, 0]
ax.plot(epochs, history["g_loss"], label="Generator", color="#e74c3c", linewidth=1.5)
ax.plot(epochs, history["d_loss"], label="Discriminator", color="#3498db", linewidth=1.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("GAN Losses")
ax.legend(); ax.grid(True, alpha=0.3)

# PSNR
ax = axes[0, 1]
ax.plot(epochs, history["psnr"], color="#2ecc71", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("PSNR (dB)"); ax.set_title("PSNR ↑")
ax.grid(True, alpha=0.3)

# SSIM
ax = axes[0, 2]
ax.plot(epochs, history["ssim"], color="#9b59b6", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("SSIM"); ax.set_title("SSIM ↑")
ax.grid(True, alpha=0.3)

# MAE
ax = axes[1, 0]
ax.plot(epochs, history["mae"], color="#e67e22", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("MAE"); ax.set_title("MAE ↓")
ax.grid(True, alpha=0.3)

# RMSE
ax = axes[1, 1]
ax.plot(epochs, history["rmse"], color="#1abc9c", linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_title("RMSE ↓")
ax.grid(True, alpha=0.3)

# Summary Table
ax = axes[1, 2]
ax.axis("off")
final_metrics = {
    "Best PSNR":  f"{max(history['psnr']):.4f} dB (ep {history['psnr'].index(max(history['psnr']))+1})",
    "Best SSIM":  f"{max(history['ssim']):.4f} (ep {history['ssim'].index(max(history['ssim']))+1})",
    "Best MAE":   f"{min(history['mae']):.4f} (ep {history['mae'].index(min(history['mae']))+1})",
    "Best RMSE":  f"{min(history['rmse']):.4f} (ep {history['rmse'].index(min(history['rmse']))+1})",
    "Final PSNR": f"{history['psnr'][-1]:.4f} dB",
    "Final SSIM": f"{history['ssim'][-1]:.4f}",
    "Final MAE":  f"{history['mae'][-1]:.4f}",
    "Final RMSE": f"{history['rmse'][-1]:.4f}",
}
table_text = "\n".join([f"{k}: {v}" for k, v in final_metrics.items()])
ax.text(0.1, 0.5, table_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.set_title("Summary")

plt.tight_layout()
plot_path = os.path.join(REPO_DIR, "fwgan_100ep_metrics.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"📈 Metric plots saved → {plot_path}")

print("\n" + "="*60)
print("  FWGAN 100-Epoch Training Complete!")
print("="*60)
print(f"\nOutputs:")
print(f"  Samples    : {SAMPLE_DIR}/")
print(f"  Checkpoints: {CKPT_DIR}/")
print(f"  History    : {history_path}")
print(f"  Plots      : {plot_path}")
