import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = False

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.utils import make_grid, save_image

from data.dataset import create_dataloader
from models.dc_net import DCNet
from models.diffusion_model import ThermalToVisibleDDPM, ConditionalUNet
from models.fwgan import FWGANArchive
from models.vq_infratrans import VQInfraTrans
from models.mamba_fusion import InterMambaBlock


class ReplayBuffer:
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


# --------------------------------------------------------------------------- #
# Mamba proxy wrapper
# --------------------------------------------------------------------------- #

class MambaTranslatorProxy(nn.Module):
    """
    Multi-scale encoder-decoder wrapping the InterMambaBlock.

    FIX: Expanded from 91K to ~2M parameters with a proper 3-level
    encoder-decoder and skip connections.  The SSM block operates at the
    bottleneck resolution (H/4, W/4) for efficiency.

    WARNING: Both 'visible' and 'thermal' SSM inputs are still derived from
    the same thermal encoder.  A proper cross-modal implementation would
    require separate encoder streams.
    """
    def __init__(self):
        super().__init__()
        # Multi-scale encoder: 1 → 64 → 128 → 256
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True), nn.SiLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True), nn.SiLU())
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True), nn.SiLU())

        # SSM fusion at bottleneck
        self.mamba = InterMambaBlock(dim=256)

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True), nn.SiLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True), nn.SiLU())
        self.out = nn.Sequential(
            nn.Conv2d(128, 3, 3, padding=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        m  = self.mamba(visible=e3, thermal=e3)
        d1 = self.dec1(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d1, e2], dim=1))
        return self.out(torch.cat([d2, e1], dim=1))


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def evaluate_batch_metrics(pred_tensor: torch.Tensor,
                            target_tensor: torch.Tensor):
    """
    Computes PSNR, SSIM, MAE, and MSE on a batch of [-1, 1] tensors.
    Returns raw MSE — RMSE is computed at epoch level as sqrt(mean(all_mse)).
    """
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

    return (np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(mae_list),
            np.mean(mse_list))


# --------------------------------------------------------------------------- #
# Unified trainer
# --------------------------------------------------------------------------- #

class UnifiedModelTrainer:
    """
    Unified training and evaluation pipeline for all five architectures.
    """

    def __init__(self, model_name: str, thermal_dir: str, visible_dir: str,
                 batch_size: int = 4, epochs: int = 5, device: str = "cuda"):
        self.device     = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.epochs     = epochs
        self.use_amp    = self.device.type == "cuda"

        print(f"Initializing DataLoader hooks for benchmark task: {model_name}...")

        # pin_memory, num_workers, and persistent_workers are handled
        # internally by create_dataloader / dataset.py.
        self.train_loader = create_dataloader(
            thermal_dir, visible_dir,
            mode="paired", is_train=True,
            batch_size=batch_size)
        self.val_loader = create_dataloader(
            thermal_dir, visible_dir,
            mode="paired", is_train=False,
            batch_size=max(batch_size, 1))

        self._init_model()

    # ---------------------------------------------------------------------- #
    # Model + optimiser initialisation
    # ---------------------------------------------------------------------- #

    def _init_model(self):
        print(f"Mounting [{self.model_name}] architecture to Device: {self.device}")

        self.opts    = {}
        self.scalers = {}
        GAN_BETAS    = (0.5, 0.999)

        if self.model_name == "DCNet":
            self.model = DCNet(input_nc=1, output_nc=3).to(self.device)
            self.opts['main']   = optim.Adam(self.model.parameters(), lr=2e-4)
            self.base_criterion = nn.L1Loss()

        elif self.model_name == "Cond-DDPM":
            unet       = ConditionalUNet(c_in=4, c_out=3)
            self.model = ThermalToVisibleDDPM(network=unet, T=1000, schedule='cosine').to(self.device)
            self.opts['main']   = optim.Adam(self.model.parameters(), lr=2e-4)
            self.base_criterion = nn.MSELoss()

        elif self.model_name == "FWGAN":
            self.model = FWGANArchive(input_nc=1, output_nc=3).to(self.device)
            self.model.lambda_temp = 0.0  # ADD THIS LINE — disables temporal loss
            self.opts['gen']  = optim.Adam(
                self.model.generator.parameters(),     lr=2e-4, betas=(0.5, 0.999))
            self.opts['disc'] = optim.Adam(
                self.model.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))  # TTUR
            self.fake_buffer = ReplayBuffer(max_size=50)

        elif self.model_name == "VQ-InfraTrans":
            self.model = VQInfraTrans(input_nc=1, output_nc=3).to(self.device)
            self.opts['main']   = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))
            self.base_criterion = nn.L1Loss()

        elif self.model_name == "Inter-Mamba":
            self.model = MambaTranslatorProxy().to(self.device)
            self.opts['main']   = optim.Adam(self.model.parameters(), lr=2e-4)
            self.base_criterion = nn.L1Loss()

        else:
            raise ValueError(f"Unknown model name: '{self.model_name}'")

        for key in self.opts:
            self.scalers[key] = GradScaler('cuda', enabled=self.use_amp)

        self.schedulers = {}
        self._step_count = 0
        steps_per_epoch = len(self.train_loader)
        total_steps = self.epochs * steps_per_epoch
        for key, opt in self.opts.items():
            def lr_lambda(step, ts=total_steps):
                decay_start = ts // 2
                if step < decay_start:
                    return 1.0
                return max(0.0, 1.0 - (step - decay_start) / (ts - decay_start))
            self.schedulers[key] = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    def load_checkpoint(self, ckpt_path: str):
        """Loads saved weights into the model (used for validation-only recovery)."""
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device))
        print(f"[LOADED] Weights restored from '{ckpt_path}'")

    # ---------------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------------- #

    def run_training_loop(self) -> dict:
        history = {"psnr": [], "ssim": [], "mae": [], "rmse": [], "fps": []}

        total_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        print(f"[{self.model_name}] Total Trainable Parameters: {total_params:,}")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch [{epoch}/{self.epochs}]")

            prev_pred_f    = None
            prev_target_f  = None
            prev_thermal_f = None
            epoch_losses   = []

            for batch in loop:
                thermal = batch['thermal'].to(self.device)
                visible = batch['visible'].to(self.device)

                loss_val = self._training_step(
                    thermal, visible,
                    prev_pred_f, prev_target_f, prev_thermal_f)

                if self.model_name == "FWGAN":
                    with torch.no_grad():
                        prev_pred_f = self.model.forward_generate(
                            thermal,
                            prev_thermal_f,
                            prev_pred_f.detach() if prev_pred_f is not None else None
                        ).detach()
                    prev_target_f  = visible.detach()
                    prev_thermal_f = thermal.detach()

                epoch_losses.append(loss_val)

                for sched in self.schedulers.values():
                    sched.step()
                self._step_count += 1

                postfix = {"loss": f"{np.mean(epoch_losses[-10:]):.3f}"}
                if hasattr(self, '_last_perplexity'):
                    postfix["ppl"] = f"{self._last_perplexity:.0f}"
                loop.set_postfix(**postfix)

            metrics = self._run_validation(epoch)
            self._save_samples(epoch)
            for k, v in metrics.items():
                history[k].append(float(v))

        history["total_parameters"] = total_params
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", f"{self.model_name}_final.pth")
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"Saved weights -> {ckpt_path}")

        return history

    def run_validation_only(self) -> dict:
        """
        Runs a single validation pass on a pre-loaded checkpoint.
        Used to recover metrics when a checkpoint exists but results.json
        has no entry for this model (i.e. training completed but the run
        crashed before writing JSON).
        """
        total_params = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad)
        print(f"[{self.model_name}] Total Trainable Parameters: {total_params:,}")
        metrics = self._run_validation(epoch=self.epochs)
        history = {
            "psnr":             [float(metrics["psnr"])],
            "ssim":             [float(metrics["ssim"])],
            "mae":              [float(metrics["mae"])],
            "rmse":             [float(metrics["rmse"])],
            "fps":              [float(metrics["fps"])],
            "total_parameters": total_params,
            "note":             "metrics recovered from checkpoint — no per-epoch history",
        }
        return history

    # ---------------------------------------------------------------------- #
    # Per-batch training step
    # ---------------------------------------------------------------------- #

    def _training_step(self, thermal, visible,
                       prev_pred_f, prev_target_f, prev_thermal_f) -> float:

        if self.model_name == "FWGAN":
            # Step 1: Discriminator
            self.opts['disc'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                pred_t    = self.model.forward_generate(
                    thermal, prev_thermal_f,
                    prev_pred_f.detach() if prev_pred_f is not None else None)
                
                pred_t_buffered = self.fake_buffer.push_and_pop(pred_t.detach())
                
                zeros_th  = torch.zeros_like(thermal)
                zeros_vis = torch.zeros_like(visible)
                if prev_thermal_f is None:
                    disc_fake = self.model.discriminator(
                        thermal, pred_t_buffered, zeros_th, zeros_vis)
                    disc_real = self.model.discriminator(
                        thermal, visible, zeros_th, zeros_vis)
                else:
                    disc_fake = self.model.discriminator(
                        thermal, pred_t_buffered, prev_thermal_f, prev_pred_f.detach())
                    disc_real = self.model.discriminator(
                        thermal, visible, prev_thermal_f, prev_target_f)
                d_loss, _ = self.model.compute_discriminator_losses(disc_real, disc_fake)

            self.scalers['disc'].scale(d_loss).backward()
            self.scalers['disc'].unscale_(self.opts['disc'])
            torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), 1.0)
            self.scalers['disc'].step(self.opts['disc'])
            self.scalers['disc'].update()

            # Step 2: Generator
            self.opts['gen'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                if prev_thermal_f is None:
                    disc_for_gen = self.model.discriminator(
                        thermal, pred_t, zeros_th, zeros_vis)
                else:
                    disc_for_gen = self.model.discriminator(
                        thermal, pred_t, prev_thermal_f, prev_pred_f.detach())
                g_loss, _ = self.model.compute_generator_losses(
                    pred_t, visible, prev_pred_f, prev_target_f, disc_for_gen)

            self.scalers['gen'].scale(g_loss).backward()
            self.scalers['gen'].unscale_(self.opts['gen'])
            torch.nn.utils.clip_grad_norm_(self.model.generator.parameters(), 1.0)
            self.scalers['gen'].step(self.opts['gen'])
            self.scalers['gen'].update()
            return g_loss.item()

        elif self.model_name == "Cond-DDPM":
            self.opts['main'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                true_noise, pred_noise, snr_weights = self.model(visible, thermal)
                # Min-SNR weighted MSE — prevents gradient conflicts across timesteps
                loss = (snr_weights * F.mse_loss(pred_noise, true_noise, reduction='none')).mean()
            self.scalers['main'].scale(loss).backward()
            self.scalers['main'].unscale_(self.opts['main'])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scalers['main'].step(self.opts['main'])
            self.scalers['main'].update()
            return loss.item()

        elif self.model_name == "VQ-InfraTrans":
            self.opts['main'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                pred, vq_loss, perplexity = self.model(thermal)
                loss = self.base_criterion(pred, visible) + vq_loss
            self.scalers['main'].scale(loss).backward()
            self.scalers['main'].unscale_(self.opts['main'])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scalers['main'].step(self.opts['main'])
            self.scalers['main'].update()
            self._last_perplexity = perplexity.item()
            return loss.item()

        elif self.model_name == "DCNet":
            self.opts['main'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                # Forward: thermal -> generated visible + encoder features
                pred, src_feats, _ = self.model(thermal, extract_features=True)
                l1_loss = self.base_criterion(pred, visible)

                # PatchNCE: pass generated image BACK through encoder
                # Convert 3ch pred to 1ch grayscale to match encoder input_nc=1
                pred_gray = pred.mean(dim=1, keepdim=True)
                _, gen_feats, _ = self.model(pred_gray, extract_features=True)

                # Compare source thermal features vs generated image features
                # at same spatial positions — this is what PatchNCE does
                patch_loss = sum(F.l1_loss(sf, gf.detach())
                                 for sf, gf in zip(src_feats, gen_feats))

                # Perceptual: generated vs real visible in VGG space
                target_vgg = self.model.perceptual_guidance(visible)
                pred_vgg = self.model.perceptual_guidance(pred)
                perc_loss = sum(F.l1_loss(pv, tv.detach())
                                for pv, tv in zip(pred_vgg, target_vgg))

                loss = l1_loss + patch_loss + 10.0 * perc_loss

            self.scalers['main'].scale(loss).backward()
            self.scalers['main'].unscale_(self.opts['main'])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scalers['main'].step(self.opts['main'])
            self.scalers['main'].update()
            return loss.item()

        else:  # Inter-Mamba proxy
            self.opts['main'].zero_grad()
            with autocast('cuda', enabled=self.use_amp):
                pred = self.model(thermal)
                loss = self.base_criterion(pred, visible)
            self.scalers['main'].scale(loss).backward()
            self.scalers['main'].unscale_(self.opts['main'])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scalers['main'].step(self.opts['main'])
            self.scalers['main'].update()
            return loss.item()

    # ---------------------------------------------------------------------- #
    # Validation pass
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def _run_validation(self, epoch: int) -> dict:
        self.model.eval()
        print(f"\n--- Running Evaluation Suite for Epoch {epoch} ---")

        psnr_list, ssim_list, mae_list, mse_list, inference_times = [], [], [], [], []
        prev_pred_val    = None
        prev_thermal_val = None

        for batch in self.val_loader:
            thermal = batch['thermal'].to(self.device)
            visible = batch['visible'].to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            with autocast('cuda', enabled=self.use_amp):
                if self.model_name == "Cond-DDPM":
                    pred = self.model.sample_ddim(
                        thermal,
                        shape=(thermal.shape[0], 3, thermal.shape[2], thermal.shape[3]),
                        ddim_steps=50)

                elif self.model_name == "FWGAN":
                    B = thermal.shape[0]
                    # Guard against batch-size shrinkage on the last batch:
                    # slice stored buffers down to B, or zero-init if first batch.
                    if prev_thermal_val is not None and prev_thermal_val.shape[0] != B:
                        prev_thermal_val = prev_thermal_val[:B]
                    if prev_pred_val is not None and prev_pred_val.shape[0] != B:
                        prev_pred_val = prev_pred_val[:B]

                    pred = self.model.forward_generate(
                        thermal,
                        prev_thermal_val if prev_thermal_val is not None
                                        else torch.zeros_like(thermal),
                        prev_pred_val    if prev_pred_val    is not None
                                        else torch.zeros_like(visible))
                    prev_pred_val    = pred.detach()
                    prev_thermal_val = thermal.detach()

                elif self.model_name == "VQ-InfraTrans":
                    pred, _, _ = self.model(thermal)

                elif self.model_name == "DCNet":
                    pred = self.model(thermal, extract_features=False)

                else:
                    pred = self.model(thermal)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append((time.time() - t0) / thermal.shape[0])

            b_psnr, b_ssim, b_mae, b_mse = evaluate_batch_metrics(pred, visible)
            psnr_list.append(b_psnr)
            ssim_list.append(b_ssim)
            mae_list.append(b_mae)
            mse_list.append(b_mse)

        final_psnr = np.mean(psnr_list)
        final_ssim = np.mean(ssim_list)
        final_mae  = np.mean(mae_list)
        final_rmse = np.sqrt(np.mean(mse_list))
        avg_time   = np.mean(inference_times)
        fps        = 1.0 / avg_time if avg_time > 0 else 0.0

        print(f"--- [Evaluation Results] Epoch {epoch} Metrics ---")
        print(f"| Model      : {self.model_name}")
        print(f"| Mean PSNR  : {final_psnr:.4f} dB (Higher reflects greater pixel fidelity)")
        print(f"| Mean SSIM  : {final_ssim:.4f} (Closer to 1.0 reflects identical contextual structure)")
        print(f"| Mean MAE   : {final_mae:.4f} (Lower is better)")
        print(f"| Mean RMSE  : {final_rmse:.4f} (Lower is better)")
        print(f"| Inference  : {avg_time*1000:.2f} ms/image ({fps:.2f} FPS)")
        print("--------------------------------------------------\n")

        return {"psnr": final_psnr, "ssim": final_ssim,
                "mae": final_mae,   "rmse": final_rmse, "fps": fps}

    def _save_samples(self, epoch):
        """Save visual samples: thermal | generated | ground truth for first 4 val images."""
        self.model.eval()
        save_dir = os.path.join("samples", self.model_name)
        os.makedirs(save_dir, exist_ok=True)
        images = []
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                if count >= 4:
                    break
                thermal = batch['thermal'].to(self.device)
                visible = batch['visible'].to(self.device)
                with autocast('cuda', enabled=self.use_amp):
                    if self.model_name == "Cond-DDPM":
                        pred = self.model.sample_ddim(thermal, shape=(thermal.shape[0], 3, thermal.shape[2], thermal.shape[3]), ddim_steps=50)
                    elif self.model_name == "FWGAN":
                        pred = self.model.forward_generate(thermal)
                    elif self.model_name == "VQ-InfraTrans":
                        pred, _, _ = self.model(thermal)
                    elif self.model_name == "DCNet":
                        pred = self.model(thermal, extract_features=False)
                    else:
                        pred = self.model(thermal)
                # Take first image from batch
                t_img = (thermal[0].cpu() + 1) / 2  # (1, H, W) -> [0,1]
                t_img = t_img.repeat(3, 1, 1)       # grayscale to 3ch
                p_img = (pred[0].cpu().float() + 1) / 2
                v_img = (visible[0].cpu() + 1) / 2
                images.extend([t_img.clamp(0,1), p_img.clamp(0,1), v_img.clamp(0,1)])
                count += 1
        if images:
            grid = make_grid(images, nrow=3, padding=2)
            save_image(grid, os.path.join(save_dir, f"epoch_{epoch:02d}.png"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    print("Initiating Unified Generative Framework Training Orchestrator...")

    THERMAL_DIR  = os.path.join("data", "LLVIP", "LLVIP", "infrared")
    VISIBLE_DIR  = os.path.join("data", "LLVIP", "LLVIP", "visible")
    RESULTS_PATH = "results.json"
    CKPT_DIR     = "checkpoints"

    if not os.path.exists(THERMAL_DIR) or not os.path.exists(VISIBLE_DIR):
        print(f"Error: Dataset paths missing.\n"
              f"  Thermal : {THERMAL_DIR}\n"
              f"  Visible : {VISIBLE_DIR}")
        exit(1)

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load any results already saved from a previous run
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)
        print(f"Loaded existing results for: {list(benchmark_results.keys())}")
    else:
        benchmark_results = {}

    model_configs = {
        "DCNet":         {"batch_size": 32, "epochs": 5},
        "FWGAN":         {"batch_size": 128, "epochs": 5},
        "VQ-InfraTrans": {"batch_size": 120, "epochs": 5},
        "Inter-Mamba":   {"batch_size": 32,  "epochs": 5},
        "FWGAN":         {"batch_size": 128, "epochs": 5},
        "VQ-InfraTrans": {"batch_size": 120, "epochs": 5},
        "Inter-Mamba":   {"batch_size": 32,  "epochs": 5},
        "Cond-DDPM":     {"batch_size": 64, items():
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_final.pth")

        trainer = UnifiedModelTrainer(
            model_name=model_name,
            thermal_dir=THERMAL_DIR,
            visible_dir=VISIBLE_DIR,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        if os.path.exists(ckpt_path) and model_name in benchmark_results:
            # Checkpoint AND results both exist — skip entirely
            print(f"\n[SKIP] {model_name} — checkpoint and results already saved. "
                  f"Delete '{ckpt_path}' to force a retrain.")
            continue

        elif os.path.exists(ckpt_path) and model_name not in benchmark_results:
            # Checkpoint exists but JSON entry is missing — training completed
            # but the run crashed before writing results. Recover by loading
            # weights and running one validation pass instead of retraining.
            print(f"\n[RECOVER] {model_name} — checkpoint found but no JSON entry. "
                  f"Loading weights and running validation to recover metrics...")
            trainer.load_checkpoint(ckpt_path)
            history = trainer.run_validation_only()

        else:
            # No checkpoint — train from scratch
            print(f"\n{'='*50}\nStarting Automated Pipeline for: {model_name}\n{'='*50}")
            history = trainer.run_training_loop()

        benchmark_results[model_name] = history

        # Write after every model so a crash mid-run loses nothing
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=4)
        print(f"[SAVED] '{model_name}' results written to '{RESULTS_PATH}'.")

    print("\nAll architecture benchmark loops complete.")
    print(f"Final metrics in '{RESULTS_PATH}'.")

    # Generate final comparison grid
    print("Generating final comparison grid...")
    os.makedirs("samples", exist_ok=True)
    comparison_images = []
    # Get 6 test images
    test_loader = create_dataloader(THERMAL_DIR, VISIBLE_DIR, mode="paired", is_train=False, batch_size=1)
    test_samples = []
    for i, batch in enumerate(test_loader):
        if i >= 6:
            break
        test_samples.append(batch)

    model_names_ordered = ["DCNet", "FWGAN", "VQ-InfraTrans", "Inter-Mamba", "Cond-DDPM"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for sample in test_samples:
        thermal = sample['thermal'].to(device)
        visible = sample['visible'].to(device)
        row = [(thermal[0].cpu() + 1) / 2]
        row[0] = row[0].repeat(3, 1, 1) if row[0].shape[0] == 1 else row[0]

        for mn in model_names_ordered:
            ckpt = os.path.join(CKPT_DIR, f"{mn}_final.pth")
            if not os.path.exists(ckpt):
                row.append(torch.zeros(3, 256, 256))
                continue
            # Create a temporary trainer just to load the model
            tmp = UnifiedModelTrainer(mn, THERMAL_DIR, VISIBLE_DIR, batch_size=1, epochs=1, device="cuda")
            tmp.load_checkpoint(ckpt)
            tmp.model.eval()
            with torch.no_grad(), autocast('cuda', enabled=True):
                if mn == "Cond-DDPM":
                    p = tmp.model.sample_ddim(thermal, shape=(1,3,256,256), ddim_steps=50)
                elif mn == "FWGAN":
                    p = tmp.model.forward_generate(thermal)
                elif mn == "VQ-InfraTrans":
                    p, _, _ = tmp.model(thermal)
                elif mn == "DCNet":
                    p = tmp.model(thermal, extract_features=False)
                else:
                    p = tmp.model(thermal)
            row.append((p[0].cpu().float() + 1) / 2)
            del tmp

        row.append((visible[0].cpu() + 1) / 2)
        comparison_images.extend([img.clamp(0, 1) for img in row])

    if comparison_images:
        grid = make_grid(comparison_images, nrow=7, padding=4)
        save_image(grid, "samples/final_comparison.png")
        print("Saved samples/final_comparison.png")
