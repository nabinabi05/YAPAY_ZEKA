import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import create_dataloader
from models.dc_net import DCNet
from models.diffusion_model import ThermalToVisibleDDPM, ConditionalUNet
from models.fwgan import FWGANArchive
from models.vq_infratrans import VQInfraTrans
from train_and_eval import MambaTranslatorProxy

def plot_learning_curves(log_file_path, output_path="learning_curves.png"):
    """
    Parses the terminal output logs from the training run to generate line plots
    showing PSNR and SSIM progression over the 5 epochs for all architectures.
    """
    if not os.path.exists(log_file_path):
        print(f"Log file '{log_file_path}' not found. Please save your terminal output to this file to generate learning curves.")
        return

    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Dictionary to hold metrics: model_name -> {"psnr": [], "ssim": [], "epochs": []}
    metrics = {}
    
    current_epoch = None
    current_model = None

    # Parse logic based on the print statements in train_and_eval.py:
    # --- [Evaluation Results] Epoch 1 Metrics ---
    # | Model     : DCNet
    # | Mean PSNR : 15.1234 dB (Higher reflects greater pixel fidelity)
    # | Mean SSIM : 0.4567 (Closer to 1.0 reflects identical contextual structure)
    for line in lines:
        line = line.strip()
        if "--- [Evaluation Results] Epoch" in line:
            # Extract epoch
            match = re.search(r"Epoch (\d+)", line)
            if match:
                current_epoch = int(match.group(1))
        elif "| Model     :" in line:
            current_model = line.split(":", 1)[1].strip()
            if current_model not in metrics:
                metrics[current_model] = {"psnr": [], "ssim": [], "epochs": []}
        elif "| Mean PSNR :" in line:
            match = re.search(r"Mean PSNR :\s*([\d.]+)", line)
            if match and current_model and current_epoch is not None:
                psnr_val = float(match.group(1))
                metrics[current_model]["psnr"].append(psnr_val)
                metrics[current_model]["epochs"].append(current_epoch)
        elif "| Mean SSIM :" in line:
            match = re.search(r"Mean SSIM :\s*([\d.]+)", line)
            if match and current_model and current_epoch is not None:
                ssim_val = float(match.group(1))
                metrics[current_model]["ssim"].append(ssim_val)

    if not metrics:
        print("No metrics found in the log file.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, data in metrics.items():
        epochs = data["epochs"]
        if not epochs:
            continue
        ax1.plot(epochs, data["psnr"], marker='o', label=model_name)
        ax2.plot(epochs, data["ssim"], marker='o', label=model_name)

    ax1.set_title("PSNR Progression (Learning Curve)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR (dB)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks(range(1, 6))

    ax2.set_title("SSIM Progression (Learning Curve)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.legend()
    ax2.grid(True)
    ax2.set_xticks(range(1, 6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Learning curves saved to {output_path}")

def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for visualization."""
    return ((tensor + 1.0) / 2.0).clamp(0, 1).cpu()

@torch.no_grad()
def generate_image_grid(thermal_dir, visible_dir, output_path="inference_grid.png"):
    """
    Takes a single batch from the validation dataset and runs inference across all 5 trained models.
    Saves a high-resolution 1x7 image grid.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for inference")

    # Load 1 batch from validation dataset
    val_loader = create_dataloader(thermal_dir, visible_dir, mode="paired", is_train=False, batch_size=1)
    
    batch = next(iter(val_loader))
    thermal = batch['thermal'].to(device)
    visible = batch['visible'].to(device)

    # Initialize all models
    print("Initializing models...")
    models = {}
    
    models['DCNet'] = DCNet(input_nc=1, output_nc=3).to(device)
    models['FWGAN'] = FWGANArchive(input_nc=1, output_nc=3).to(device)
    models['VQ-InfraTrans'] = VQInfraTrans(input_nc=1, output_nc=3).to(device)
    models['Inter-Mamba'] = MambaTranslatorProxy().to(device)
    
    unet = ConditionalUNet(c_in=4, c_out=3)
    models['Cond-DDPM'] = ThermalToVisibleDDPM(network=unet, T=100).to(device)

    # Note: Models are instantiated here. If trained weights (.pth) were saved, 
    # they would be loaded using model.load_state_dict(). Since the training orchestrator 
    # evaluated in-memory without saving checkpoints locally, we evaluate them as currently initialized.

    for name, model in models.items():
        checkpoint_path = os.path.join("checkpoints", f"{name}_final.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading trained weights for {name} from {checkpoint_path}...")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Warning: No saved weights found for {name} at {checkpoint_path}. Using untrained initialization.")
        model.eval()

    print("Running inference across all architectures...")
    
    outputs = {}
    
    # 1. DCNet
    outputs['DCNet'] = models['DCNet'](thermal)
    
    # 2. FWGAN
    outputs['FWGAN'] = models['FWGAN'].forward_generate(thermal, thermal, torch.zeros_like(visible).to(device))
    
    # 3. VQ-InfraTrans
    pred_vq, _ = models['VQ-InfraTrans'](thermal)
    outputs['VQ-InfraTrans'] = pred_vq
    
    # 4. Inter-Mamba
    outputs['Inter-Mamba'] = models['Inter-Mamba'](thermal)
    
    # 5. Cond-DDPM
    outputs['Cond-DDPM'] = models['Cond-DDPM'].sample(thermal, shape=(thermal.shape[0], 3, thermal.shape[2], thermal.shape[3]))

    # Plotting the 1x7 grid
    print("Plotting high-resolution image grid...")
    titles = ['Thermal Input', 'Visible GT', 'DCNet Output', 'FWGAN Output', 'VQ-InfraTrans Output', 'Inter-Mamba Output', 'Cond-DDPM Output']
    
    fig, axes = plt.subplots(1, 7, figsize=(28, 4))
    
    # Denormalize and convert to numpy for plotting
    img_thermal = denormalize(thermal)[0].permute(1, 2, 0).numpy()
    img_visible = denormalize(visible)[0].permute(1, 2, 0).numpy()
    
    # Plot Inputs
    axes[0].imshow(img_thermal, cmap='gray')
    axes[1].imshow(img_visible)
    
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].axis('off')

    # Plot Model Outputs
    model_names = ['DCNet', 'FWGAN', 'VQ-InfraTrans', 'Inter-Mamba', 'Cond-DDPM']
    for idx, name in enumerate(model_names):
        ax = axes[idx + 2]
        pred_img = denormalize(outputs[name])[0].permute(1, 2, 0).numpy()
        ax.imshow(pred_img)
        ax.set_title(titles[idx + 2], fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"High-resolution grid successfully saved to {output_path}")

if __name__ == '__main__':
    print("="*50)
    print("Visualizing Quantitative and Qualitative Results")
    print("="*50)
    
    # Datasets
    THERMAL_DIR = os.path.join("data", "LLVIP", "LLVIP", "infrared")
    VISIBLE_DIR = os.path.join("data", "LLVIP", "LLVIP", "visible")
    
    # Expected log file path containing terminal outputs from previous run
    LOG_FILE = "training_logs.txt"
    
    # 1. Parse logs and plot learning curves
    plot_learning_curves(LOG_FILE)
    
    # 2. Run inference and plot image grid
    if os.path.exists(THERMAL_DIR) and os.path.exists(VISIBLE_DIR):
        generate_image_grid(THERMAL_DIR, VISIBLE_DIR)
    else:
        print(f"\nWarning: Dataset paths not found at {THERMAL_DIR} or {VISIBLE_DIR}.")
        print("Cannot run inference to generate the image grid. Please download the datasets first.")
