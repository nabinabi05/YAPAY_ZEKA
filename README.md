# Thermal-to-Visible Image Translation Benchmark

This repository contains a unified framework for benchmarking five state-of-the-art generative architectures for Thermal-to-Visible (Infrared-to-Visible) image translation on the **LLVIP** dataset.

## Supported Architectures

1.  **DCNet**: A Dual-Constraint Network utilizing PatchNCE and Perceptual Guidance.
2.  **FWGAN**: A Flow-based Wait GAN for stable adversarial translation.
3.  **VQ-InfraTrans**: A Vector Quantized architecture for discrete representation learning in thermal translation.
4.  **Inter-Mamba**: A multi-scale fusion block based on Selective Structured State Spaces (Mamba).
5.  **Cond-DDPM**: A Conditional Denoising Diffusion Probabilistic Model with Min-SNR weighting.

## Features

- **Unified Training Orchestrator**: A single script (`train_and_eval.py`) to train and evaluate all models with consistent hyper-parameters.
- **Metric Suite**: Automatic computation of **PSNR**, **SSIM**, **MAE**, and **RMSE**.
- **Performance Tracking**: Logs inference speed (FPS) and trainable parameter counts.
- **Visual Validation**: Generates side-by-side comparisons (Thermal | Generated | Ground Truth) during training.
- **Automated Recovery**: Can resume evaluation from checkpoints if training results are missing.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (with CUDA support)
- Scikit-Image
- Tqdm
- Inter-Mamba dependencies (if applicable)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Structure

The framework expects the LLVIP dataset in the following structure:
```text
data/
└── LLVIP/
    └── LLVIP/
        ├── infrared/ (Thermal images)
        └── visible/  (Ground truth images)
```

### Usage

To start the benchmarking process for all models:
```bash
python train_and_eval.py
```

Results will be saved to `results.json` and visual samples will be stored in the `samples/` directory.

### Running on Google Colab

The script reads dataset and output paths from environment variables, so a Drive-mounted dataset works without code changes. Tuned for an A100 (40 GB) instance; reduce batch sizes in `train_and_eval.py` if you fall back to T4/V100.

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone the repo and install dependencies
!git clone <YOUR_REPO_URL> /content/yapay_zeka
%cd /content/yapay_zeka
!pip install -r requirements.txt

# 3. Point at your Drive-mounted dataset and a persistent checkpoint dir
%env LLVIP_ROOT=/content/drive/MyDrive/LLVIP
%env CKPT_DIR=/content/drive/MyDrive/yapay_zeka_checkpoints
%env RESULTS_PATH=/content/drive/MyDrive/yapay_zeka_checkpoints/results.json

# 4. Train all five models for 50 epochs each
!python train_and_eval.py
```

`LLVIP_ROOT` must contain `infrared/` and `visible/` subfolders. Alternatively, set `THERMAL_DIR` and `VISIBLE_DIR` independently.

If a Colab session disconnects mid-run, just re-run the script: completed models are skipped (checkpoint + JSON entry both present) and crash-recovered models reload weights and re-evaluate without retraining.

## Repository Structure

- `models/`: Architecture definitions for all five models.
- `data/`: Dataset loading and preprocessing utilities.
- `utils/`: Helper functions.
- `train_and_eval.py`: The main entry point for training and benchmarking.
- `checkpoints/`: (Ignored) Model weights are saved here after training.
- `samples/`: (Ignored) Visual comparison grids.

## License

[Add License Here]
