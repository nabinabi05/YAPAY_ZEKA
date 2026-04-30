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

## Repository Structure

- `models/`: Architecture definitions for all five models.
- `data/`: Dataset loading and preprocessing utilities.
- `utils/`: Helper functions.
- `train_and_eval.py`: The main entry point for training and benchmarking.
- `checkpoints/`: (Ignored) Model weights are saved here after training.
- `samples/`: (Ignored) Visual comparison grids.

## License

[Add License Here]
