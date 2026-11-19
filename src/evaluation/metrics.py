from typing import Dict
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)


@torch.no_grad()
def compute_reconstruction_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    # Computes average pixel-wise MSE, PSNR (dB), and SSIM over image tensors in [0, 1].
    # Expected tensor shape: (N, 1, 28, 28).
    preds = torch.clamp(preds, 0.0, 1.0)
    targets = torch.clamp(targets, 0.0, 1.0)

    mse = F.mse_loss(preds, targets).item()
    psnr = peak_signal_noise_ratio(preds, targets, data_range=1.0).item()
    ssim = structural_similarity_index_measure(preds, targets, data_range=1.0).item()

    return {
        "mse": round(mse, 6),
        "psnr": round(psnr, 4),
        "ssim": round(ssim, 4),
    }