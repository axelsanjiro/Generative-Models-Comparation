from typing import Dict, Callable, Tuple
import torch
import torch.nn.functional as F
import time
import torch.nn as nn
from torch.utils.data import DataLoader

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


@torch.no_grad()
def measure_inference_latency_ms(
    forward_fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup_runs: int = 20,
    timed_runs: int = 100,
) -> float:
    # Measures average forward-pass inference latency in milliseconds
    # for a standardized batch of 64 samples.

    for _ in range(warmup_runs):
        _ = forward_fn()

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_t = time.perf_counter()
    for _ in range(timed_runs):
        _ = forward_fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_t = time.perf_counter()

    avg_ms = ((end_t - start_t) / timed_runs) * 1000.0
    return round(avg_ms, 4)


@torch.no_grad()
def evaluate_ae_on_testset(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_vae: bool = False,
) -> Tuple[Dict[str, float], float]:
    # Evaluates reconstruction metrics (MSE, PSNR, SSIM), latent sparsity ratio,
    # and batch-64 inference latency on the Test Set.
    
    model.to(device).eval()
    all_x, all_recon, all_z = [], [], []

    for x, _ in test_loader:
        x = x.to(device)
        if is_vae:
            x_recon, _, _, z = model(x)
        else:
            x_recon, z = model(x)

        all_x.append(x.cpu())
        all_recon.append(x_recon.cpu())
        all_z.append(z.cpu())

    test_x = torch.cat(all_x, dim=0)
    test_recon = torch.cat(all_recon, dim=0)
    test_z = torch.cat(all_z, dim=0)

    metrics = compute_reconstruction_metrics(test_recon, test_x)
    metrics["latent_zero_ratio"] = round((test_z.abs() < 1e-3).float().mean().item(), 4)

    # Measure batch-64 inference latency
    dummy_batch = torch.randn((64, 1, 28, 28), device=device)
    latency_ms = measure_inference_latency_ms(lambda: model(dummy_batch), device=device)

    return metrics, latency_ms