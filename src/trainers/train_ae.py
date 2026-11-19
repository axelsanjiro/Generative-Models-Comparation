import time
from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.vae import VariationalAutoencoder, vae_loss_function

def train_deterministic_ae(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    l1_lambda: float = 0.0,
    model_name: str = "Vanilla AE",
) -> Dict[str, object]:

    # Trains Vanilla Autoencoder (l1_lambda=0.0) or Sparse Autoencoder (l1_lambda > 0.0)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, epochs + 1):
        start_t = time.perf_counter()
        model.train()
        running_loss, running_mse, total = 0.0, 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, z = model(x)

            mse = F.mse_loss(x_recon, x)
            sparsity_penalty = l1_lambda * torch.mean(torch.abs(z)) if l1_lambda > 0 else 0.0
            loss = mse + sparsity_penalty

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_mse += mse.item() * bs
            total += bs

        train_loss = running_loss / total
        train_mse = running_mse / total

        # Validation
        model.eval()
        val_loss_sum, val_mse_sum, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, z = model(x)
                mse = F.mse_loss(x_recon, x)
                sparsity_penalty = l1_lambda * torch.mean(torch.abs(z)) if l1_lambda > 0 else 0.0
                loss = mse + sparsity_penalty

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_mse_sum += mse.item() * bs
                val_total += bs

        val_loss = val_loss_sum / val_total
        val_mse = val_mse_sum / val_total
        elapsed = time.perf_counter() - start_t

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["epoch_time_sec"].append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.6f} (MSE: {train_mse:.6f}) | "
                f"Val MSE: {val_mse:.6f} | Time: {elapsed:.2f}s"
            )

    avg_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
    return {"history": history, "avg_epoch_time_sec": round(avg_epoch_time, 4)}


def train_vae(
    model: VariationalAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    kl_weight: float,
    device: torch.device,
) -> Dict[str, object]:
    # Trains Variational Autoencoder
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_bce": [],
        "train_kld": [],
        "val_loss": [],
        "val_mse": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, epochs + 1):
        start_t = time.perf_counter()
        model.train()
        run_loss, run_bce, run_kld, total = 0.0, 0.0, 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, _ = model(x)
            loss, bce, kld = vae_loss_function(x_recon, x, mu, logvar, kl_weight=kl_weight)

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            run_loss += loss.item() * bs
            run_bce += bce.item() * bs
            run_kld += kld.item() * bs
            total += bs

        train_loss = run_loss / total
        train_bce = run_bce / total
        train_kld = run_kld / total

        # Validation
        model.eval()
        val_loss_sum, val_mse_sum, val_total = 0.0, 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_recon, mu, logvar, _ = model(x)
                loss, _, _ = vae_loss_function(x_recon, x, mu, logvar, kl_weight=kl_weight)
                mse = F.mse_loss(x_recon, x)

                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_mse_sum += mse.item() * bs
                val_total += bs

        val_loss = val_loss_sum / val_total
        val_mse = val_mse_sum / val_total
        elapsed = time.perf_counter() - start_t

        history["train_loss"].append(train_loss)
        history["train_bce"].append(train_bce)
        history["train_kld"].append(train_kld)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["epoch_time_sec"].append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[VAE] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train ELBO: {train_loss:.4f} (BCE: {train_bce:.4f}, KLD: {train_kld:.4f}) | "
                f"Val MSE: {val_mse:.6f} | Time: {elapsed:.2f}s"
            )

    avg_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
    return {"history": history, "avg_epoch_time_sec": round(avg_epoch_time, 4)}