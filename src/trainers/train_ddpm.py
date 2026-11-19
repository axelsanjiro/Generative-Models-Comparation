import time
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from src.dataset import to_minus_one_one
from src.models.ddpm import DDPM


def train_ddpm(
    model: DDPM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, object]:
    # Trains DDPM using simplified MSE noise prediction objective across T=1000 timesteps.
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, epochs + 1):
        start_t = time.perf_counter()
        model.train()
        running_loss, total_samples = 0.0, 0

        for x, _ in train_loader:
            x_0 = to_minus_one_one(x.to(device))
            optimizer.zero_grad()
            loss = model.compute_loss(x_0)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

        train_loss = running_loss / total_samples

        # Validation noise prediction loss
        model.eval()
        val_loss_sum, val_total = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x_0 = to_minus_one_one(x.to(device))
                loss = model.compute_loss(x_0)
                bs = x.size(0)
                val_loss_sum += loss.item() * bs
                val_total += bs

        val_loss = val_loss_sum / val_total
        elapsed = time.perf_counter() - start_t

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time_sec"].append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[DDPM] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Noise MSE: {train_loss:.6f} | Val Noise MSE: {val_loss:.6f} | "
                f"Time: {elapsed:.2f}s"
            )

    avg_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
    return {"history": history, "avg_epoch_time_sec": round(avg_epoch_time, 4)}