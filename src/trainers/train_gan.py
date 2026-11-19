import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import to_minus_one_one


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    betas: Tuple[float, float],
    noise_dim: int,
    device: torch.device,
    model_name: str = "GAN",
) -> Dict[str, object]:
    # Standard minimax GAN training loop for Vanilla GAN and DCGAN using BCEWithLogitsLoss.

    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    history: Dict[str, List[float]] = {
        "g_loss": [],
        "d_loss": [],
        "epoch_time_sec": [],
    }

    for epoch in range(1, epochs + 1):
        start_t = time.perf_counter()
        generator.train()
        discriminator.train()

        running_g_loss, running_d_loss, total_samples = 0.0, 0.0, 0

        for real_imgs, _ in train_loader:
            bs = real_imgs.size(0)
            real_imgs = to_minus_one_one(real_imgs.to(device))

            real_labels = torch.ones((bs, 1), device=device)
            fake_labels = torch.zeros((bs, 1), device=device)

            # 1. Update Discriminator
    
            opt_d.zero_grad()

            logits_real = discriminator(real_imgs)
            loss_d_real = criterion(logits_real, real_labels)

            z = torch.randn((bs, noise_dim), device=device)
            fake_imgs = generator(z)
            logits_fake = discriminator(fake_imgs.detach())
            loss_d_fake = criterion(logits_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # 2. Update Generator
            
            opt_g.zero_grad()

            logits_fake_for_g = discriminator(fake_imgs)
            loss_g = criterion(logits_fake_for_g, real_labels)
            loss_g.backward()
            opt_g.step()

            running_d_loss += loss_d.item() * bs
            running_g_loss += loss_g.item() * bs
            total_samples += bs

        epoch_d_loss = running_d_loss / total_samples
        epoch_g_loss = running_g_loss / total_samples
        elapsed = time.perf_counter() - start_t

        history["d_loss"].append(epoch_d_loss)
        history["g_loss"].append(epoch_g_loss)
        history["epoch_time_sec"].append(elapsed)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[{model_name}] Epoch {epoch:02d}/{epochs:02d} | "
                f"D Loss: {epoch_d_loss:.4f} | G Loss: {epoch_g_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

    avg_epoch_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
    return {"history": history, "avg_epoch_time_sec": round(avg_epoch_time, 4)}