from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from src.dataset import to_zero_one


def _select_diverse_test_samples(test_loader, num_samples: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # Selects 1 image per distinct Fashion-MNIST class from the Test Set.
    chosen_imgs, chosen_labels = {}, {}
    for x, y in test_loader:
        for img, label in zip(x, y):
            lbl = int(label.item())
            if lbl not in chosen_imgs:
                chosen_imgs[lbl] = img
                chosen_labels[lbl] = lbl
            if len(chosen_imgs) == num_samples:
                break
        if len(chosen_imgs) == num_samples:
            break

    ordered_keys = sorted(chosen_imgs.keys())[:num_samples]
    imgs = torch.stack([chosen_imgs[k] for k in ordered_keys], dim=0)
    labels = torch.tensor([chosen_labels[k] for k in ordered_keys])
    return imgs, labels


@torch.no_grad()
def generate_figure_1_reconstructions(
    vanilla_ae: nn.Module,
    sparse_ae: nn.Module,
    vae: nn.Module,
    test_loader,
    class_names: List[str],
    device: torch.device,
    save_path: Path,
) -> None:
    # Figure 1: Grid Rekonstruksi Side-by-Side (8 Test Set samples).
    # Row 1: Ground Truth | Row 2: Vanilla AE | Row 3: Sparse AE | Row 4: VAE.
    imgs, labels = _select_diverse_test_samples(test_loader, num_samples=8)
    imgs_dev = imgs.to(device)

    vanilla_ae.to(device).eval()
    sparse_ae.to(device).eval()
    vae.to(device).eval()

    recon_vae_ae, _ = vanilla_ae(imgs_dev)
    recon_sparse_ae, _ = sparse_ae(imgs_dev)
    recon_vae, _, _, _ = vae(imgs_dev)

    rows = [
        ("Ground Truth", imgs.cpu()),
        ("Vanilla AE (z=64)", recon_vae_ae.cpu()),
        ("Sparse AE (L1)", recon_sparse_ae.cpu()),
        ("VAE (z=64)", recon_vae.cpu()),
    ]

    fig, axes = plt.subplots(4, 8, figsize=(13, 6.8), dpi=300)
    for r_idx, (row_title, batch) in enumerate(rows):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(batch[c_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(class_names[labels[c_idx]], fontsize=8.5, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=9, fontweight="bold", rotation=90, labelpad=8)

    plt.suptitle("Figure 1: Perbandingan Rekonstruksi Citra Test Set (Vanilla AE, Sparse AE, dan VAE)", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_figure_2_random_sampling(
    vae: nn.Module,
    vanilla_gan_gen: nn.Module,
    dcgan_gen: nn.Module,
    ddpm: nn.Module,
    device: torch.device,
    save_path: Path,
) -> None:
    # Figure 2: Grid 4x4 Random Generative Sampling across 4 generative families:
    # Panel A: VAE | Panel B: Vanilla GAN | Panel C: DCGAN | Panel D: DDPM (T=1000).
    vae.to(device).eval()
    vanilla_gan_gen.to(device).eval()
    dcgan_gen.to(device).eval()
    ddpm.to(device).eval()

    panels = [
        ("Panel A: VAE (z ~ N(0, I))", vae.sample(16, device).cpu()),
        ("Panel B: Vanilla GAN (MLP)", vanilla_gan_gen.sample(16, device).cpu()),
        ("Panel C: DCGAN (Conv)", dcgan_gen.sample(16, device).cpu()),
        ("Panel D: DDPM (U-Net, T=1000)", ddpm.sample(16, device).cpu()),
    ]

    fig = plt.figure(figsize=(11, 11), dpi=300)
    outer_grid = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.18)

    for idx, (panel_title, samples) in enumerate(panels):
        sub_ax = fig.add_subplot(outer_grid[idx // 2, idx % 2])
        sub_ax.set_title(panel_title, fontsize=10.5, fontweight="bold", pad=8)
        sub_ax.axis("off")

        inner_grid = outer_grid[idx // 2, idx % 2].subgridspec(4, 4, wspace=0.05, hspace=0.05)
        for s_idx in range(16):
            ax = fig.add_subplot(inner_grid[s_idx // 4, s_idx % 4])
            ax.imshow(samples[s_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

    plt.suptitle("Figure 2: Sampel Generatif Acak (Grid 4x4) dari Prior Derau Gaussian", fontsize=12.5, fontweight="bold", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_figure_3_mlp_limitations(
    mlp_visuals: Dict[str, torch.Tensor],
    vanilla_ae: nn.Module,
    class_names: List[str],
    device: torch.device,
    save_path: Path,
) -> None:
    """
    Figure 3: 4-Panel Empirical Proof of MLP Classifier Weight & Latent Limitations:
      Panel 1: Ground Truth & Top-Down Class Weight Transposition
      Panel 2: Direct Latent Weight Transposition
      Panel 3: Frozen Latent Decoder vs Vanilla AE
      Panel 4: Random Prior Sampling in MLP Latent Space
    """
    vanilla_ae.to(device).eval()
    gt = mlp_visuals["ground_truth"][:8]
    class_wt = mlp_visuals["class_transpose"][:8]
    latent_wt = mlp_visuals["latent_transpose"][:8]
    frozen_dec = mlp_visuals["frozen_decoder_recon"][:8]
    mlp_prior = mlp_visuals["mlp_prior_samples"][:8]
    vanilla_recon, _ = vanilla_ae(gt.to(device))
    vanilla_recon = vanilla_recon.cpu()

    rows = [
        ("1. Ground Truth (Test)", gt),
        ("2a. W^T Class One-Hot", class_wt),
        ("2b. W^T Latent z_MLP", latent_wt),
        ("3a. Frozen MLP Decoder", frozen_dec),
        ("3b. Vanilla AE (z=64)", vanilla_recon),
        ("4. MLP Prior Sampling", mlp_prior),
    ]

    fig, axes = plt.subplots(len(rows), 8, figsize=(13, 9.5), dpi=300)
    for r_idx, (title, imgs) in enumerate(rows):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 1:
                ax.set_title(class_names[c_idx], fontsize=8)
            if c_idx == 0:
                ax.set_ylabel(title, fontsize=8.5, fontweight="bold", labelpad=8)

    plt.suptitle("Figure 3: Pembuktian Empiris Batasan Bobot (W^T) dan Fitur Laten (Z_MLP) MLP Classifier", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def generate_figure_4_latent_interpolation(
    vanilla_ae: nn.Module,
    sparse_ae: nn.Module,
    vae: nn.Module,
    vanilla_gan_gen: nn.Module,
    dcgan_gen: nn.Module,
    test_loader,
    device: torch.device,
    save_path: Path,
) -> None:
    # Figure 4: Linear Latent Space Interpolation
    # for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] across Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN.

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Pick two contrasting test images: Class 0 (T-shirt/top) and Class 9 (Ankle boot)
    x_a, x_b = None, None
    for x, y in test_loader:
        for img, label in zip(x, y):
            if int(label.item()) == 0 and x_a is None:
                x_a = img.unsqueeze(0).to(device)
            elif int(label.item()) == 9 and x_b is None:
                x_b = img.unsqueeze(0).to(device)
        if x_a is not None and x_b is not None:
            break

    for m in [vanilla_ae, sparse_ae, vae, vanilla_gan_gen, dcgan_gen]:
        m.to(device).eval()

    def interp_vectors(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return torch.cat([(1.0 - a) * z_a + a * z_b for a in alphas], dim=0)

    # Encode endpoints for AE/VAE models
    z_vae_ae = interp_vectors(vanilla_ae.encode(x_a), vanilla_ae.encode(x_b))
    imgs_vae_ae = vanilla_ae.decode(z_vae_ae).cpu()

    z_sparse = interp_vectors(sparse_ae.encode(x_a), sparse_ae.encode(x_b))
    imgs_sparse = sparse_ae.decode(z_sparse).cpu()

    mu_a, _ = vae.encode(x_a)
    mu_b, _ = vae.encode(x_b)
    imgs_vae = vae.decode(interp_vectors(mu_a, mu_b)).cpu()

    # Sample two fixed prior endpoints for GANs
    z_gan_a = torch.randn((1, vanilla_gan_gen.noise_dim), device=device)
    z_gan_b = torch.randn((1, vanilla_gan_gen.noise_dim), device=device)
    imgs_vgan = to_zero_one(vanilla_gan_gen(interp_vectors(z_gan_a, z_gan_b))).cpu()

    z_dc_a = torch.randn((1, dcgan_gen.noise_dim), device=device)
    z_dc_b = torch.randn((1, dcgan_gen.noise_dim), device=device)
    imgs_dcgan = to_zero_one(dcgan_gen(interp_vectors(z_dc_a, z_dc_b))).cpu()

    model_rows = [
        ("Vanilla AE", imgs_vae_ae),
        ("Sparse AE", imgs_sparse),
        ("VAE", imgs_vae),
        ("Vanilla GAN", imgs_vgan),
        ("DCGAN", imgs_dcgan),
    ]

    fig, axes = plt.subplots(5, len(alphas), figsize=(10.5, 8.2), dpi=300)
    for r_idx, (name, batch) in enumerate(model_rows):
        for c_idx, alpha in enumerate(alphas):
            ax = axes[r_idx, c_idx]
            ax.imshow(batch[c_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(f"α = {alpha:.1f}", fontsize=9.5, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(name, fontsize=9.5, fontweight="bold", labelpad=8)

    plt.suptitle("Figure 4: Interpolasi Linier Ruang Laten z(α) = (1 - α)z_A + αz_B", fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figure_5_loss_and_ddpm_trajectory(
    histories: Dict[str, dict],
    ddpm_snapshots: Dict[int, torch.Tensor],
    save_path: Path,
) -> None:
    # Figure 5: Training Loss Curves across all 6 models + DDPM Reverse Denoising Trajectory.

    fig = plt.figure(figsize=(14, 9.5), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85], hspace=0.38, wspace=0.25)

    # 1. Vanilla AE
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(histories["vanilla_ae"]["train_mse"], label="Train MSE")
    ax1.plot(histories["vanilla_ae"]["val_mse"], label="Val MSE", linestyle="--")
    ax1.set_title("Vanilla AE Loss", fontsize=9.5, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. Sparse AE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(histories["sparse_ae"]["train_loss"], label="Train Total (MSE+L1)")
    ax2.plot(histories["sparse_ae"]["val_loss"], label="Val Total", linestyle="--")
    ax2.set_title("Sparse AE Loss", fontsize=9.5, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. VAE
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(histories["vae"]["train_loss"], label="Train ELBO")
    ax3.plot(histories["vae"]["val_loss"], label="Val ELBO", linestyle="--")
    ax3.set_title("VAE Loss (BCE + KLD)", fontsize=9.5, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    # 4. Vanilla GAN
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(histories["vanilla_gan"]["g_loss"], label="Generator Loss")
    ax4.plot(histories["vanilla_gan"]["d_loss"], label="Discriminator Loss")
    ax4.set_title("Vanilla GAN Adversarial Loss", fontsize=9.5, fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    # 5. DCGAN
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(histories["dcgan"]["g_loss"], label="Generator Loss")
    ax5.plot(histories["dcgan"]["d_loss"], label="Discriminator Loss")
    ax5.set_title("DCGAN Adversarial Loss", fontsize=9.5, fontweight="bold")
    ax5.set_xlabel("Epoch")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    # 6. DDPM
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(histories["ddpm"]["train_loss"], label="Train Noise MSE")
    ax6.plot(histories["ddpm"]["val_loss"], label="Val Noise MSE", linestyle="--")
    ax6.set_title("DDPM Noise Prediction Loss", fontsize=9.5, fontweight="bold")
    ax6.set_xlabel("Epoch")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)

    # Bottom row: DDPM Reverse Denoising Trajectory
    steps = sorted(ddpm_snapshots.keys(), reverse=True)
    traj_gs = gs[2, :].subgridspec(1, len(steps), wspace=0.08)
    for idx, step in enumerate(steps):
        ax_t = fig.add_subplot(traj_gs[0, idx])
        ax_t.imshow(ddpm_snapshots[step][0].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax_t.set_title(f"t = {step}", fontsize=9, fontweight="bold")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        if idx == 0:
            ax_t.set_ylabel("Reverse\nDDPM", fontsize=9, fontweight="bold")

    plt.suptitle("Figure 5: Kurva Training Loss Keenam Model & Trajektori Reverse Denoising DDPM (x_T -> x_0)", fontsize=11.5, fontweight="bold", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)