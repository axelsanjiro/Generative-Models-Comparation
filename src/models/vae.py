from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    # Variational Autoencoder (VAE) with Gaussian latent prior N(0, I)
    # and reparameterization trick (latent_dim = 64).

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        h = self.encoder_shared(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = self.decoder(z)
        return out.view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn((num_samples, self.latent_dim), device=device)
        return self.decode(z)


def vae_loss_function(
    x_recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    batch_size = x.size(0)
    bce_loss = F.binary_cross_entropy(x_recon, x, reduction="sum") / batch_size
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = bce_loss + kl_weight * kld_loss
    return total_loss, bce_loss, kld_loss