import torch
import torch.nn as nn
from src.dataset import to_zero_one


class VanillaGenerator(nn.Module):
    # Fully-Connected (MLP) Generator mapping prior noise z ~ N(0, I)
    # to flattened 28x28 grayscale images in [-1, 1].

    def __init__(self, noise_dim: int = 100, output_dim: int = 784):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # Generates synthetic images in [0, 1] from standard Gaussian prior.
        z = torch.randn((num_samples, self.noise_dim), device=device)
        return to_zero_one(self.forward(z))


class VanillaDiscriminator(nn.Module):
    # Fully-Connected (MLP) Discriminator outputting raw logits (real vs fake).

    def __init__(self, input_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        return self.net(x_flat)