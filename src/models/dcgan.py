import torch
import torch.nn as nn
from src.dataset import to_zero_one

def weights_init_dcgan(m: nn.Module) -> None:
    # Custom weight initialization N(0, 0.02) per DCGAN architectural guidelines.
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class DCGANGenerator(nn.Module):
    # Full-Convolutional Generator upsampling z (100x1x1) -> 7x7 -> 14x14 -> 28x28.
    def __init__(self, noise_dim: int = 100, feature_maps: int = 64, img_channels: int = 1):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            # Input: (B, noise_dim, 1, 1) -> (B, feature_maps*4, 7, 7)
            nn.ConvTranspose2d(noise_dim, feature_maps * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # State: (B, 256, 7, 7) -> (B, feature_maps*2, 14, 14)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # State: (B, 128, 14, 14) -> (B, 1, 28, 28)
            nn.ConvTranspose2d(feature_maps * 2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.noise_dim, 1, 1)
        return self.net(z)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # Generates synthetic images in [0, 1] from standard Gaussian prior.
        z = torch.randn((num_samples, self.noise_dim, 1, 1), device=device)
        return to_zero_one(self.forward(z))


class DCGANDiscriminator(nn.Module):
    # Full-Convolutional Discriminator downsampling 28x28 -> 14x14 -> 7x7 -> 1x1 logits.

    def __init__(self, feature_maps: int = 64, img_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, 1, 28, 28) -> (B, feature_maps, 14, 14) [No BatchNorm on input layer]
            nn.Conv2d(img_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (B, 64, 14, 14) -> (B, feature_maps*2, 7, 7)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (B, 128, 7, 7) -> (B, 1, 1, 1)
            nn.Conv2d(feature_maps * 2, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(-1, 1)