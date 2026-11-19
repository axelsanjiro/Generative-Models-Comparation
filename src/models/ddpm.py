import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dataset import to_zero_one


class SinusoidalTimeEmbedding(nn.Module):
    # Maps scalar diffusion timesteps.

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResBlock(nn.Module):
    # Residual convolutional block conditioned on timestep embedding.

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.residual_conv(x)


class SmallUNet(nn.Module):
    # Lightweight U-Net noise predictor epsilon_theta(x_t, t) for 28x28 grayscale images.
    # Spatial path: 28x28 -> 14x14 -> 7x7 (bottleneck) -> 14x14 -> 28x28.

    def __init__(self, img_channels: int = 1, base_channels: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Encoder (Downsampling)
        self.enc1 = ResBlock(img_channels, base_channels, time_dim)         # 28x28
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)  # 14x14

        self.enc2 = ResBlock(base_channels, base_channels * 2, time_dim)    # 14x14
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)  # 7x7

        # Bottleneck
        self.bot1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)  # 7x7
        self.bot2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)  # 7x7

        # Decoder (Upsampling with skip connections)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)  # 14x14
        self.dec1 = ResBlock(base_channels * 4, base_channels, time_dim)

        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)  # 28x28
        self.dec2 = ResBlock(base_channels * 2, base_channels, time_dim)

        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        # Down
        h1 = self.enc1(x, t_emb)
        h2 = self.enc2(self.down1(h1), t_emb)
        h_bot = self.down2(h2)

        # Bottleneck
        h_bot = self.bot2(self.bot1(h_bot, t_emb), t_emb)

        # Up
        u1 = self.up1(h_bot)
        u1 = self.dec1(torch.cat([u1, h2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, h1], dim=1), t_emb)

        return self.out_conv(u2)


class DDPM(nn.Module):
    # Denoising Diffusion Probabilistic Model with linear variance schedule
    # and T=1000 reverse denoising steps.

    def __init__(
        self,
        unet: SmallUNet,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.unet = unet
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Forward diffusion process
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ab * x_0 + sqrt_one_minus_ab * noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        # Simplified variational bound loss
        bs = x_0.size(0)
        t = torch.randint(0, self.timesteps, (bs,), device=x_0.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.unet(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample_step(self, x_t: torch.Tensor, t_idx: int) -> torch.Tensor:
        # Single reverse Markov transition
        bs = x_t.size(0)
        t_tensor = torch.full((bs,), t_idx, device=x_t.device, dtype=torch.long)

        pred_noise = self.unet(x_t, t_tensor)
        beta_t = self.betas[t_idx]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]

        mean = sqrt_recip_alpha_t * (x_t - (beta_t / sqrt_one_minus_ab_t) * pred_noise)

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.posterior_variance[t_idx])
            return mean + sigma_t * noise
        return mean

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        # Full 1000-step reverse diffusion sampling returning images in [0, 1].
        x = torch.randn((num_samples, 1, 28, 28), device=device)
        for t_idx in reversed(range(self.timesteps)):
            x = self.p_sample_step(x, t_idx)
        return to_zero_one(x)

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        num_samples: int,
        device: torch.device,
        capture_steps: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:

        # Runs reverse denoising and saves intermediate snapshots for Figure 5 trajectory plot.
        if capture_steps is None:
            capture_steps = [999, 800, 600, 400, 200, 100, 50, 0]
        capture_set = set(capture_steps)
        snapshots: Dict[int, torch.Tensor] = {}

        x = torch.randn((num_samples, 1, 28, 28), device=device)
        for t_idx in reversed(range(self.timesteps)):
            if t_idx in capture_set:
                snapshots[t_idx] = to_zero_one(x.clone()).cpu()
            x = self.p_sample_step(x, t_idx)

        snapshots[0] = to_zero_one(x.clone()).cpu()
        return to_zero_one(x), snapshots