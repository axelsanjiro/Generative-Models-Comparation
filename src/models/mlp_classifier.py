from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    # Discriminative MLP Classifier (784 -> 256 -> 64 -> 10) 
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # W1: (256, 784)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)  # W2: (64, 256)
        self.fc3 = nn.Linear(latent_dim, num_classes) # W3: (10, 64)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Extracts penultimate latent representation z_MLP in R^64.
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_mlp = self.extract_features(x)
        logits = self.fc3(z_mlp)
        return logits, z_mlp

    @torch.no_grad()
    def transpose_from_class(self, y_onehot: torch.Tensor) -> torch.Tensor:
        # Stage 2: Top-Down Weight Transposition directly from class one-hot vector y_c
        z_hat = F.relu(y_onehot @ self.fc3.weight)
        h1_hat = F.relu(z_hat @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def transpose_from_latent(self, z_mlp: torch.Tensor) -> torch.Tensor:
        # Stage 3: Latent Transpose Projection directly from extracted test latent z_MLP
        h1_hat = F.relu(z_mlp @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)

class FrozenLatentDecoder(nn.Module):
    # Stage 4: Separate MLP Decoder (64 -> 256 -> 784) trained on frozen z_MLP
    # to quantify intra-class spatial information lost during discriminative training.
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_mlp: torch.Tensor) -> torch.Tensor:
        out = self.net(z_mlp)
        return out.view(-1, 1, 28, 28)