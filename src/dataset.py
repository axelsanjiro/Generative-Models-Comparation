from typing import Tuple
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms
from src.config import Config


def to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    # ngubah rentang tensor dari [0, 1] ke [-1, 1] (untuk GAN & DDPM).
    return x * 2.0 - 1.0


def to_zero_one(x: torch.Tensor) -> torch.Tensor:
    # Mengembalikan rentang tensor dari [-1, 1] ke [0, 1] untuk evaluasi seragam.
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def get_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    seed: int = Config.SEED,
    data_dir: str = str(Config.DATA_DIR),
    num_workers: int = Config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    transform = transforms.ToTensor()  # Transform tensor [1, 28, 28] dalam rentang [0, 1]

    raw_train = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    raw_test = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    full_dataset = ConcatDataset([raw_train, raw_test])
    total_len = len(full_dataset)  

    train_len = int(total_len * Config.TRAIN_RATIO)  
    val_len = int(total_len * Config.VAL_RATIO)      
    test_len = total_len - train_len - val_len       

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader