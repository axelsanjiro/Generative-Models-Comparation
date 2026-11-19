import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier


def evaluate_classifier(model: MLPClassifier, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    return total_loss / total, (correct / total) * 100.0


def train_mlp_classifier(
    model: MLPClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, object]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        start_t = time.perf_counter()
        model.train()
        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        train_loss = running_loss / total_samples
        val_loss, val_acc = evaluate_classifier(model, val_loader, device)
        elapsed = time.perf_counter() - start_t

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[MLP Classifier] Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | Time: {elapsed:.2f}s"
        )

    test_loss, test_acc = evaluate_classifier(model, test_loader, device)
    print(f"\n--> MLP Classifier Final Test Accuracy: {test_acc:.2f}% (Target >= 88.00%)")
    assert test_acc >= 88.0, f"Test accuracy {test_acc:.2f}% is below the 88% requirement!"

    return {"history": history, "test_loss": test_loss, "test_acc": test_acc}


def train_frozen_latent_decoder(
    classifier: MLPClassifier,
    decoder: FrozenLatentDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    
    # Freezes all MLP Classifier parameters and trains a separate decoder (64 -> 256 -> 784)
    # to reconstruct input images solely from z_MLP using MSE loss.
    classifier.to(device).eval()
    for param in classifier.parameters():
        param.requires_grad = False

    decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_mse": [], "val_mse": []}

    for epoch in range(1, epochs + 1):
        decoder.train()
        running_loss, total = 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_mlp = classifier.extract_features(x)

            optimizer.zero_grad()
            x_recon = decoder(z_mlp)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_mse = running_loss / total

        decoder.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                z_mlp = classifier.extract_features(x)
                x_recon = decoder(z_mlp)
                val_loss += criterion(x_recon, x).item() * x.size(0)
                val_total += x.size(0)

        val_mse = val_loss / val_total
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"[Frozen MLP Decoder] Epoch {epoch:02d}/{epochs:02d} | "
                f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}"
            )

    return history