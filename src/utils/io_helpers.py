import json
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    filepath: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra_info is not None:
        payload["extra_info"] = extra_info
    torch.save(payload, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: Path,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_table(records: list[Dict[str, Any]], csv_path: Path) -> pd.DataFrame:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    return df