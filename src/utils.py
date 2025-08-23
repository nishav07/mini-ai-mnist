import os
import random
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
from torch import nn
import yaml


@dataclass
class AverageMeter:
    """Keeps running average of a metric."""
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1):
        self.total += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_checkpoint(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model: nn.Module, path: str, map_location=None):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
