import argparse
import os
import torch
from torch import nn

from .model import MNISTCNN
from .data import get_dataloaders
from .utils import load_checkpoint, get_device, load_config

@torch.no_grad()
def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in test_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    print(f"[INFO] Using device: {device}")

    # We only need the test loader here
    _, _, test_loader = get_dataloaders(
        root=os.path.expanduser(cfg["data"]["root"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["train"]["val_split"],
        seed=cfg["seed"],
    )

    model = MNISTCNN(num_classes=10).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    print(f"[INFO] Loaded checkpoint from: {args.checkpoint}")

    acc = test_accuracy(model, test_loader, device)
    print(f"[RESULT] Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
