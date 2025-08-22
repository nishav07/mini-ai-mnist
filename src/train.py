import argparse
import os
from typing import Tuple

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .model import MNISTCNN
from .data import get_dataloaders
from .utils import (
    seed_everything, get_device, save_checkpoint, accuracy, load_config
)

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    log_interval: int,
    use_amp: bool
) -> Tuple[float, float]:
    model.train()
    loss_meter = 0.0
    acc_meter = 0.0
    n_samples = 0

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train", leave=False)
    for batch_idx, (x, y) in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch_size = y.size(0)
            acc = accuracy(logits, y)
            loss_meter += loss.item() * batch_size
            acc_meter += acc * batch_size
            n_samples += batch_size

        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss_meter / n_samples:.4f}",
                              "acc": f"{acc_meter / n_samples:.4f}"})

    return loss_meter / n_samples, acc_meter / n_samples

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    n_samples = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = y.size(0)
        loss_meter += loss.item() * batch_size
        acc_meter += accuracy(logits, y) * batch_size
        n_samples += batch_size

    return loss_meter / n_samples, acc_meter / n_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    device = get_device()
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        root=os.path.expanduser(cfg["data"]["root"]),
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_split=cfg["train"]["val_split"],
        seed=cfg["seed"],
    )

    model = MNISTCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])

    use_amp = bool(cfg["hardware"]["use_amp"] and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    artifacts_dir = cfg["paths"]["artifacts_dir"]
    best_ckpt = os.path.join(artifacts_dir, cfg["paths"]["best_ckpt_name"])
    os.makedirs(artifacts_dir, exist_ok=True)

    epochs = cfg["train"]["epochs"]
    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scaler,
            log_interval=cfg["train"]["log_interval"], use_amp=use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(f"[Epoch {epoch}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_ckpt)
            print(f"[INFO] New best model saved to: {best_ckpt} (val_acc={best_val_acc:.4f})")

    print(f"\n[DONE] Best Val Acc: {best_val_acc:.4f}")
    print(f"[TIP] Evaluate with: python -m src.eval --config {args.config} --checkpoint {best_ckpt}")

if __name__ == "__main__":
    main()
