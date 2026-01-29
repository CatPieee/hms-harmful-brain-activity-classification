"""
src/train.py
Single-run training entry (baseline).

Examples:
  python -m src.train --synthetic --epochs 5 --out_dir outputs/smoke
  python -m src.train --data_dir data/hms --epochs 10 --loss kldiv --out_dir outputs/baseline
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .datasets import (
    HMSSpectrogramPngDataset,
    SyntheticSpectrogramDataset,
    stratified_group_split,
)
from .models import SpectrogramResNet
from .utils import AverageMeter, plot_curves, save_history_csv, set_seed


def accuracy_from_probs(probs: torch.Tensor, target_dist: torch.Tensor) -> float:
    y_pred = probs.argmax(dim=1)
    y_true = target_dist.argmax(dim=1)
    return (y_pred == y_true).float().mean().item()


def build_loss(loss_name: str) -> nn.Module:
    if loss_name == "kldiv":
        return nn.KLDivLoss(reduction="batchmean")
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {loss_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    loss_name: str,
) -> Tuple[float, float]:
    model.train()
    meter_loss = AverageMeter("train_loss")
    meter_acc = AverageMeter("train_acc")

    for x, y_dist, _sid in loader:
        x = x.to(device)
        y_dist = y_dist.to(device)

        logits = model(x)

        if loss_name == "kldiv":
            log_probs = torch.log_softmax(logits, dim=1)
            loss = loss_fn(log_probs, y_dist)
            probs = torch.softmax(logits, dim=1)
        else:  # ce
            y_hard = y_dist.argmax(dim=1)
            loss = loss_fn(logits, y_hard)
            probs = torch.softmax(logits, dim=1)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = accuracy_from_probs(probs, y_dist)
        meter_loss.update(loss.item(), n=x.size(0))
        meter_acc.update(acc, n=x.size(0))

    return meter_loss.avg, meter_acc.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    loss_name: str,
) -> Tuple[float, float]:
    model.eval()
    meter_loss = AverageMeter("val_loss")
    meter_acc = AverageMeter("val_acc")

    for x, y_dist, _sid in loader:
        x = x.to(device)
        y_dist = y_dist.to(device)

        logits = model(x)

        if loss_name == "kldiv":
            log_probs = torch.log_softmax(logits, dim=1)
            loss = loss_fn(log_probs, y_dist)
            probs = torch.softmax(logits, dim=1)
        else:
            y_hard = y_dist.argmax(dim=1)
            loss = loss_fn(logits, y_hard)
            probs = torch.softmax(logits, dim=1)

        acc = accuracy_from_probs(probs, y_dist)
        meter_loss.update(loss.item(), n=x.size(0))
        meter_acc.update(acc, n=x.size(0))

    return meter_loss.avg, meter_acc.avg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="", help="Folder containing train.csv and spec_png/")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic dataset (smoke test).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", type=str, default="kldiv", choices=["kldiv", "ce"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs/run")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    if args.synthetic:
        ds = SyntheticSpectrogramDataset(n=2048)
        idx_train, idx_val = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=args.seed, shuffle=True)
        ds_train = Subset(ds, idx_train)
        ds_val = Subset(ds, idx_val)
    else:
        ds_full = HMSSpectrogramPngDataset(data_dir=args.data_dir, split_csv="train.csv")
        if "patient_id" in ds_full.df.columns:
            idx_train, idx_val = stratified_group_split(ds_full.df, group_col="patient_id", frac_val=0.2, seed=args.seed)
        else:
            idx_train, idx_val = train_test_split(np.arange(len(ds_full)), test_size=0.2, random_state=args.seed, shuffle=True)
        ds_train = Subset(ds_full, idx_train)
        ds_val = Subset(ds_full, idx_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SpectrogramResNet(in_ch=1, num_classes=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = build_loss(args.loss)

    history: List[Dict] = []
    best_val = float("inf")
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optimizer, loss_fn, device, args.loss)
        va_loss, va_acc = evaluate(model, dl_val, loss_fn, device, args.loss)

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc, "lr": args.lr, "loss": args.loss}
        history.append(row)
        print(row)

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss}, best_path)

    save_history_csv(history, os.path.join(args.out_dir, "history.csv"))
    plot_curves(history, os.path.join(args.out_dir, "curves.png"))
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
