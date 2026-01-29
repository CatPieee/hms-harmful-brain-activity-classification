"""
src/experiments.py
Runs the exact experiment sets required by the course:
- baseline curve
- different loss function
- different learning rates (sweep)
- different batch sizes (sweep)
- first 100 predictions table/figure

Notes:
- This uses the spectrogram-only baseline to satisfy deliverables first.
- Extend later with EEG waveform branch and multimodal fusion.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .datasets import HMSSpectrogramPngDataset, SyntheticSpectrogramDataset, stratified_group_split
from .models import SpectrogramResNet
from .train import build_loss, evaluate, train_one_epoch
from .utils import plot_curves, save_first100_table, save_history_csv, set_seed


def run_single(
    *,
    ds_train,
    ds_val,
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    loss: str,
    seed: int,
) -> List[Dict]:
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SpectrogramResNet(in_ch=1, num_classes=6).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = build_loss(loss)

    hist: List[Dict] = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, optim, loss_fn, device, loss)
        va_loss, va_acc = evaluate(model, dl_val, loss_fn, device, loss)
        hist.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc, "lr": lr, "loss": loss, "batch_size": batch_size})
        print(hist[-1])

    save_history_csv(hist, os.path.join(out_dir, "history.csv"))
    plot_curves(hist, os.path.join(out_dir, "curves.png"))
    return hist


@torch.no_grad()
def first100_predictions(ds_val, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramResNet(in_ch=1, num_classes=6).to(device)
    model.eval()

    # random weights => only for format; load best.pt in real runs
    dl = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=2)
    rows = []
    idx = 0
    for x, y, sid in dl:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        y_pred = probs.argmax(dim=1).numpy()
        y_true = y.argmax(dim=1).numpy()
        for i in range(len(sid)):
            rows.append({"index": idx, "sample_id": sid[i], "y_true": int(y_true[i]), "y_pred": int(y_pred[i])})
            idx += 1
            if idx >= 100:
                break
        if idx >= 100:
            break

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "first100.csv"), index=False)
    save_first100_table(rows, os.path.join(out_dir, "first100.png"), max_rows=100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--exp", type=str, required=True, choices=["baseline", "loss_compare", "lr_sweep", "bs_sweep", "first100"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # dataset
    if args.synthetic:
        ds = SyntheticSpectrogramDataset(n=2048)
        idx_train = list(range(0, 1600))
        idx_val = list(range(1600, 2048))
        ds_train = Subset(ds, idx_train)
        ds_val = Subset(ds, idx_val)
    else:
        ds_full = HMSSpectrogramPngDataset(data_dir=args.data_dir, split_csv="train.csv")
        if "patient_id" in ds_full.df.columns:
            idx_train, idx_val = stratified_group_split(ds_full.df, group_col="patient_id", frac_val=0.2, seed=args.seed)
        else:
            idx_train = list(range(0, int(len(ds_full)*0.8)))
            idx_val = list(range(int(len(ds_full)*0.8), len(ds_full)))
        ds_train = Subset(ds_full, idx_train)
        ds_val = Subset(ds_full, idx_val)

    root_out = os.path.join("outputs", args.exp)
    if args.exp == "baseline":
        run_single(ds_train=ds_train, ds_val=ds_val, out_dir=root_out, epochs=args.epochs, batch_size=32, lr=1e-3, loss="kldiv", seed=args.seed)
    elif args.exp == "loss_compare":
        run_single(ds_train=ds_train, ds_val=ds_val, out_dir=os.path.join(root_out, "kldiv"), epochs=args.epochs, batch_size=32, lr=1e-3, loss="kldiv", seed=args.seed)
        run_single(ds_train=ds_train, ds_val=ds_val, out_dir=os.path.join(root_out, "ce"), epochs=args.epochs, batch_size=32, lr=1e-3, loss="ce", seed=args.seed)
    elif args.exp == "lr_sweep":
        for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
            run_single(ds_train=ds_train, ds_val=ds_val, out_dir=os.path.join(root_out, f"lr_{lr:g}"), epochs=args.epochs, batch_size=32, lr=lr, loss="kldiv", seed=args.seed)
    elif args.exp == "bs_sweep":
        for bs in [8, 16, 32, 64, 128]:
            run_single(ds_train=ds_train, ds_val=ds_val, out_dir=os.path.join(root_out, f"bs_{bs}"), epochs=args.epochs, batch_size=bs, lr=1e-3, loss="kldiv", seed=args.seed)
    elif args.exp == "first100":
        first100_predictions(ds_val, out_dir=root_out)


if __name__ == "__main__":
    main()
