from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path to ensure src package is found
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import HMSDataset, HMSPaths, build_splits
from src.metrics import accuracy_from_logits, kl_divergence_from_logits
from src.models import MultiModalNet, EEGOnlyNet, SpecOnlyNet
from src.utils import get_device, plot_training_curves, save_json, seed_everything


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = float(alpha)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
        kld = kl_divergence_from_logits(logits, target_prob)
        hard = torch.argmax(target_prob, dim=1)
        ce = self.ce(logits, hard)
        return self.alpha * kld + (1.0 - self.alpha) * ce


def build_criterion(loss_name: str, alpha: float = 0.8) -> nn.Module:
    if loss_name == "kldiv":
        class KLDivWrap(nn.Module):
            def forward(self, logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
                return kl_divergence_from_logits(logits, target_prob)
        return KLDivWrap()
    if loss_name == "ce":
        class CEWrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.ce = nn.CrossEntropyLoss()
            def forward(self, logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
                hard = torch.argmax(target_prob, dim=1)
                return self.ce(logits, hard)
        return CEWrap()
    if loss_name == "combined":
        return CombinedLoss(alpha=alpha)
    raise ValueError(f"Unknown loss: {loss_name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("HMS course-ready trainer")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="both", choices=["both", "spec", "eeg"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--loss", type=str, default="kldiv", choices=["kldiv", "ce", "combined"])
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--target_h", type=int, default=600)
    p.add_argument("--target_w", type=int, default=400)
    p.add_argument("--eeg_len", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--pretrained_spec", action="store_true")
    return p.parse_args()


def build_model(model_name: str, eeg_in_ch: int, pretrained_spec: bool) -> nn.Module:
    if model_name == "both":
        return MultiModalNet(n_classes=6, eeg_in_ch=eeg_in_ch, eeg_feat=256, spec_in_ch=1, spec_feat=256, pretrained_spec=pretrained_spec, drop=0.3)
    if model_name == "spec":
        return SpecOnlyNet(n_classes=6, in_ch=1, feat_dim=256, pretrained=pretrained_spec, drop=0.3)
    if model_name == "eeg":
        return EEGOnlyNet(n_classes=6, in_ch=eeg_in_ch, feat_dim=256, drop=0.3)
    raise ValueError(f"Unsupported model: {model_name}")


def forward_model(model: nn.Module, batch: Dict[str, np.ndarray], device: torch.device, model_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eeg = torch.from_numpy(batch["eeg"]).to(device)
    spec = torch.from_numpy(batch["spec"]).to(device)
    target = torch.from_numpy(batch["target"]).to(device)
    if model_name == "both":
        logits = model(eeg, spec)
    elif model_name == "spec":
        logits = model(spec)
    else:
        logits = model(eeg)
    return logits, target, spec


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module, model_name: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_kld = 0.0
    total_acc = 0.0
    n = 0
    for batch in loader:
        logits, target, _spec = forward_model(model, batch, device, model_name)
        loss = criterion(logits, target)
        kld = kl_divergence_from_logits(logits, target)
        acc = accuracy_from_logits(logits, target)
        bs = target.size(0)
        total_loss += float(loss.item()) * bs
        total_kld += float(kld.item()) * bs
        total_acc += float(acc) * bs
        n += bs
    return {
        "val_loss": total_loss / max(n, 1),
        "val_kld": total_kld / max(n, 1),
        "val_acc": total_acc / max(n, 1),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()

    train_csv = os.path.join(args.data_dir, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found at: {train_csv}")

    df = pd.read_csv(train_csv)
    tr_idx, va_idx = build_splits(df, n_folds=args.n_folds, fold=args.fold)
    paths = HMSPaths(data_dir=args.data_dir)
    ds_tr = HMSDataset(df, paths, tr_idx, train=True, target_h=args.target_h, target_w=args.target_w, eeg_len=args.eeg_len)
    ds_va = HMSDataset(df, paths, va_idx, train=False, target_h=args.target_h, target_w=args.target_w, eeg_len=args.eeg_len)

    def collate(batch):
        out = {}
        for k in ["eeg", "spec", "target"]:
            out[k] = np.stack([b[k] for b in batch], axis=0)
        out["eeg_id"] = [int(b["eeg_id"]) for b in batch]
        out["spec_id"] = [int(b["spec_id"]) for b in batch]
        return out

    loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate)
    loader_va = torch.utils.data.DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), collate_fn=collate)

    sample = ds_tr[0]["eeg"]
    eeg_in_ch = int(sample.shape[0])
    model = build_model(args.model, eeg_in_ch=eeg_in_ch, pretrained_spec=args.pretrained_spec).to(device)
    criterion = build_criterion(args.loss, alpha=args.alpha)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        total_steps = max(args.epochs * max(len(loader_tr), 1) // max(args.accum, 1), 1)
        warmup_steps = max(args.warmup_epochs * max(len(loader_tr), 1) // max(args.accum, 1), 1)
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * min(max(t, 0.0), 1.0)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))
    run_name = args.run_name or f'{args.model}_{args.loss}_fold{args.fold}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    save_json(vars(args), os.path.join(out_dir, "config.json"))

    history: List[Dict] = []
    best_kld = float("inf")
    optimizer.zero_grad(set_to_none=True)
    step_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader_tr, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        running_acc = 0.0
        count = 0
        batches_since_step = 0

        for step, batch in enumerate(pbar, start=1):
            logits, target, _spec = forward_model(model, batch, device, args.model)
            loss = criterion(logits, target) / max(args.accum, 1)
            acc = accuracy_from_logits(logits.detach(), target)

            scaler.scale(loss).backward()
            batches_since_step += 1

            should_step = (step % args.accum == 0) or (step == len(loader_tr))
            if should_step:
                if args.clip and args.clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step_count += 1
                batches_since_step = 0

            bs = target.size(0)
            running_loss += float(loss.item()) * bs * max(args.accum, 1)
            running_acc += float(acc) * bs
            count += bs
            pbar.set_postfix(loss=running_loss / max(count, 1), acc=running_acc / max(count, 1), lr=optimizer.param_groups[0]["lr"])

        val = evaluate(model, loader_va, device, criterion, args.model)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(count, 1),
            "train_acc": running_acc / max(count, 1),
            "val_loss": val["val_loss"],
            "val_kld": val["val_kld"],
            "val_acc": val["val_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
        plot_training_curves(history, os.path.join(out_dir, "curves.png"))

        if row["val_kld"] < best_kld:
            best_kld = row["val_kld"]
            torch.save({
                "model": model.state_dict(),
                "best_kld": best_kld,
                "epoch": epoch,
                "model_name": args.model,
                "args": vars(args),
            }, os.path.join(out_dir, "best.pt"))

        print(f"[Run {run_name}] epoch={epoch} train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} val_kld={row['val_kld']:.4f}")

    print("Done. Best val_kld:", best_kld)
    print("Output directory:", out_dir)


if __name__ == "__main__":
    main()
