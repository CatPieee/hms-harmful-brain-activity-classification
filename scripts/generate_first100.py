from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import HMSDataset, HMSPaths, build_splits
from src.metrics import accuracy_from_logits
from src.models import MultiModalNet, EEGOnlyNet, SpecOnlyNet
from src.utils import LABELS, plot_first100_grid, save_csv


def parse_args():
    p = argparse.ArgumentParser("Generate first-100 validation predictions")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="both", choices=["both", "spec", "eeg"])
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--target_h", type=int, default=600)
    p.add_argument("--target_w", type=int, default=400)
    p.add_argument("--eeg_len", type=int, default=5000)
    p.add_argument("--max_items", type=int, default=100)
    p.add_argument("--pretrained_spec", action="store_true")
    return p.parse_args()


def build_model(model_name: str, eeg_in_ch: int, pretrained_spec: bool):
    if model_name == "both":
        return MultiModalNet(n_classes=6, eeg_in_ch=eeg_in_ch, eeg_feat=256, spec_in_ch=1, spec_feat=256, pretrained_spec=pretrained_spec, drop=0.3)
    if model_name == "spec":
        return SpecOnlyNet(n_classes=6, in_ch=1, feat_dim=256, pretrained=pretrained_spec, drop=0.3)
    return EEGOnlyNet(n_classes=6, in_ch=eeg_in_ch, feat_dim=256, drop=0.3)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    _tr_idx, va_idx = build_splits(df, val_size=args.val_size)
    ds_va = HMSDataset(df, HMSPaths(data_dir=args.data_dir), va_idx, train=False, target_h=args.target_h, target_w=args.target_w, eeg_len=args.eeg_len)

    sample = ds_va[0]["eeg"]
    model = build_model(args.model, eeg_in_ch=int(sample.shape[0]), pretrained_spec=args.pretrained_spec).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    rows = []
    grid_specs = []
    y_true_list = []
    y_pred_list = []

    n = min(args.max_items, len(ds_va))
    with torch.no_grad():
        for i in range(n):
            item = ds_va[i]
            eeg = torch.from_numpy(item["eeg"]).unsqueeze(0).to(device)
            spec = torch.from_numpy(item["spec"]).unsqueeze(0).to(device)
            target = torch.from_numpy(item["target"]).unsqueeze(0).to(device)
            if args.model == "both":
                logits = model(eeg, spec)
            elif args.model == "spec":
                logits = model(spec)
            else:
                logits = model(eeg)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            true_idx = int(torch.argmax(target, dim=1).item())
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            rows.append({
                "index": i,
                "eeg_id": int(item["eeg_id"]),
                "spectrogram_id": int(item["spec_id"]),
                "true_label": LABELS[true_idx],
                "pred_label": LABELS[pred_idx],
                "pred_confidence": float(probs[pred_idx]),
            })
            grid_specs.append(item["spec"][0])
            y_true_list.append(true_idx)
            y_pred_list.append(pred_idx)

    save_csv(rows, os.path.join(args.output_dir, "first100_predictions.csv"))
    plot_first100_grid(np.array(grid_specs), y_true_list, y_pred_list, os.path.join(args.output_dir, "first100_grid.png"))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
