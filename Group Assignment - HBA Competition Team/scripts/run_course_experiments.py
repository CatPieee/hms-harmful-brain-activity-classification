from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_history_csv, plot_metric_overlay


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser("Run all course-required experiments")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/course_runs")
    p.add_argument("--model", type=str, default="both", choices=["both", "spec", "eeg"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_spec", action="store_true")
    return p.parse_args()


def common_base(args):
    cmd = [
        sys.executable, "-m", "src.train",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--fold", str(args.fold),
        "--n_folds", str(args.n_folds),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]
    if args.pretrained_spec:
        cmd.append("--pretrained_spec")
    return cmd


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    py = common_base(args)

    # 1) baseline
    run_cmd(py + ["--run_name", "baseline_kldiv", "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"])

    # 2) different loss functions
    run_cmd(py + ["--run_name", "loss_kldiv", "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"])
    run_cmd(py + ["--run_name", "loss_ce", "--loss", "ce", "--batch_size", "16", "--lr", "1e-3"])

    # 3) learning-rate sweep
    lr_runs = OrderedDict()
    for lr in ["1e-1", "1e-2", "1e-3", "1e-4"]:
        name = f"lr_{lr}"
        lr_runs[name] = os.path.join(args.output_dir, name, "history.csv")
        run_cmd(py + ["--run_name", name, "--loss", "kldiv", "--batch_size", "16", "--lr", lr])

    # 4) batch-size sweep
    bs_runs = OrderedDict()
    for bs in ["8", "16", "32", "64", "128"]:
        name = f"bs_{bs}"
        bs_runs[name] = os.path.join(args.output_dir, name, "history.csv")
        run_cmd(py + ["--run_name", name, "--loss", "kldiv", "--batch_size", bs, "--lr", "1e-3"])

    # 5) sweep comparison figures (two for LR, two for BS)
    lr_hist = OrderedDict((k, load_history_csv(v)) for k, v in lr_runs.items())
    plot_metric_overlay(lr_hist, metric="train_loss", out_png=os.path.join(args.output_dir, "lr_sweep_loss.png"), title="Learning Rate Sweep - Training Loss", ylabel="Train Loss")
    plot_metric_overlay(lr_hist, metric="val_acc", out_png=os.path.join(args.output_dir, "lr_sweep_accuracy.png"), title="Learning Rate Sweep - Validation Accuracy", ylabel="Validation Accuracy")

    bs_hist = OrderedDict((k, load_history_csv(v)) for k, v in bs_runs.items())
    plot_metric_overlay(bs_hist, metric="train_loss", out_png=os.path.join(args.output_dir, "bs_sweep_loss.png"), title="Batch Size Sweep - Training Loss", ylabel="Train Loss")
    plot_metric_overlay(bs_hist, metric="val_acc", out_png=os.path.join(args.output_dir, "bs_sweep_accuracy.png"), title="Batch Size Sweep - Validation Accuracy", ylabel="Validation Accuracy")

    # 6) first 100 visualization using baseline checkpoint
    run_cmd([
        sys.executable, "scripts/generate_first100.py",
        "--data_dir", args.data_dir,
        "--checkpoint", os.path.join(args.output_dir, "baseline_kldiv", "best.pt"),
        "--output_dir", os.path.join(args.output_dir, "first100"),
        "--model", args.model,
        "--fold", str(args.fold),
        "--n_folds", str(args.n_folds),
        "--num_workers", str(args.num_workers),
    ] + (["--pretrained_spec"] if args.pretrained_spec else []))

    print("\nAll course-required experiments finished.")


if __name__ == "__main__":
    main()
