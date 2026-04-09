import argparse
import os
import subprocess
import sys
from collections import OrderedDict
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_history_csv, plot_metric_overlay


def parse_args():
    p = argparse.ArgumentParser("Run all course-required experiments")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="outputs/course_runs")
    p.add_argument("--model", type=str, default="both", choices=["both", "spec", "eeg"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained_spec", action="store_true")
    return p.parse_args()


def common_base(args):
    cmd = [
        sys.executable, "-m", "src.train",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--val_size", str(args.val_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]
    if args.pretrained_spec:
        cmd.append("--pretrained_spec")
    return cmd


def run_cmd(cmd, run_name, output_dir):
    """Run a training command if the output history doesn't exist yet."""
    history_file = os.path.join(output_dir, run_name, "history.csv")
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        print(f"\n[INFO] Skipping {run_name}, history already exists at {history_file}")
        return
    
    print(f"\n>>> Running Experiment: {run_name}")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_results(src_name, dst_name, output_dir):
    """Copy results from one run to another to avoid redundant training."""
    src_path = os.path.join(output_dir, src_name)
    dst_path = os.path.join(output_dir, dst_name)
    
    if os.path.exists(dst_path):
        return
    
    if not os.path.exists(src_path):
        print(f"[WARNING] Cannot copy from {src_path} to {dst_path}, source missing.")
        return

    print(f"[INFO] Copying results from {src_name} to {dst_name} (redundant run avoid)")
    shutil.copytree(src_path, dst_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    py = common_base(args)

    # 1) Model Selection Order: Spec -> EEG -> Both (Multimodal)
    
    # 1.1) Spec-only Baseline
    spec_base = "baseline_spec"
    run_cmd(py + ["--run_name", spec_base, "--model", "spec", "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"], spec_base, args.output_dir)
    
    # 1.2) EEG-only Baseline
    eeg_base = "baseline_eeg"
    run_cmd(py + ["--run_name", eeg_base, "--model", "eeg", "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"], eeg_base, args.output_dir)
    
    # 1.3) Multimodal Baseline (The "Both" model)
    both_base = "baseline_both"
    run_cmd(py + ["--run_name", both_base, "--model", "both", "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"], both_base, args.output_dir)

    # Use the requested model for the rest of the ablation studies
    target_model = args.model # default is 'both'
    baseline_name = f"baseline_{target_model}"
    # Ensure baseline folder exists (it might be one of the above)
    if not os.path.exists(os.path.join(args.output_dir, baseline_name)):
        run_cmd(py + ["--run_name", baseline_name, "--model", target_model, "--loss", "kldiv", "--batch_size", "16", "--lr", "1e-3"], baseline_name, args.output_dir)

    # 2) different loss functions (using target model)
    run_cmd(py + ["--run_name", f"loss_ce_{target_model}", "--model", target_model, "--loss", "ce", "--batch_size", "16", "--lr", "1e-3"], f"loss_ce_{target_model}", args.output_dir)

    # 3) learning-rate sweep (using target model)
    lrs = ["1e-1", "1e-2", "1e-3", "1e-4"]
    lr_runs = OrderedDict()
    for lr in lrs:
        name = f"lr_{lr}_{target_model}"
        lr_runs[name] = os.path.join(args.output_dir, name, "history.csv")
        if lr == "1e-3":
            copy_results(baseline_name, name, args.output_dir)
        else:
            run_cmd(py + ["--run_name", name, "--model", target_model, "--loss", "kldiv", "--batch_size", "16", "--lr", lr], name, args.output_dir)

    # 4) batch-size sweep (using target model)
    bss = ["8", "16", "32", "64"]
    bs_runs = OrderedDict()
    for bs in bss:
        name = f"bs_{bs}_{target_model}"
        bs_runs[name] = os.path.join(args.output_dir, name, "history.csv")
        if bs == "16":
            copy_results(baseline_name, name, args.output_dir)
        else:
            cmd = py + ["--run_name", name, "--model", target_model, "--loss", "kldiv", "--batch_size", bs, "--lr", "1e-3"]
            if int(bs) >= 32:
                for i, part in enumerate(cmd):
                    if part == "--num_workers":
                        cmd[i+1] = "0"
            run_cmd(cmd, name, args.output_dir)

    # 5) sweep comparison figures
    print("\nGenerating comparison figures...")
    try:
        lr_hist = OrderedDict()
        for k, v in lr_runs.items():
            if os.path.exists(v):
                lr_hist[k] = load_history_csv(v)
        
        if lr_hist:
            plot_metric_overlay(lr_hist, metric="train_loss", out_png=os.path.join(args.output_dir, "lr_sweep_loss.png"), title="Learning Rate Sweep - Training Loss", ylabel="Train Loss")
            plot_metric_overlay(lr_hist, metric="val_acc", out_png=os.path.join(args.output_dir, "lr_sweep_accuracy.png"), title="Learning Rate Sweep - Validation Accuracy", ylabel="Validation Accuracy")

        bs_hist = OrderedDict()
        for k, v in bs_runs.items():
            if os.path.exists(v):
                bs_hist[k] = load_history_csv(v)
        
        if bs_hist:
            plot_metric_overlay(bs_hist, metric="train_loss", out_png=os.path.join(args.output_dir, "bs_sweep_loss.png"), title="Batch Size Sweep - Training Loss", ylabel="Train Loss")
            plot_metric_overlay(bs_hist, metric="val_acc", out_png=os.path.join(args.output_dir, "bs_sweep_accuracy.png"), title="Batch Size Sweep - Validation Accuracy", ylabel="Validation Accuracy")
    except Exception as e:
        print(f"[ERROR] Failed to generate comparison figures: {e}")

    # 6) first 100 visualization using baseline checkpoint
    ckpt_path = os.path.join(args.output_dir, baseline_name, "best.pt")
    if os.path.exists(ckpt_path):
        run_cmd([
            sys.executable, "scripts/generate_first100.py",
            "--data_dir", args.data_dir,
            "--checkpoint", ckpt_path,
            "--output_dir", os.path.join(args.output_dir, "first100"),
            "--model", target_model,
            "--val_size", str(args.val_size),
            "--num_workers", "0",
        ] + (["--pretrained_spec"] if args.pretrained_spec else []), "first100", args.output_dir)
    else:
        print(f"[ERROR] Skipping first-100 visualization because baseline checkpoint missing: {ckpt_path}")

    print("\nAll course-required experiments finished.")


if __name__ == "__main__":
    main()
