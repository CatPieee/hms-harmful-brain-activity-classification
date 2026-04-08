import csv
import json
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


LABELS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[Dict], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(history: List[Dict], out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row.get("val_loss", 0) for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis: Loss
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (KLDiv)', color=color_loss)
    ln1 = ax1.plot(epochs, train_loss, label="Train Loss", color=color_loss, linestyle='--', marker='o')
    ln2 = ax1.plot(epochs, val_loss, label="Val Loss", color='darkred', linestyle='-', marker='s')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Right Y-axis: Accuracy
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc)
    ln3 = ax2.plot(epochs, train_acc, label="Train Acc", color=color_acc, linestyle='--', marker='^')
    ln4 = ax2.plot(epochs, val_acc, label="Val Acc", color='darkblue', linestyle='-', marker='v')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(0, 1.0) # Accuracy is between 0 and 1

    # Combined Legend
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True)

    plt.title("HMS Harmful Brain Activity: Training & Validation Metrics")
    fig.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_metric_overlay(run_to_history: Dict[str, List[Dict]], metric: str, out_png: str, title: str, ylabel: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    plt.figure(figsize=(8.5, 5.2))
    for run_name, history in run_to_history.items():
        epochs = [row["epoch"] for row in history]
        values = [row[metric] for row in history]
        plt.plot(epochs, values, label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_first100_grid(spec_images: np.ndarray, y_true: List[int], y_pred: List[int], out_png: str) -> None:
    ensure_dir(os.path.dirname(out_png))
    n = min(100, len(spec_images))
    cols = 10
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(18, 18))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(spec_images[i], cmap="gray", aspect="auto")
        ax.set_title(f"T:{LABELS[y_true[i]]}\nP:{LABELS[y_pred[i]]}", fontsize=6)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def load_history_csv(path: str) -> List[Dict]:
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
