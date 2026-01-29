"""
src/utils.py
Utility helpers: seeding, meters, plotting.
"""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    name: str
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.val = float(value)
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def save_history_csv(history: List[Dict], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if not history:
        return
    keys = list(history[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in history:
            w.writerow(row)


def plot_curves(history: List[Dict], out_png: str) -> None:
    """
    Plots:
    - train_loss
    - train_acc
    - val_acc
    """
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, train_acc, label="train_acc")
    plt.plot(epochs, val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_first100_table(
    rows: List[Dict],
    out_png: str,
    max_rows: int = 100,
) -> None:
    """
    Create a simple figure/table for the first N predictions required by the course.
    rows: list of dicts with keys: index, y_true, y_pred, sample_id
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    df = pd.DataFrame(rows).head(max_rows)

    fig, ax = plt.subplots(figsize=(10, 18))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
