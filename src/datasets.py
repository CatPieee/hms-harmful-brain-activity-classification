"""
src/datasets.py
Dataset loaders.

Important: The real Kaggle HMS dataset requires preprocessing.
This repo supports two modes:
1) Synthetic mode (always works): --synthetic
2) PNG spectrogram mode: expects you have pre-generated spectrogram images
   at: <data_dir>/spec_png/{spectrogram_id}.png

For a competition-grade pipeline, you can later add:
- reading parquet spectrograms directly
- EEG waveform loading + montage + filtering
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

LABELS = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTE_COLS = [f"{k}_vote" for k in LABELS]


def normalize_votes(votes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    votes = votes.astype(np.float32)
    s = votes.sum()
    if s <= 0:
        # fallback uniform
        return np.ones_like(votes) / len(votes)
    return votes / (s + eps)


class SyntheticSpectrogramDataset(Dataset):
    """Always runnable: random spectrogram + random vote distribution."""

    def __init__(self, n: int = 2048, h: int = 128, w: int = 256, num_classes: int = 6, seed: int = 42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(n, 1, h, w)).astype(np.float32)
        raw = rng.integers(0, 20, size=(n, num_classes))
        self.y = np.stack([normalize_votes(r) for r in raw], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx]), f"synthetic_{idx}"


@dataclass
class HMSPaths:
    data_dir: str
    csv_path: str
    spec_png_dir: str


def make_hms_paths(data_dir: str, split: str = "train") -> HMSPaths:
    csv_path = os.path.join(data_dir, f"{split}.csv")
    spec_png_dir = os.path.join(data_dir, "spec_png")
    return HMSPaths(data_dir=data_dir, csv_path=csv_path, spec_png_dir=spec_png_dir)


class HMSSpectrogramPngDataset(Dataset):
    """
    Expects:
      data_dir/
        train.csv
        spec_png/
          <spectrogram_id>.png

    The train.csv must contain vote columns:
      seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

    It also typically contains patient_id for leakage-safe splits.
    """

    def __init__(
        self,
        data_dir: str,
        split_csv: str = "train.csv",
        spec_png_subdir: str = "spec_png",
        transform=None,
        limit: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.df = pd.read_csv(os.path.join(data_dir, split_csv))
        self.spec_dir = os.path.join(data_dir, spec_png_subdir)
        self.transform = transform

        missing = [c for c in VOTE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing vote columns in {split_csv}: {missing}")

        if limit is not None:
            self.df = self.df.iloc[: int(limit)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec_id = int(row["spectrogram_id"]) if "spectrogram_id" in row else int(row["spec_id"])
        fp = os.path.join(self.spec_dir, f"{spec_id}.png")
        img = Image.open(fp).convert("L")  # 1-channel
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np[None, :, :]  # (1,H,W)

        votes = row[VOTE_COLS].values
        y = normalize_votes(votes)

        if self.transform is not None:
            # albumentations expects HWC; convert
            t = self.transform(image=img_np.transpose(1, 2, 0))
            img_np = t["image"].transpose(2, 0, 1)

        x = torch.from_numpy(img_np)
        y = torch.from_numpy(y.astype(np.float32))
        return x, y, str(spec_id)


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str = "patient_id",
    frac_val: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple group split (not full StratifiedGroupKFold).
    Ensures no patient_id overlap between train/val.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].dropna().unique()
    rng.shuffle(groups)
    n_val = int(len(groups) * frac_val)
    val_groups = set(groups[:n_val])
    is_val = df[group_col].isin(val_groups).values
    idx_val = np.where(is_val)[0]
    idx_train = np.where(~is_val)[0]
    return idx_train, idx_val
