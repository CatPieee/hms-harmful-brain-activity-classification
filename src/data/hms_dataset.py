from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold


VOTE_COLS_DEFAULT = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _find_vote_cols(df: pd.DataFrame) -> List[str]:
    # Prefer canonical vote columns; otherwise pick columns ending with "_vote"
    if all(c in df.columns for c in VOTE_COLS_DEFAULT):
        return VOTE_COLS_DEFAULT
    vote_cols = [c for c in df.columns if c.endswith("_vote")]
    if len(vote_cols) >= 6:
        return vote_cols[:6]
    raise ValueError("Could not find vote columns in train.csv.")


def _pad_crop_h(arr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = arr.shape
    if h == target_h:
        return arr
    if h < target_h:
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        return np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode="constant", constant_values=0.0)
    # h > target_h
    start = (h - target_h) // 2
    return arr[start:start + target_h, :]


def load_spec_png(png_path: str, target_h: int = 600, target_w: int = 400) -> np.ndarray:
    img = Image.open(png_path).convert("L")  # HxW
    if img.size[0] != target_w:
        img = img.resize((target_w, img.size[1]))
    arr = (np.asarray(img).astype(np.float32) / 255.0)  # H,W
    arr = _pad_crop_h(arr, target_h)
    return arr  # H,W


def load_eeg_from_parquet(parquet_path: str) -> np.ndarray:
    # Uses pyarrow via pandas
    df = pd.read_parquet(parquet_path)
    # Many Kaggle EEG parquet files have time index + channels; select numeric columns
    data = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    # Shape: (T, C). Return (C, T)
    if data.ndim != 2:
        raise ValueError(f"Unexpected EEG parquet shape: {data.shape}")
    return data.T


def crop_or_pad_1d(x: np.ndarray, target_len: int, train: bool = True) -> np.ndarray:
    # x: (C, T)
    c, t = x.shape
    if t == target_len:
        return x
    if t < target_len:
        pad_left = (target_len - t) // 2
        pad_right = target_len - t - pad_left
        return np.pad(x, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0.0)
    # t > target_len
    if train:
        start = np.random.randint(0, t - target_len + 1)
    else:
        start = (t - target_len) // 2
    return x[:, start:start + target_len]


def build_splits(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits the dataframe into train and validation sets (default 80/20).
    Ensures patient-level separation to prevent data leakage.
    """
    group_col = None
    if "patient_id" in df.columns:
        group_col = "patient_id"
    elif "eeg_id" in df.columns:
        group_col = "eeg_id"

    if group_col is None:
        idx = np.arange(len(df))
        np.random.seed(seed)
        np.random.shuffle(idx)
        split = int((1 - val_size) * len(idx))
        return idx[:split], idx[split:]

    # Group-aware split using GroupKFold to get a single 80/20 split
    # (n_splits = 1/val_size)
    n_splits = int(1.0 / val_size)
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    
    # Just take the first split
    train_idx, val_idx = next(gkf.split(df, groups=groups))
    return train_idx, val_idx


@dataclass
class HMSPaths:
    data_dir: str
    spec_png_dirname: str = "spec_png"
    train_eegs_dirname: str = "train_eegs"
    eeg_cache_dirname: str = "eeg_npy"


class HMSDataset:
    """
    Multimodal dataset returning (eeg, spec, target_prob).
    Expects:
      - train.csv in data_dir
      - spec_png/<spectrogram_id>.png  (recommended)
      - train_eegs/<eeg_id>.parquet
    Optionally caches EEG as .npy in eeg_npy/ to speed up Windows training.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        paths: HMSPaths,
        indices: np.ndarray,
        train: bool = True,
        target_h: int = 600,
        target_w: int = 400,
        eeg_len: int = 5000,
        normalize_eeg: bool = True,
    ):
        self.df = df.iloc[indices].reset_index(drop=True)
        self.paths = paths
        self.train = train
        self.target_h = target_h
        self.target_w = target_w
        self.eeg_len = eeg_len
        self.normalize_eeg = normalize_eeg
        self.vote_cols = _find_vote_cols(df)

        self.spec_dir = os.path.join(paths.data_dir, paths.spec_png_dirname)
        self.eeg_dir = os.path.join(paths.data_dir, paths.train_eegs_dirname)
        self.eeg_cache = os.path.join(paths.data_dir, paths.eeg_cache_dirname)
        os.makedirs(self.eeg_cache, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    def _get_target(self, row: pd.Series) -> np.ndarray:
        votes = row[self.vote_cols].to_numpy(dtype=np.float32)
        s = votes.sum()
        if s <= 0:
            return np.ones_like(votes) / len(votes)
        return votes / s

    def _load_spec(self, spectrogram_id: int) -> np.ndarray:
        fp = os.path.join(self.spec_dir, f"{int(spectrogram_id)}.png")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Missing spectrogram PNG: {fp}\n"
                "Run: python scripts/make_spec_png_from_parquet.py --data_dir <DATA> --out_dir spec_png"
            )
        return load_spec_png(fp, target_h=self.target_h, target_w=self.target_w)

    def _load_eeg(self, eeg_id: int) -> np.ndarray:
        cache_fp = os.path.join(self.eeg_cache, f"{int(eeg_id)}.npy")
        x: Optional[np.ndarray] = None

        if os.path.exists(cache_fp):
            try:
                # Check if file is empty
                if os.path.getsize(cache_fp) > 0:
                    x = np.load(cache_fp).astype(np.float32)
                else:
                    # Remove empty file
                    os.remove(cache_fp)
            except (EOFError, ValueError, OSError) as e:
                # Corrupted file, remove it
                print(f"[Warning] Corrupted cache file detected and removed: {cache_fp}. Error: {e}")
                if os.path.exists(cache_fp):
                    os.remove(cache_fp)

        if x is None:
            pq = os.path.join(self.eeg_dir, f"{int(eeg_id)}.parquet")
            if not os.path.exists(pq):
                raise FileNotFoundError(f"Missing EEG parquet: {pq}")
            x = load_eeg_from_parquet(pq)  # (C,T)
            # Save valid cache for next time
            try:
                np.save(cache_fp, x)
            except OSError:
                # If disk full or no permission, just proceed without caching
                pass
        
        x = crop_or_pad_1d(x, self.eeg_len, train=self.train)
        if self.normalize_eeg:
            # per-channel zscore
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mu) / sd
        return x

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        row = self.df.iloc[i]
        eeg_id = row["eeg_id"] if "eeg_id" in row else row.get("eeg_id", i)
        spec_id = row["spectrogram_id"] if "spectrogram_id" in row else row.get("spectrogram_id", i)

        eeg = self._load_eeg(eeg_id)  # (C,T)
        spec = self._load_spec(spec_id)  # (H,W)
        target = self._get_target(row)  # (6,)

        # to tensor-friendly shapes
        eeg = eeg.astype(np.float32)
        spec = spec[None, ...].astype(np.float32)  # (1,H,W)

        return {"eeg": eeg, "spec": spec, "target": target, "eeg_id": int(eeg_id), "spec_id": int(spec_id)}
