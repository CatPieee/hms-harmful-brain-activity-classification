"""
scripts/make_spec_png_from_parquet.py

Goal: Convert Kaggle spectrogram parquet files into grayscale PNGs that the
HMSSpectrogramPngDataset can load.

IMPORTANT:
- The exact parquet schema can differ by source/notebook.
- This script is designed to be *defensive*: it tries to extract the numeric
  matrix from the parquet and save it as an image.
- If your parquet has a different layout, adapt `extract_matrix()`.

Usage:
  python scripts/make_spec_png_from_parquet.py --data_dir E:\data\hms --limit 2000
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image


def extract_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristic extraction:
    - keep only numeric columns
    - drop columns that look like ids / time
    - return as (H, W) float32
    """
    num = df.select_dtypes(include=[np.number]).copy()
    # drop common metadata columns if present
    for col in ["time", "seconds", "freq", "frequency", "eeg_id", "spectrogram_id"]:
        if col in num.columns:
            num = num.drop(columns=[col])

    mat = num.to_numpy(dtype=np.float32)
    if mat.ndim != 2:
        mat = np.squeeze(mat)
    return mat


def to_uint8_image(mat: np.ndarray) -> np.ndarray:
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.log1p(np.maximum(mat, 0.0))
    # normalize
    mn, mx = float(mat.min()), float(mat.max())
    if mx - mn < 1e-6:
        return np.zeros_like(mat, dtype=np.uint8)
    mat = (mat - mn) / (mx - mn)
    return (mat * 255.0).clip(0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing train.csv and train_spectrograms/")
    ap.add_argument("--split_csv", type=str, default="train.csv")
    ap.add_argument("--spec_dir", type=str, default="train_spectrograms")
    ap.add_argument("--out_dir", type=str, default="spec_png")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    csv_path = os.path.join(args.data_dir, args.split_csv)
    df = pd.read_csv(csv_path)

    out_dir = os.path.join(args.data_dir, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ids = df["spectrogram_id"].astype(int).tolist()
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    ok = 0
    for i, sid in enumerate(ids, 1):
        fp = os.path.join(args.data_dir, args.spec_dir, f"{sid}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            spec_df = pd.read_parquet(fp)
            mat = extract_matrix(spec_df)
            img_u8 = to_uint8_image(mat)
            Image.fromarray(img_u8).save(os.path.join(out_dir, f"{sid}.png"))
            ok += 1
        except Exception as e:
            print(f"[WARN] sid={sid} failed: {e}")

        if i % 200 == 0:
            print(f"Processed {i} / {len(ids)}; wrote {ok} PNGs")

    print(f"Done. Wrote {ok} PNGs to: {out_dir}")


if __name__ == "__main__":
    main()
