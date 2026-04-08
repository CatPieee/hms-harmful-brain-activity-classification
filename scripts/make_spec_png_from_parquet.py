import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Convert HMS spectrogram parquet to PNG (fixed size)")
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing train_spectrograms/")
    p.add_argument("--in_dir", type=str, default="train_spectrograms")
    p.add_argument("--out_dir", type=str, default="spec_png")
    p.add_argument("--target_h", type=int, default=600)
    p.add_argument("--target_w", type=int, default=400)
    p.add_argument("--limit", type=int, default=0, help="0 = all files")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.data_dir) / args.in_dir
    out_dir = Path(args.data_dir) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    for fp in tqdm(files, desc="spec->png"):
        sid = fp.stem
        out_fp = out_dir / f"{sid}.png"
        if out_fp.exists():
            continue

        df = pd.read_parquet(fp)
        arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if arr.ndim != 2:
            continue

        # log scaling + normalize to 0..255
        arr = np.log1p(np.maximum(arr, 0))
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        img = img.resize((args.target_w, args.target_h))
        img.save(out_fp)

    print("Done. spec_png at:", out_dir)


if __name__ == "__main__":
    main()
