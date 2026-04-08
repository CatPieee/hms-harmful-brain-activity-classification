import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Convert HMS EEG parquet to cached NPY")
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing train_eegs/")
    p.add_argument("--in_dir", type=str, default="train_eegs")
    p.add_argument("--out_dir", type=str, default="eeg_npy")
    p.add_argument("--max_files", type=int, default=0, help="0 = all files")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.data_dir) / args.in_dir
    out_dir = Path(args.data_dir) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.parquet"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    for fp in tqdm(files, desc="eeg->npy"):
        eid = fp.stem
        out_fp = out_dir / f"{eid}.npy"
        if out_fp.exists():
            continue

        df = pd.read_parquet(fp)
        x = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)  # (T,C)
        if x.ndim != 2:
            continue
        x = x.T  # (C,T)
        np.save(out_fp, x)

    print("Done. eeg_npy at:", out_dir)


if __name__ == "__main__":
    main()
