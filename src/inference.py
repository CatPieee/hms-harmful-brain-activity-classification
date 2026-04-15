import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from src.data import HMSDataset, HMSPaths
from src.models import MultiModalNet, EEGOnlyNet, SpecOnlyNet
from src.utils import get_device

# Import script logic to generate PNGs for test
from scripts.make_spec_png_from_parquet import main as generate_pngs

def parse_args():
    p = argparse.ArgumentParser("HMS Inference Script")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--target_h", type=int, default=600)
    p.add_argument("--target_w", type=int, default=400)
    p.add_argument("--eeg_len", type=int, default=5000)
    p.add_argument("--output_file", type=str, default="submission.csv")
    return p.parse_args()

def build_model(model_name: str, eeg_in_ch: int, pretrained_spec: bool) -> nn.Module:
    if model_name == "both":
        return MultiModalNet(n_classes=6, eeg_in_ch=eeg_in_ch, eeg_feat=256, spec_in_ch=1, spec_feat=256, pretrained_spec=pretrained_spec, drop=0.0)
    if model_name == "spec":
        return SpecOnlyNet(n_classes=6, in_ch=1, feat_dim=256, pretrained=pretrained_spec, drop=0.0)
    if model_name == "eeg":
        return EEGOnlyNet(n_classes=6, in_ch=eeg_in_ch, feat_dim=256, drop=0.0)
    raise ValueError(f"Unsupported model: {model_name}")

def main():
    args = parse_args()
    device = get_device()

    test_csv = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(test_csv):
        # On local PC for debug, if test.csv missing, we might not proceed.
        # But for Kaggle submission, it will be there.
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")

    df_test = pd.read_csv(test_csv)
    
    # 1. Generate PNGs for test spectrograms
    # We use a temporary or specific directory for test PNGs
    test_spec_png_dir = "test_spec_png"
    print(f"Generating test spectrogram PNGs in {os.path.join(args.data_dir, test_spec_png_dir)}...")
    
    # Mock sys.argv to call generate_pngs
    old_argv = sys.argv
    sys.argv = [
        "make_spec_png_from_parquet.py",
        "--data_dir", args.data_dir,
        "--in_dir", "test_spectrograms",
        "--out_dir", test_spec_png_dir,
        "--target_h", str(args.target_h),
        "--target_w", str(args.target_w)
    ]
    try:
        generate_pngs()
    finally:
        sys.argv = old_argv

    # 2. Setup Dataset & Loader
    paths = HMSPaths(
        data_dir=args.data_dir,
        spec_png_dirname=test_spec_png_dir,
        eeg_dirname="test_eegs",
        eeg_cache_dirname="test_eeg_npy"
    )
    
    ds_test = HMSDataset(
        df_test, paths, 
        train=False, 
        target_h=args.target_h, 
        target_w=args.target_w, 
        eeg_len=args.eeg_len
    )

    def collate(batch):
        out = {}
        for k in ["eeg", "spec"]:
            out[k] = np.stack([b[k] for b in batch], axis=0)
        out["eeg_id"] = [b["eeg_id"] for b in batch]
        return out

    loader_test = torch.utils.data.DataLoader(
        ds_test, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate
    )

    # 3. Load Model
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["model_name"]
    model_args = ckpt["args"]
    
    # Check EEG input channels from a sample
    sample_eeg = ds_test[0]["eeg"]
    eeg_in_ch = int(sample_eeg.shape[0])
    
    model = build_model(model_name, eeg_in_ch, pretrained_spec=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 4. Inference
    print("Starting inference...")
    all_preds = []
    eeg_ids = []
    
    vote_cols = [
        "seizure_vote", "lpd_vote", "gpd_vote", 
        "lrda_vote", "grda_vote", "other_vote"
    ]

    with torch.no_grad():
        for batch in tqdm(loader_test, desc="Inference"):
            eeg = torch.from_numpy(batch["eeg"]).to(device)
            spec = torch.from_numpy(batch["spec"]).to(device)
            
            if model_name == "both":
                logits = model(eeg, spec)
            elif model_name == "spec":
                logits = model(spec)
            else:
                logits = model(eeg)
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.append(probs)
            eeg_ids.extend(batch["eeg_id"])

    # 5. Save submission
    predictions = np.concatenate(all_preds, axis=0)
    sub_df = pd.DataFrame({"eeg_id": eeg_ids})
    for i, col in enumerate(vote_cols):
        sub_df[col] = predictions[:, i]
    
    sub_df.to_csv(args.output_file, index=False)
    print(f"Submission saved to {args.output_file}")

if __name__ == "__main__":
    main()
