import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_model, build_criterion
from src.metrics import kl_divergence_from_logits

def debug_nan():
    print("=== HMS NaN Debugger ===")
    device = torch.device("cpu")
    
    # 1. Build Model and Criterion
    print("\n1. Building model and criterion...")
    try:
        model = build_model("both", eeg_in_ch=16, pretrained_spec=False).to(device)
        criterion = build_criterion("kldiv")
        print("Model and Criterion built successfully.")
    except Exception as e:
        print(f"Error building model/criterion: {e}")
        return

    # 2. Test with Normal Data
    print("\n2. Testing with normal random data...")
    eeg = torch.randn(2, 16, 5000)
    spec = torch.randn(2, 1, 600, 400)
    target = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]], dtype=torch.float32)
    
    logits = model(eeg, spec)
    loss = criterion(logits, target)
    print(f"Normal Data - Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"Normal Data - Loss: {loss.item():.4f}")
    
    if torch.isnan(loss):
        print("!! ALERT: Loss is NaN even with normal data !!")

    # 3. Test with Extreme Data (Potential NaN sources)
    print("\n3. Testing with extreme data (simulating artifacts)...")
    eeg_extreme = torch.randn(2, 16, 5000) * 1000.0 # Huge voltage
    spec_extreme = torch.randn(2, 1, 600, 400) * 100.0
    
    logits_ex = model(eeg_extreme, spec_extreme)
    loss_ex = criterion(logits_ex, target)
    print(f"Extreme Data - Logits range: [{logits_ex.min().item():.4f}, {logits_ex.max().item():.4f}]")
    print(f"Extreme Data - Loss: {loss_ex.item():.4f}")

    # 4. Test KL Divergence separately
    print("\n4. Testing KL Divergence with edge case targets...")
    # Pure zero targets often cause issues
    target_edge = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0]], dtype=torch.float32)
    logits_edge = torch.randn(2, 6) * 10.0
    
    kld = kl_divergence_from_logits(logits_edge, target_edge)
    print(f"Edge Case Targets - KLD: {kld.item():.4f}")
    
    if torch.isnan(kld):
        print("!! ALERT: KLD calculation produced NaN !!")
    else:
        print("KLD calculation is stable.")

    print("\n=== Debugging Finished ===")

if __name__ == "__main__":
    debug_nan()
