import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import MultiModalNet, EEGOnlyNet, SpecOnlyNet

def test_forward_pass(batch_size=4):
    print(f"Testing forward pass with batch_size={batch_size}...")
    device = torch.device("cpu")
    
    # Mock data dimensions based on train.py defaults
    # eeg_len=5000, target_h=600, target_w=400, in_ch=16
    eeg_in_ch = 16
    eeg_len = 5000
    spec_h = 600
    spec_w = 400
    n_classes = 6
    
    # Create dummy inputs
    eeg_input = torch.randn(batch_size, eeg_in_ch, eeg_len).to(device)
    spec_input = torch.randn(batch_size, 1, spec_h, spec_w).to(device)
    
    print(f"EEG input shape: {eeg_input.shape}")
    print(f"Spec input shape: {spec_input.shape}")
    
    # 1. Test MultiModalNet
    print("\n--- Testing MultiModalNet ---")
    model = MultiModalNet(n_classes=n_classes, eeg_in_ch=eeg_in_ch, pretrained_spec=False).to(device)
    model.eval()
    with torch.no_grad():
        output = model(eeg_input, spec_input)
    print(f"MultiModalNet output shape: {output.shape}")
    assert output.shape == (batch_size, n_classes), f"Expected {(batch_size, n_classes)}, got {output.shape}"
    
    # 2. Test EEGOnlyNet
    print("\n--- Testing EEGOnlyNet ---")
    model_eeg = EEGOnlyNet(n_classes=n_classes, in_ch=eeg_in_ch).to(device)
    model_eeg.eval()
    with torch.no_grad():
        output_eeg = model_eeg(eeg_input)
    print(f"EEGOnlyNet output shape: {output_eeg.shape}")
    assert output_eeg.shape == (batch_size, n_classes)
    
    # 3. Test SpecOnlyNet
    print("\n--- Testing SpecOnlyNet ---")
    model_spec = SpecOnlyNet(n_classes=n_classes, in_ch=1, pretrained=False).to(device)
    model_spec.eval()
    with torch.no_grad():
        output_spec = model_spec(spec_input)
    print(f"SpecOnlyNet output shape: {output_spec.shape}")
    assert output_spec.shape == (batch_size, n_classes)
    
    print("\nForward pass verification SUCCESSFUL!")

if __name__ == "__main__":
    # Test with batch size 2 and 4 as requested
    try:
        test_forward_pass(batch_size=2)
        print("-" * 30)
        test_forward_pass(batch_size=4)
    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
