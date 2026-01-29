"""
scripts/verify_env.py

Quick environment verification:
- Python version
- torch import
- CUDA availability
"""

import platform
import sys

def main() -> None:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    try:
        import torch
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("cuda device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("ERROR importing torch:", repr(e))
        print("Fix: install PyTorch from https://pytorch.org/get-started/locally/")
        raise

if __name__ == "__main__":
    main()
