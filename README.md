# CDS525 Group Project — Harmful Brain Activity Classification (PyTorch)

This repository is a competition‑style deep learning pipeline for the Kaggle
**HMS – Harmful Brain Activity Classification** dataset.

## 1) What you must deliver (per course instruction)
You need to provide:
- A written report **>= 2000 words**
- Source code organized cleanly
- Performance visualizations:
  1) Training loss, training accuracy, and test/validation accuracy vs epochs
  2) Same plot but using a *different loss function*
  3) Same plot but using *different learning rates* (e.g., 0.1 / 0.01 / 0.001 / 0.0001)
  4) Same plot but using *different batch sizes* (e.g., 8 / 16 / 32 / 64 / 128)
  5) A visualization (figure or table) of the **first 100 test samples**:
     predicted label + ground-truth label + corresponding inputs

These requirements come directly from the course project instruction.

## 2) Recommended environment
- Windows 10/11
- Python 3.10+ (3.11 also OK)
- Git
- Optional: NVIDIA GPU + CUDA for faster training

## 3) Project structure

```
harmful_brain_activity_project/
  src/
    datasets.py
    models.py
    train.py
    experiments.py
    utils.py
  scripts/
    verify_env.py
  outputs/
  requirements.txt
  README.md
```

## 4) Quick start (CPU)
```bash
# 1) create venv
python -m venv .venv
# 2) activate (PowerShell)
.\.venv\Scripts\Activate.ps1
# 3) upgrade pip
python -m pip install -U pip
# 4) install deps (CPU PyTorch)
pip install -r requirements.txt
# 5) verify environment
python scripts/verify_env.py
# 6) run a small smoke test using synthetic data
python -m src.train --synthetic --epochs 3 --batch_size 32 --lr 1e-3 --out_dir outputs/smoke
```

## 5) Installing PyTorch properly (GPU vs CPU)
PyTorch wheels differ by CUDA version. Use the official “Get Started” selector.
If you are using:
- CPU only: install CPU wheel
- NVIDIA GPU: install the CUDA wheel matching your installed CUDA runtime

This repository’s `requirements.txt` keeps PyTorch separate so you can install
the correct one first.

## 6) Using the Kaggle dataset
1. Join the Kaggle competition and accept rules.
2. Install Kaggle API:
   ```bash
   pip install kaggle
   ```
3. Put your `kaggle.json` token in:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - WSL/Linux: `~/.kaggle/kaggle.json`
4. Download:
   ```bash
   kaggle competitions download -c hms-harmful-brain-activity-classification
   ```
5. Unzip under:
   ```
   data/hms/
     train.csv
     train_eegs/
     train_spectrograms/
     test.csv
     ...
   ```

## 7) Running experiments for required figures
You will generate the required plots by running:

```bash
# baseline
python -m src.experiments --data_dir data/hms --exp baseline

# different loss function
python -m src.experiments --data_dir data/hms --exp loss_compare

# different learning rates
python -m src.experiments --data_dir data/hms --exp lr_sweep

# different batch sizes
python -m src.experiments --data_dir data/hms --exp bs_sweep

# first 100 test predictions table/figure
python -m src.experiments --data_dir data/hms --exp first100
```

All results will be saved in `outputs/` as PNGs + CSV logs.

## 8) Notes for “competition level”
- Use vote distributions + KLDiv loss as the primary objective
- Try spectrogram-only baseline first; then add EEG waveform branch
- Use StratifiedGroupKFold by `patient_id` to avoid leakage
- Use strong augmentations on spectrograms (CutMix / MixUp)
- Calibrate label smoothing and vote weighting (more votes => higher weight)
