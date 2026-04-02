# CDS525 Group Project — Harmful Brain Activity Classification (Competition-Level Package)

This package is a cleaned and upgraded submission-ready version of the project for the **HMS – Harmful Brain Activity Classification** topic.

It is designed to satisfy the course requirements while also improving the technical level of the original work:
- **Multimodal model support**: EEG 1D + Spectrogram 2D fusion
- **Single-modality baselines**: EEG-only or Spectrogram-only
- **Course-required experiment automation**
- **Cleaner submission structure**
- **Leakage-aware validation split** using `patient_id` first
- **First-100 prediction export** for the held-out validation set

## What was improved in this version
1. Removed submission-unfriendly files such as `.venv`, `__MACOSX`, `.DS_Store`, `.idea`, and oversized intermediate junk.
2. Upgraded the training code so it records:
   - training loss
   - training accuracy
   - validation accuracy
   - validation KL divergence
3. Added **three model modes**:
   - `both` = multimodal fusion (recommended final model)
   - `spec` = spectrogram only
   - `eeg` = EEG only
4. Added a one-command course experiment runner:
   - baseline
   - loss comparison
   - learning-rate sweep
   - batch-size sweep
   - first-100 prediction export
5. Fixed the split logic so the code prefers `patient_id` over `eeg_id`, which is safer for leakage control.

## Submission folder naming
The instructor requires a folder named like:

`Group Assignment - Your Group Name`

This zip already uses that format. If you want to match your exact group name, rename the top-level folder before final Moodle submission.

## Recommended directory layout for your dataset
```
<DATA_DIR>/
  train.csv
  train_eegs/
    *.parquet
  train_spectrograms/
    *.parquet
  spec_png/
    *.png
  eeg_npy/
    *.npy   # optional cache, auto-generated
```

## Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python scripts/verify_env.py
```

## Preprocessing
Generate spectrogram PNG files:
```bash
python scripts/make_spec_png_from_parquet.py --data_dir <DATA_DIR> --out_dir spec_png --target_h 600 --target_w 400
```

Generate EEG cache files:
```bash
python scripts/make_eeg_npy_from_parquet.py --data_dir <DATA_DIR> --out_dir eeg_npy --max_files 5000
```

## Recommended training run
Use the multimodal model as the final showcase model:
```bash
python -m src.train       --data_dir <DATA_DIR>       --model both       --loss kldiv       --epochs 10       --batch_size 16       --lr 1e-3       --fold 0       --output_dir outputs
```

## One-command course experiments
```bash
python scripts/run_course_experiments.py       --data_dir <DATA_DIR>       --output_dir outputs/course_runs       --model both       --epochs 10
```

This will generate the following deliverables under `outputs/course_runs/`:
- `baseline_kldiv/curves.png`
- `loss_kldiv/curves.png`
- `loss_ce/curves.png`
- `lr_sweep_loss.png`
- `lr_sweep_accuracy.png`
- `bs_sweep_loss.png`
- `bs_sweep_accuracy.png`
- `first100/first100_predictions.csv`
- `first100/first100_grid.png`

## Important note on “test accuracy”
In the Kaggle HMS competition, the public `test.csv` does **not** include labels, so a true test-set accuracy cannot be computed locally.
For coursework, this package uses a **held-out validation fold** as the evaluation set. In the report and presentation, describe it explicitly as a validation split used as the course evaluation set.

## Suggested presentation angle
For a stronger presentation, frame your model progression like this:
1. Problem difficulty: EEG signals are noisy and spectrograms preserve time-frequency patterns.
2. Baseline: single-modality spectrogram model.
3. Upgrade: multimodal fusion with gated fusion.
4. Objective: KL divergence aligns better with soft-vote targets than plain CE.
5. Reliability: patient-level split reduces leakage risk.

## Files added for your group
- `docs/improvement_audit.md`: what was weak and what was fixed
- `docs/report_outline.md`: a clean 2000+ word report structure
- `docs/submission_checklist.md`: last-minute submission checklist
