#!/bin/bash
#SBATCH --job-name=hms_data_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --partition=cpu-intel3t          # 请根据您的 HPC 分区名称修改，如 'cpu' 或 'standard'
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=05:00:00               # 预计下载和解压时间

source .venv/bin/activate

echo "开始数据预处理"

# Generate spectrogram PNG files
python scripts/make_spec_png_from_parquet.py --data_dir data/hms --out_dir spec_png --target_h 600 --target_w 400

# Generate EEG cache files
python scripts/make_eeg_npy_from_parquet.py --data_dir data/hms --out_dir eeg_npy --max_files 5000

echo "完成数据预处理"
