#!/bin/bash
#SBATCH --job-name=hms_preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --partition=cpu-intel3t
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=05:00:00

# 1. 环境准备
source .venv/bin/activate

DATA_DIR="data/hms"

echo "========================================"
echo "预处理开始时间: $(date)"
echo "数据目录: $DATA_DIR"
echo "========================================"

# 2. 生成频谱图 PNG 图像
# 将 .parquet 转换为 .png，以便模型更快加载
echo "正在生成 Spectrogram PNG 文件..."
python scripts/make_spec_png_from_parquet.py \
    --data_dir $DATA_DIR \
    --out_dir spec_png \
    --target_h 600 \
    --target_w 400

# 3. 生成 EEG 缓存文件
# 将 .parquet 转换为 .npy 格式，大幅提升训练速度
echo "正在生成 EEG 缓存文件..."
python scripts/make_eeg_npy_from_parquet.py \
    --data_dir $DATA_DIR \
    --out_dir eeg_npy \
    --max_files 0  # 0 表示处理所有文件

echo "========================================"
echo "预处理完成时间: $(date)"
echo "========================================"
