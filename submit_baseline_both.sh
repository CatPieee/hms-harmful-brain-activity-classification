#!/bin/bash
#SBATCH --job-name=hms_baseline_both
#SBATCH --partition=gpu-h100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                     
#SBATCH --cpus-per-task=4             
#SBATCH --time=00:30:00                # Reduced time for quick training
#SBATCH --output=logs/baseline_both_%j.out
#SBATCH --error=logs/baseline_both_%j.err

set -euo pipefail
mkdir -p logs

# --- 核心修复：SLURM 作业中定位项目根目录 ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_ROOT"

# --- 激活虚拟环境 ---
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "[ERROR] 未找到虚拟环境激活脚本: $PROJECT_ROOT/.venv/bin/activate"
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
DATA_DIR="data/hms"
RUN_NAME="baseline_both"

echo "========================================"
echo "任务开始时间: $(date)"
echo "使用的 GPU: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo "数据目录: $DATA_DIR"
echo "项目目录: $PROJECT_ROOT"
echo "========================================"

# 执行快速训练 (仅 1 个 Epoch，快速生成 best.pt)
echo "Training baseline_both model for 1 epoch to generate best.pt quickly..."
python -m src.train \
    --data_dir "$DATA_DIR" \
    --model both \
    --loss kldiv \
    --epochs 1 \
    --batch_size 16 \
    --lr 1e-3 \
    --val_size 0.2 \
    --num_workers 2 \
    --output_dir "outputs/course_runs" \
    --run_name "$RUN_NAME"

echo "========================================"
echo "任务结束时间: $(date)"
echo "权重文件已保存至: outputs/course_runs/$RUN_NAME/best.pt"
echo "========================================"
