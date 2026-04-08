#!/bin/bash
#SBATCH --job-name=hms_train
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G                     # 建议增加内存，训练时数据加载较多
#SBATCH --cpus-per-task=4             # 增加 CPU 核心以加速数据加载 (num_workers)
#SBATCH --time=12:00:00                
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

mkdir -p logs

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-data/hms}"

echo "========================================"
echo "任务开始时间: $(date)"
echo "使用的 GPU: $CUDA_VISIBLE_DEVICES"
echo "数据目录: $DATA_DIR"
echo "========================================"

# 3. 执行训练 (以多模态模型为例)
python -m src.train \
    --data_dir "$DATA_DIR" \
    --model both \
    --loss kldiv \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-3 \
    --fold 0 \
    --num_workers 4 \
    --output_dir outputs

echo "========================================"
echo "任务结束时间: $(date)"
echo "========================================"
