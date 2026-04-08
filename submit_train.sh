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

if [ -n "${ENV_ACTIVATE:-}" ]; then
  source "$ENV_ACTIVATE"
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
else
  echo "[ERROR] 未找到虚拟环境激活脚本。"
  echo "  - 期望路径: $PROJECT_ROOT/.venv/bin/activate"
  echo "  - 或者提交前设置: ENV_ACTIVATE=/path/to/activate"
  exit 2
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-data/hms}"

echo "========================================"
echo "任务开始时间: $(date)"
echo "使用的 GPU: $CUDA_VISIBLE_DEVICES"
echo "数据目录: $DATA_DIR"
echo "项目目录: $PROJECT_ROOT"
echo "Python 路径: $(which python)"
echo "========================================"

if [ ! -d "$PROJECT_ROOT/src/data" ] && [ ! -f "$PROJECT_ROOT/src/data.py" ]; then
  echo "[ERROR] 未找到数据模块: $PROJECT_ROOT/src/data (或 src/data.py)"
  echo "请确认你在服务器上的项目目录里包含 src/data/__init__.py 和 src/data/hms_dataset.py"
  echo "当前 $PROJECT_ROOT/src 目录内容："
  ls -la "$PROJECT_ROOT/src"
  exit 3
fi

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
