#!/bin/bash
#SBATCH --job-name=hms_course_experiments
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G                     # 运行多个实验建议分配更多内存 (尤其 Batch Size 128)
#SBATCH --cpus-per-task=8             # 增加 CPU 以加速多轮实验的数据加载
#SBATCH --time=24:00:00               # 运行所有消融实验可能需要较长时间
#SBATCH --output=logs/course_exp_%j.out
#SBATCH --error=logs/course_exp_%j.err

set -euo pipefail

mkdir -p logs

# --- 核心修复：SLURM 作业中定位项目根目录 ---
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
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
OUTPUT_DIR="outputs/course_runs"

echo "========================================"
echo "课程全自动化实验开始: $(date)"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "Python 路径: $(which python)"
echo "========================================"

if [ ! -d "$PROJECT_ROOT/src/data" ] && [ ! -f "$PROJECT_ROOT/src/data.py" ]; then
  echo "[ERROR] 未找到数据模块: $PROJECT_ROOT/src/data (或 src/data.py)"
  echo "请确认你在服务器上的项目目录里包含 src/data/__init__.py 和 src/data/hms_dataset.py"
  echo "当前 $PROJECT_ROOT/src 目录内容："
  ls -la "$PROJECT_ROOT/src"
  exit 3
fi

# 2. 运行课程要求的全系列实验
# 包括: Baseline, Loss 对比, Learning Rate Sweep, Batch Size Sweep, First 100 预测可视化
python scripts/run_course_experiments.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model both \
    --epochs 10 \
    --val_size 0.2 \
    --num_workers 8 \
    --pretrained_spec

echo "========================================"
echo "所有实验已完成: $(date)"
echo "请查看 $OUTPUT_DIR 目录下的 PNG 和 CSV 文件。"
echo "========================================"
