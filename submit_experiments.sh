#!/bin/bash
#SBATCH --job-name=hms_course_experiments
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G                     # 运行多个实验建议分配更多内存
#SBATCH --cpus-per-task=8             # 增加 CPU 以加速多轮实验的数据加载
#SBATCH --time=24:00:00               # 运行所有消融实验可能需要较长时间
#SBATCH --output=logs/course_exp_%j.out
#SBATCH --error=logs/course_exp_%j.err

# 1. 环境准备
source .venv/bin/activate

DATA_DIR="data"
OUTPUT_DIR="outputs/course_runs"

echo "========================================"
echo "课程全自动化实验开始: $(date)"
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 2. 运行课程要求的全系列实验
# 包括: Baseline, Loss 对比, Learning Rate Sweep, Batch Size Sweep, First 100 预测可视化
python scripts/run_course_experiments.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --model both \
    --epochs 10 \
    --num_workers 8 \
    --pretrained_spec

echo "========================================"
echo "所有实验已完成: $(date)"
echo "请查看 $OUTPUT_DIR 目录下的 PNG 和 CSV 文件。"
echo "========================================"
