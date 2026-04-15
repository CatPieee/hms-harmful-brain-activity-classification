#!/bin/bash
#SBATCH --job-name=hms_test
#SBATCH --partition=gpu-rtx4090
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G                     # 增加内存，4G 对训练太小了
#SBATCH --cpus-per-task=2             # 减少 CPU 核心以适应 4G 内存
#SBATCH --time=2:00:00                
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# Configuration
DATA_DIR="data/hms"
CHECKPOINT="outputs/course_runs/baseline_both/best.pt"
OUTPUT_FILE="submission.csv"

# Run inference
echo "Running HMS inference on test set..."
python3 src/inference.py \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --batch_size 16 \
    --num_workers 2 \
    --output_file "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Inference completed successfully. Submission file generated at $OUTPUT_FILE"
else
    echo "Inference failed."
    exit 1
fi
