#!/bin/bash

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
