#!/bin/bash
#SBATCH --job-name=hms_train      # 任务名称
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G                      
#SBATCH --time=12:00:00                
#SBATCH --output=logs/train_%j.out

echo "开始模型训练"

python -m src.train       --data_dir data/hms      --model both       --loss kldiv       --epochs 10       --batch_size 16       --lr 1e-3       --fold 0       --output_dir outputs

echo "模型训练完毕"
