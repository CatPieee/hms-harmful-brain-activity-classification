#!/bin/bash
#SBATCH --job-name=hms_data_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=cpu-intel3t
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00

# 1. 环境准备
source .venv/bin/activate

# 2. 检查 Kaggle API Key
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "错误: 未找到 $HOME/.kaggle/kaggle.json 文件。"
    echo "请先在本地下载 kaggle.json 并上传到服务器的 ~/.kaggle/ 目录下，并执行 chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# 3. 下载并解压
mkdir -p data
cd data

echo "========================================"
echo "开始下载数据: $(date)"
echo "========================================"

# 使用直接安装的 kaggle 命令
kaggle competitions download -c hms-harmful-brain-activity-classification

echo "正在解压数据..."
unzip -q hms-harmful-brain-activity-classification.zip
rm hms-harmful-brain-activity-classification.zip

echo "========================================"
echo "数据下载与解压完成: $(date)"
echo "========================================"
ls -lh
