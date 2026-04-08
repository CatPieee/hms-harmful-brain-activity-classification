#!/bin/bash
#SBATCH --job-name=hms_data_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --partition=cpu-intel3t          # 请根据您的 HPC 分区名称修改，如 'cpu' 或 'standard'
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G                     # 数据集较大，解压需要一定内存
#SBATCH --time=05:00:00               # 预计下载和解压时间

# 1. 创建日志目录
mkdir -p logs
mkdir -p data

# 2. 加载 Python 模块 (根据您的 HPC 实际模块名称修改)
# module load python/3.10 

# 3. 激活虚拟环境 (假设您已经在项目根目录创建了 .venv)
source .venv/bin/activate

# 4. 确保安装了 kaggle 客户端
pip install kaggle

# 5. 检查 Kaggle API Key 是否存在
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "错误: 未找到 $HOME/.kaggle/kaggle.json 文件。"
    echo "请先在本地下载 kaggle.json 并上传到服务器的 ~/.kaggle/ 目录下，并执行 chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "开始下载 HMS Harmful Brain Activity Classification 数据集..."

# 6. 进入数据目录并下载
cd data
# python -m kaggle competitions download -c hms-harmful-brain-activity-classification
kaggle competitions download -c hms-harmful-brain-activity-classification

# 7. 解压数据
echo "正在解压数据 (这可能需要一些时间)..."
unzip -q hms-harmful-brain-activity-classification.zip

# 8. 清理压缩包
echo "正在清理压缩包..."
rm hms-harmful-brain-activity-classification.zip

echo "数据准备完成！"
ls -lh .