import os
from kaggle.api.kaggle_api_extended import KaggleApi
 
# 显式设置配置目录（适用于Jupyter或非标准环境）
os.environ['KAGGLE_CONFIG_DIR'] = '/Users/ruiling/.kaggle'
 
api = KaggleApi()
try:
    api.authenticate()
    print("✅ Kaggle API认证成功")
    print("当前认证用户名:", api.get_config_value('username'))
    print("当前认证密钥:", api.get_config_value('key'))
    # print("用户:", api.config_value('username'))

    # 在刚才成功的脚本后面添加
    api.competition_download_files('hms-harmful-brain-activity-classification')
    print("下载指令已发送")
    
except Exception as e:
    print(f"❌ 认证失败: {str(e)}")
