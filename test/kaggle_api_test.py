import os
from kaggle.api.kaggle_api_extended import KaggleApi
 
# 显式设置配置目录（适用于Jupyter或非标准环境）
os.environ['KAGGLE_CONFIG_DIR'] = '/Users/ruiling/.kaggle'
 
api = KaggleApi()
try:
    api.authenticate()
    print("✅ Kaggle API authentication successful")
    print("Current authenticated username:", api.get_config_value('username'))
    print("Current authenticated key:", api.get_config_value('key'))
    # print("User:", api.config_value('username'))

    # 在刚才成功的脚本后面添加
    api.competition_download_files('hms-harmful-brain-activity-classification')
    print("✅ Competition files downloaded successfully")
    
except Exception as e:
    print(f"❌ Authentication failed: {str(e)}")
