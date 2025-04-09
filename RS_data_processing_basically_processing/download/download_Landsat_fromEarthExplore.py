import pandas as pd
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import os
import json
from datetime import datetime
from shapely.geometry import Polygon

# 自定义JSON编码器以处理datetime和Polygon对象
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Polygon):
            return obj.__geo_interface__
        return super(CustomEncoder, self).default(obj)
    
# 登录Earth Explorer
username = 'jmen@ua.edu'
password = 'Mjl19931209~'
api = API(username, password)

# 读取CSV文件
df = pd.read_csv(r'H:\GLORIA_PAR\Sampling_date.csv')
output_dir = r'H:\GLORIA_PAR\L1'
unique_scenes_file = os.path.join(output_dir,'unique_scenes.json')

# 检查是否存在unique_scenes文件
if os.path.exists(unique_scenes_file):
    print('unique_scene exists, reading')
    with open(unique_scenes_file, 'r') as f:
        unique_scenes = json.load(f)
else:
    print('creating unique_scenes list')
    all_scenes = []
    
    # 假设CSV文件有 'latitude', 'longitude', 'date' 三列
    for index, row in df.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']
        date = row['Date']
    
        # 搜索Landsat 8数据
        scenes = api.search(
            dataset='landsat_ot_c2_l1',
            latitude=latitude,
            longitude=longitude,
            start_date=date,
            end_date=date,
            max_cloud_cover=80
        )
        
        print(f'{len(scenes)} scenes found for {date} at ({latitude}, {longitude}).')
        
        # 添加到列表中
        all_scenes.extend(scenes)
        
    # 去除重复的场景
    unique_scenes = {scene['entity_id']: scene for scene in all_scenes}.values()
    print(f'{len(unique_scenes)} unique scenes found.')
    
    # 保存unique_scenes到本地文件
    
    with open(unique_scenes_file, 'w') as f:
        json.dump(list(unique_scenes), f, cls=CustomEncoder)
    
    print(f'{len(unique_scenes)} unique scenes found and saved to {unique_scenes_file}.')

# 下载数据
ee = EarthExplorer(username, password)

for scene in unique_scenes:
    file_path = os.path.join(output_dir, f"{scene['entity_id']}.tar.gz")
    if not os.path.exists(file_path):
        try:
            ee.download(scene['entity_id'], output_dir=output_dir)
        except Exception as e:
            print(e)
            
    else:
        print(f"File {file_path} already exists, skipping download.")
# 退出
ee.logout()
api.logout()
