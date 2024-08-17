#  Copyright (c) 2024. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

"""
@Project: 2023-GCN-action-recognize-tutorial
@FileName: visualization.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2024/7/17 10:52 at PyCharm
"""
import pickle

with open(r"D:\temporal_download\k400_2d.pkl", 'rb') as file:
    data = pickle.load(file)

# 读取PKL文件
with open(r"D:\temporal_download\epoch1_test_score.pkl", 'rb') as file:
    data = pickle.load(file)

# 打印数据内容
samples = []
cnt = 0
for key, value in data.items():
    if cnt % 32 == 0:
        samples.append(key)
    cnt += 1
file_path = "output.txt"

# 将列表内容写入文件
with open(file_path, "w") as file:
    for line in samples:
        file.write(line + "\n")
