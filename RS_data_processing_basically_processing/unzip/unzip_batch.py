# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:27:19 2024

@author: jmen
"""
import os
import tarfile
from tqdm import tqdm  # 用于显示进度条

# 定义解压函数，显示整体进度和单个文件的解压进度
def extract_all_tar_gz(directory):
    # 获取所有 .tar.gz 文件的列表
    tar_files = [f for f in os.listdir(directory) if f.endswith(".tar.gz")]
    total_files = len(tar_files)

    # 显示整体任务进度条
    with tqdm(total=total_files, desc="Overall Progress", unit="file") as overall_progress:
        # 遍历每个 .tar.gz 文件
        for filename in tar_files:
            # 构建完整文件路径
            file_path = os.path.join(directory, filename)
            
            # 创建解压目标文件夹，命名与文件名一致
            extract_dir = os.path.join(directory, filename.replace(".tar.gz", ""))
            os.makedirs(extract_dir, exist_ok=True)

            # 打开 .tar.gz 文件
            with tarfile.open(file_path, "r:gz") as tar:
                members = tar.getmembers()  # 获取所有成员（文件）
                total_members = len(members)  # 获取成员总数

                # 显示每个 .tar.gz 文件的解压进度条
                with tqdm(total=total_members, desc=f"Extracting {filename}", unit="file", leave=False) as file_progress:
                    for member in members:
                        tar.extract(member, path=extract_dir)
                        file_progress.update(1)  # 更新单个文件解压进度条
                
            # 每解压完一个 .tar.gz 文件，更新整体进度
            overall_progress.update(1)

# 示例：设置要解压的文件夹路径
directory = r"D:\10-Landsat_Aquatic_Reflectance\Timeseries\AR"  # 目录路径
extract_all_tar_gz(directory)

