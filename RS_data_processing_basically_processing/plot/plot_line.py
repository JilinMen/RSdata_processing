# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:47:48 2024

@author: jmen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_lines(df, output, x_labels=None, xlabel="X", ylabel="Y",  
                       figsize=(3.5, 3.5), line_width=1, with_markers=False, 
                       with_legend=True):
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 设置颜色循环
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))  # 使用tab20配色方案
    
    font = {'family': 'Times New Roman', 
         'color': 'black',
         'weight': 'bold', 
         'size': 13
        }
    
    # 绘制每一行的折线
    for idx, (index, row) in enumerate(df.iterrows()):
        data = row.values.astype(float)  
        x = np.arange(len(data))  # 使用numpy数组替代range
        
        if with_markers:
            plt.plot(x, data, linewidth=line_width, marker='o', color=colors[idx])
        else:
            plt.plot(x, data, linewidth=line_width, color=colors[idx])
    
    # 设置图表属性
    # plt.title(title, fontsize=14, pad=15)
    plt.xlabel(xlabel, fontdict=font)
    plt.ylabel(ylabel, fontdict=font)
    
    # Set axis styles
    ax.tick_params(axis='x', which='major', direction='inout', length=10, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='y', which='major', direction='inout', length=10, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='x', which='minor', direction='in', length=5, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='y', which='minor', direction='in', length=5, width=1, colors='black', labelsize=10)
    
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    
    # Add grid lines for major ticks only
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray')
    
    # 获取数据点数量（用于设置x轴刻度）
    num_points = len(df.iloc[0])
    
    # 设置x轴刻度标签
    if x_labels is not None:
        if len(x_labels) != num_points:
            print(f"Warning: x_labels length ({len(x_labels)}) doesn't match data points ({num_points})")
            x_labels = x_labels[:num_points] if len(x_labels) > num_points else x_labels + ['']*(num_points-len(x_labels))
        plt.xticks(range(num_points), x_labels, weight='bold',fontsize=12)
    else:
        plt.xticks(weight='bold',fontsize=12)
        
    plt.yticks(weight='bold',fontsize=12)
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    # 调整布局，确保图例不被裁切
    plt.tight_layout()
    
    plt.savefig(output, dpi=600)

    plt.close()

# 使用示例
if __name__ == "__main__":
    path = r'D:\10-Landsat_Aquatic_Reflectance\GLORIA_matchups\2-match_results\GLORIA_multispectral_validation.xlsx'
    
    heads = ['B443','B483','B561','B655','B865']
    # 创建示例数据
    df = pd.read_excel(path,usecols=heads,sheet_name='AR')
    
    output = r'D:\10-Landsat_Aquatic_Reflectance\Rrs_spectral_shape.png'
    xlabels = ['B1','B2','B3','B4','B5']
    # 绘制图表
    plt = plot_multiple_lines(df*np.pi, output,
                              xlabels,
                            xlabel='Bands',
                            ylabel='In-situ AR',
                            figsize=(3.5, 2.5))
    

