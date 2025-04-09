# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:17:24 2024

@author: jmen
"""
import numpy as np

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def plot_violin(df, xlabel, ylabel, figname, col_name):
    import matplotlib.pyplot as plt
    
    # 获取指定列的数据
    data = [df[col].dropna().tolist() for col in col_name]
    
    # 绘制violin图
    #创建绘图
    fig = plt.figure(num=1, figsize=(4, 3.5))    #figsize单位为英寸
    ax = plt.subplot(111)
    
    plt.rcParams['font.sans-serif']=['Times New Roman']
    
    font2 = {'family': 'Times New Roman', 
         'color': 'black',
         'weight': 'bold', 
         'size': 13
        }
    
    ax.tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True,
                   direction='inout', labelsize=10, length=10,width=1, colors='black')
    ax.tick_params(axis='x', which='minor', bottom=True, top=False, labelbottom=True,
                   direction='in', labelsize=10, length=5,width=1, colors='black')
    ax.tick_params(axis='y', which='major', left=True, right=False,labelbottom=True,
                   direction='inout', labelsize=10, length=10,width=1, colors='black')
    ax.tick_params(axis='y', which='minor', left=True, right=False,labelbottom=True,
                   direction='in', labelsize=10, length=5,width=1, colors='black')
    
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    
    parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)
    
    # 设置对数刻度
    ax.set_yscale('log')
    
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # 找出最长的子列表长度
    max_len = max(len(lst) for lst in data)
    
    # 使用 NaN 填充每个子列表，使它们的长度一致
    padded_data = np.array([lst + [np.nan] * (max_len - len(lst)) for lst in data])

    quartile1, medians, quartile3 = np.percentile(padded_data, [25, 50, 75], axis=1)
    
    whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(padded_data, quartile1, quartile3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
    
    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # 添加标题和标签
    # ax.set_xlabel(xlabel, fontdict=font2)
    ax.set_ylabel(ylabel, fontdict=font2)
    
    ax.set_xticklabels(xlabel, rotation=45, fontdict=font2)
    
    # plt.ticklabel_format(axis='both',style='sci')   #sci文章的风格
    plt.tight_layout()#rect=[left,bottom,right,top]   #设置图框与图片边缘的距离
    
    # 保存图片
    plt.savefig(figname, dpi=600)
    
