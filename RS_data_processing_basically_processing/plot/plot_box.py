# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:15:21 2024

@author: jmen
"""
def plot_box(data, labels, tag, ylabels, colors):
    """
    绘制具有独立次 y 轴的箱线图
    
    参数:
    data: list of arrays, 每组数据的值
    labels: list of str, 每组数据的标签
    tag: str, 保存的图片文件名
    xlabel: str, x轴的标签
    ylabels: list of str, 每个次 y 轴的标签
    colors: list of str, 每个箱线图的颜色
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    
    if not data or len(data) != len(labels) or len(data) != len(ylabels) or len(data) != len(colors):
        print("数据、标签、y轴标签或颜色数量不匹配，或者数据为空。")
        return
    
    # 设置全局字体为 Times New Roman
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12  # 全局字体大小
    
    # 创建主图和第一个 y 轴
    fig, ax_main = plt.subplots(figsize=(4.5, 2.5))
    axes = [ax_main]  # 初始化轴列表，将主轴加入列表

    # 绘制第一个箱线图
    box = ax_main.boxplot(data[0], positions=[1], patch_artist=True, notch=True,
                          medianprops={'linestyle': '-', 'color': 'crimson'},
                          widths=0.5,
                          showmeans=False, showfliers=False)
    
    ax_main.set_ylabel(ylabels[0], fontdict={'size': 12, 'weight': 'bold'}, color=colors[0])
    for patch in box['boxes']:
        patch.set_facecolor(colors[0])
    ax_main.tick_params(axis='y', labelcolor=colors[0])
    ax_main.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 依次为每列数据添加次 y 轴
    for i in range(1, len(data)):
        ax_secondary = ax_main.twinx()  # 创建次 y 轴
        axes.append(ax_secondary)
        
        # 调整 y 轴的间距
        ax_secondary.spines["right"].set_position(("outward", 45 * (i - 1)))

        box = ax_secondary.boxplot(data[i], positions=[i + 1], patch_artist=True, notch=True,
                                   medianprops={'linestyle': '-', 'color': 'crimson'},
                                   widths=0.5,
                                   showmeans=False, showfliers=False)
        
        ax_secondary.set_ylabel(ylabels[i], fontdict={'size': 12, 'weight': 'bold'}, color=colors[i])
        for patch in box['boxes']:
            patch.set_facecolor(colors[i])
        ax_secondary.tick_params(axis='y', labelcolor=colors[i])
        ax_secondary.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # 设置 x 轴标签和刻度
    ax_main.set_xticks(range(1, len(labels) + 1))
    ax_main.set_xticklabels(labels)
    # ax_main.set_xlabel(xlabel, fontdict={'size': 12, 'weight': 'bold'})

    # 保存图像
    plt.tight_layout()
    plt.savefig(f'./{tag}_boxplot.png', dpi=300)
    plt.close()


