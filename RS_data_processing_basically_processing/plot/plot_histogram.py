# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:38:00 2024

@author: jmen
"""
def plot_histogram(data, tag, xlabel, ylabel, remove_negatives=False):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    '''
    data: array-like, the data to plot
    tag: plot filename
    xlabel: label for the x-axis
    ylabel: label for the y-axis
    remove_negatives: Whether to remove negative values (default is False)
    '''

    # Remove negative values if specified
    if remove_negatives:
        data = data[data >= 0]

    if len(data) == 0:
        print(f"No valid data to plot for tag {tag}. Skipping plot.")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Increased figure size for better visibility

    # Set font properties
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    font1 = {'family': 'Times New Roman', 'color': 'green', 'weight': 'bold', 'size': 13}
    font2 = {'family': 'Times New Roman', 'color': 'black', 'weight': 'bold', 'size': 13}

    # Set axis styles
    ax.tick_params(axis='x', which='major', direction='inout', length=10, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='y', which='major', direction='inout', length=10, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='x', which='minor', direction='in', length=5, width=1, colors='black', labelsize=10)
    ax.tick_params(axis='y', which='minor', direction='in', length=5, width=1, colors='black', labelsize=10)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    
    # Add grid lines for major ticks only
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray')

    # Plot data
    ax.hist(data, bins=30, color='blue', alpha=0.7, edgecolor='black')  # Histogram plot

    # Set labels
    ax.set_xlabel(xlabel, fontdict=font2)
    ax.set_ylabel(ylabel, fontdict=font2)
    ax.minorticks_on()
    plt.tight_layout()

    # Save figure
    figname = f'./{tag}_600dpi.png'
    plt.savefig(figname, dpi=600)
    plt.close()
