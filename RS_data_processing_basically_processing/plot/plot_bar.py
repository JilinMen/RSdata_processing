# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:41:17 2024

@author: jmen
"""

def plot_bar(data, tag, xlabel, ylabel, remove_negatives=False):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    '''
    data: array-like, the data to plot (3x30 array)
    tag: plot filename
    xlabel: label for the x-axis
    ylabel: label for the y-axis
    remove_negatives: Whether to remove negative values (default is False)
    '''

    # Remove negative values if specified
    if remove_negatives:
        data = np.where(data < 0, 0, data)

    if data.size == 0:
        print(f"No valid data to plot for tag {tag}. Skipping plot.")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size for better visibility

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
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    
    # Add grid lines for major ticks only
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray')

    # Water types and methods
    water_types = np.arange(30)+1
    methods = ['SeaDAS', 'Acolite', 'OC-SMART']

    # Set bar width
    bar_width = 0.25

    # Set positions of bars on X axis
    r1 = np.arange(len(water_types))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars
    ax.bar(r1, data.iloc[:,0], color='b', width=bar_width, edgecolor='grey', label=methods[0])
    ax.bar(r2, data.iloc[:,1], color='darkorange', width=bar_width, edgecolor='grey', label=methods[1])
    ax.bar(r3, data.iloc[:,2], color='green', width=bar_width, edgecolor='grey', label=methods[2])

    # Add labels
    ax.set_xlabel(xlabel, fontdict=font2)
    ax.set_ylabel(ylabel, fontdict=font2)
    ax.set_xticks([r + bar_width for r in range(len(water_types))])
    ax.set_xticklabels(water_types, ha='right')

    # Add legend
    ax.legend()

    plt.tight_layout()

    # Save figure
    figname = f'./{tag}_600dpi.png'
    plt.savefig(figname, dpi=600)
    plt.close()


def plot_circular_bar(data, tag, xlabel, ylabel, remove_negatives=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import pandas as pd
    """
    Create a circular bar plot with enhanced customization and NaN handling

    Parameters:
    -----------
    data : pandas.DataFrame
        The data to plot (rows represent categories, columns represent methods)
    tag : str
        Plot filename and title
    xlabel : str
        Label for the x-axis
    ylabel : str
        Label for the y-axis
    remove_negatives : bool, optional
        Whether to remove negative values (default is False)
    """
    # Validate input data
    if data is None or data.empty:
        print(f"No data to plot for tag {tag}. Skipping plot.")
        return

    # Replace NaN values with 0 or remove them
    data_cleaned = data.fillna(0)  # or data.dropna() if you want to remove rows with NaN

    # Remove negative values if specified
    if remove_negatives:
        data_cleaned = data_cleaned.clip(lower=0)

    # Check if data still has values after cleaning
    if data_cleaned.empty or data_cleaned.isnull().all().all():
        print(f"No valid data to plot for tag {tag}. Skipping plot.")
        return

    # Create figure and polar axes
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)

    # Configure fonts and text rendering
    plt.rcParams.update({
        'axes.unicode_minus': False,
        'font.sans-serif': ['Times New Roman'],
        'font.family': 'Times New Roman'
    })

    # Prepare plot parameters
    water_types = np.arange(1, len(data_cleaned) + 1)
    methods = data_cleaned.columns.tolist()
    N = len(water_types)

    # Color palette (more vibrant and distinct)
    colors = ['#1E90FF', '#FF6347', '#3CB371']  # Dodger Blue, Tomato, Medium Sea Green

    # Compute angles and width
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    width = 2 * np.pi / N * 0.9  # Slightly reduced spacing between bars

    # Plot bars for each method
    for i, method in enumerate(methods):
        # Compute offset to cluster bars
        offset = i * (width / len(methods))

        # Plot bars with improved styling
        bars = ax.bar(
            theta + offset,
            data_cleaned[method],
            width=width / len(methods),
            bottom=0.0,
            color=colors[i % len(colors)],
            alpha=0.7,
            label=method,
            edgecolor='white',
            linewidth=1
        )

    # Customize polar plot orientation
    ax.set_theta_offset(np.pi / 2)  # Start from top
    ax.set_theta_direction(-1)  # Clockwise rotation

    # Set x-ticks and labels
    ax.set_xticks(theta + np.pi / N)
    ax.set_xticklabels(water_types)

    # Styling title and labels
    plt.title(
        tag,
        fontdict={
            'fontname': 'Times New Roman',
            'color': '#006400',  # Hexadecimal for dark green
            'fontweight': 'bold',
            'fontsize': 15
        }
    )

    # Labels and legend
    ax.set_xlabel(xlabel, fontdict={'fontsize': 12})
    ax.set_ylabel(ylabel, fontdict={'fontsize': 12})

    # Improved legend placement
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.2, -0.1),
        title='Methods',
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # Save high-resolution figure
    plt.tight_layout()
    figname = f'./{tag}_circular_plot_600dpi.png'
    plt.savefig(
        figname,
        dpi=600,
        bbox_inches='tight',
        transparent=False
    )
    plt.close()
