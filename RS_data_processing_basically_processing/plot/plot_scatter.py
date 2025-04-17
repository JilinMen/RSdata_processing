# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:21:00 2024

@author: jmen
"""

def plot_scatter(x, y, tag, xlabel, ylabel, remove_negatives=False, size=50, color='blue', log_x=False, log_y=False, line11=True, degree=1,poly=None,
                 reg_line=True, confidence=False, ci=95, std=True,edgecolors='k', xylimit=True, alpha_values=None,figsize=(3.5, 3.5), legend=True):
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    from scipy.stats import linregress, t

    '''
    x: predict
    y: label
    tag: plot filename
    xlabel: label for the x-axis
    ylabel: label for the y-axis
    remove_negatives: Whether to remove negative values (default is False)
    ci: Confidence interval percentage (default is 95)
    '''
    # Remove negative values if specified
    if remove_negatives:
        valid_indices = (x > 0) & (y > 0)  # Ensure values > 0 for log scale
        x = x[valid_indices]
        y = y[valid_indices]

    if len(x) == 0 or len(y) == 0:
        print(f"No valid data to plot for tag {tag}. Skipping plot.")
        return

    min_val = np.amin([x, y])
    max_val = np.amax([x, y])
    
    # Perform linear regression
    if poly == None:
        if log_x:
            logx = np.log10(x)
        else:
            logx = x
        if log_y:
            logy = np.log10(y)
        else:
            logy = y
        coeffs = np.polyfit(logx, logy, degree) 
        poly = np.poly1d(coeffs)
        # slope, intercept, r_value, p_value, std_err = linregress(logx, logy)
        print("coeffs:",coeffs)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)  # Increased figure size for better visibility

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

    if xylimit == True:
        ax.set_xlim(0.9 * min_val, 1.1 * max_val)
        ax.set_ylim(0.9 * min_val, 1.1 * max_val)
        ax.set_xticks(np.linspace(0.9 * min_val, 1.1 * max_val, 5))
        ax.set_yticks(np.linspace(0.9 * min_val, 1.1 * max_val, 5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    
    if log_x:
        plt.xscale('log')  # 设置 x 轴为对数坐标
    if log_y:
        plt.yscale('log')  # 设置 y 轴为对数坐标

    if line11:
        x0 = np.array([0.8 * min_val, 1.2 * max_val])
        y0 = np.array([0.8 * min_val, 1.2 * max_val])
        ax.plot(x0, y0, color='black', linewidth=1, label='1:1')
    
    # Add grid lines for major ticks only
    ax.grid(which='major', linestyle='--', linewidth=0.5, color='gray')

    # Set individual transparency for each point
    if alpha_values is None:
        alpha_values = np.full_like(x, 0.7, dtype=np.float64)  # Default global transparency
    else:
        alpha_values = np.clip(alpha_values, 0, 1)  # Ensure alpha values are within [0, 1]

    # Plot scatter data
    ax.scatter(x, y, s=size, color=color, alpha=alpha_values, edgecolors=edgecolors, linewidth=0.5)  # Enhanced scatter plot
    
    if reg_line == True:
        # Plot regression line
        x_range = np.linspace(
            np.min(x),
            np.max(x),
            100
        )
        if log_x:
            x_range_log = np.log10(x_range)
            y_pred_log = poly(x_range_log)
            if log_y:
                y_pred = 10**y_pred_log
            else:
                y_pred = y_pred_log
        elif log_y:
            y_pred_log = poly(x_range)
            y_pred = 10 ** y_pred_log
        else:
            y_pred = poly(logx)

        ax.plot(x_range, y_pred, color='k', linewidth=1.5, label='Fitting line')
    
    if confidence == True:
        # Compute confidence interval
        n = len(x)
        t_value = t.ppf((1 + ci / 100) / 2, df=n - 2)  # t-critical value for given CI
        if log:
            se_y_log = std_err * np.sqrt(1 / n + ((x_range_log - np.mean(np.log10(x)))**2) / np.sum((np.log10(x) - np.mean(np.log10(x)))**2))
            ci_upper_log = y_pred_log + t_value * se_y_log
            ci_lower_log = y_pred_log - t_value * se_y_log
            ci_upper = 10**ci_upper_log
            ci_lower = 10**ci_lower_log
        else:
            se_y = std_err * np.sqrt(1 / n + ((x_range - np.mean(x))**2) / np.sum((x - np.mean(x))**2))
            ci_upper = y_pred + t_value * se_y
            ci_lower = y_pred - t_value * se_y
    
        ax.fill_between(x_range, ci_lower, ci_upper, color='red', alpha=0.2, label=f'{ci}% Confidence Interval')
        
    if std:
        # Calculate standard deviation range
        residuals = np.log10(y) - (slope * np.log10(x) + intercept) if log else y - (slope * x + intercept)
        std_dev = np.std(residuals)
    
        if log:
            std_upper_log = y_pred_log + 1.5 * std_dev
            std_lower_log = y_pred_log - 1.5 * std_dev
            std_upper = 10**std_upper_log
            std_lower = 10**std_lower_log
        else:
            std_upper = y_pred + 1.5 * std_dev
            std_lower = y_pred - 1.5 * std_dev
    
        # Fill the standard deviation region
        ax.fill_between(x_range, std_lower, std_upper, color='darkorange', alpha=0.2, label='1.5 STD')
    
        # Calculate proportion of points within 1.5 std
        if log:
            within_std = (y >= 10**(np.log10(x) * slope + intercept - 1.5 * std_dev)) & \
                         (y <= 10**(np.log10(x) * slope + intercept + 1.5 * std_dev))
        else:
            within_std = (y >= (x * slope + intercept - 1.5 * std_dev)) & \
                         (y <= (x * slope + intercept + 1.5 * std_dev))
    
        proportion_within_std = np.sum(within_std) / len(x)
        print(f'Proportion of points within 1.5 standard deviations: {proportion_within_std:.2%}')

    # Set labels
    ax.set_xlabel(xlabel, fontdict=font2)
    ax.set_ylabel(ylabel, fontdict=font2)
    ax.minorticks_on()

    if legend:
        # Add legend
        ax.legend()

    plt.tight_layout()

    # Save figure
    figname = f'./Scatter_{tag}_600dpi.png'
    plt.savefig(figname, dpi=600)
    plt.close()
