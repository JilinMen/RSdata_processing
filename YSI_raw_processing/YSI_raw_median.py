# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:22:05 2024

@author: jmen
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import chardet
import numpy as np
from tqdm import tqdm

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def find_median_and_indices(arr):
    # delete NaN 值
    valid_data = arr[~np.isnan(arr)]
    
    # sort
    sorted_data = np.sort(valid_data)
    n = len(sorted_data)
    
    # compute median
    median_value = np.median(sorted_data)

    # if even, return the middle two
    if n % 2 == 0:
        lower_median_index = (n // 2) - 1
        upper_median_index = n // 2
        lower_value = sorted_data[lower_median_index]
        upper_value = sorted_data[upper_median_index]

        indices = np.argwhere(np.isin(arr, [lower_value, upper_value])).flatten().tolist()
    else:
        # if odd, return median and index
        median_value = sorted_data[n // 2]
        indices = np.argwhere(np.isclose(arr, median_value)).flatten().tolist()

    return median_value, indices
# input and output path
input_folder = r'C:\Users\jmen\Box\ERSL_FieldDatabase\M6_AlabamaRiver_Montgomery\2024June25_M6_M4\M6Montgomery\YSI_manual\Raw_YSI_manual'
target_path = r'C:\Users\jmen\Box\ERSL_FieldDatabase\M6_AlabamaRiver_Montgomery\2024June25_M6_M4\M6Montgomery\YSI_manual\median'
output_name = r'M6_EXO2_20240625'

out_path = os.path.join(target_path,output_name+'_median.csv')
Sonde = 2
if Sonde == 2:
    skiprows = 17
elif Sonde == 3:
    skiprows =14

data = []

# iterate all csv or xlsx files
for filename in tqdm(os.listdir(input_folder)):
    file_path = os.path.join(input_folder, filename)
    encoding = detect_encoding(file_path)
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path, skiprows=skiprows, index_col=False, encoding=encoding)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(file_path, skiprows=skiprows, index_col=False, engine=encoding)
    
    chlorophyll = df['Chl ug/L'] 
    if len(chlorophyll)%2 == 0:
        median_index = np.argsort(chlorophyll)[len(chlorophyll)//2]
        median_index = np.array(median_index[...,np.newaxis])
        line_median = df.iloc[median_index,:]
        data.append(line_median)
    else:
        index_median = np.argwhere(chlorophyll == np.nanmedian(chlorophyll))
        if len(index_median) == 1:
            line_median = df.iloc[index_median[0],:]
            data.append(line_median)
        elif len(index_median) > 1:
            line_median = df.iloc[index_median[0],:]
            data.append(line_median)
        else:
            print('No median data')
            
# merge all data to one DataFrame
combined_df = pd.concat(data, ignore_index=True)

combined_df.to_csv(out_path)
