# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:37:00 2024

@author: jmen
"""
import numpy as np
from osgeo import gdal, osr
import h5py as h5
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import os
import glob
import re

def save_as_geotif(lat, lon, image, output_path):
    lat_min, lat_max = np.nanmin(lat), np.nanmax(lat)
    lon_min, lon_max = np.nanmin(lon), np.nanmax(lon)
    
    if image.ndim == 2:
        image = image[np.newaxis, :, :]  # Add a band dimension
    elif image.ndim != 3:
        raise ValueError("Input image array must be 2D or 3D")
        
    channels, rows, cols = image.shape
    
    # Calculate pixel size
    pixel_width = (lon_max - lon_min) / (cols - 1)
    pixel_height = (lat_max - lat_min) / (rows - 1)
    
    # Create the transformation
    transform = from_origin(lon_min - pixel_width / 2, lat_max + pixel_height / 2, pixel_width, pixel_height)
    
    # Define the CRS (WGS84)
    crs = CRS.from_epsg(4326)
    
    # Create the GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=channels,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        # Write the data
        for i in range(channels):
            dst.write(image[i, :, :], i+1)  # rasterio band index starts from 1

    print(f"GeoTIFF saved to {output_path}")

def save_as_geotif_NEON(image,bands,transform, crs, output_path):
    rows, cols,channels = image.shape
    # Create the GeoTIFF file
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=rows,
            width=cols,
            count=len(bands),
            dtype=image.dtype,
            crs=crs,
            transform=transform,
    ) as dst:
        # Write the data
        for i in range(len(bands)):
            dst.write(image[:, :,bands[i]], i + 1)  # rasterio band index starts from 1

    print(f"GeoTIFF saved to {output_path}")

if __name__ == '__main__':
    path = r'D:\NEON_hyperspectra\NEON_refl-surf-bidir-ortho-mosaic\NEON_refl-surf-bidir-ortho-mosaic\NEON.D08.DELA.DP3.30006.002.2024-05.basic.20250221T162647Z.PROVISIONAL'
    out_path = r'D:\NEON_hyperspectra\NEON_refl-surf-bidir-ortho-mosaic\NEON_refl-surf-bidir-ortho-mosaic\Transfer_to_tif'
    files = glob.glob(os.path.join(path,'*.h5'))
    for file in files:
        output_path = os.path.join(out_path,os.path.basename(file).split('.')[0] + '.tif')

        ds = h5.File(file, 'r')

        sensor = 'NEON'  # Choose the sensor type

        # Select bands based on sensor type
        if sensor == 'L8':
            bands = ['Rrs_443', 'Rrs_483', 'Rrs_561', 'Rrs_655', 'Rrs_865']
        elif sensor == 'L9':
            bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_654', 'Rrs_865']

        if sensor == "L8" or sensor == "L9":
            # Create an empty 3D array to store all bands
            image = np.array([np.array(ds[band]) for band in bands])

            # Get latitude and longitude data
            lat, lon = np.array(ds['lat']), np.array(ds['lon'])

            # Save as GeoTIFF file
            save_as_geotif(lat, lon, image, output_path)

        if sensor == 'NEON':
            bands = [20,35,57]
            date =np.array(ds['DELA/Reflectance/Metadata/Ancillary_Imagery/Acquisition_Date'])
            unique_date = np.unique(date)
            if 20240521 in unique_date and 20240528 in unique_date:
                date_overpass = "both"
            elif 20240521 in unique_date:
                date_overpass = "20240521"
            elif 20240528 in unique_date:
                date_overpass = "20240528"
            elif -9999 in unique_date:
                date_overpass = '9999'
            else:
                date_overpass = "other"
                print('Other day included: ', unique_date)
            output_path = os.path.join(out_path, os.path.basename(file).split('.')[0] +date_overpass+ '.tif')

            image = np.array(ds['DELA/Reflectance/Reflectance_Data'])
            transform_code = np.array(ds['DELA/Reflectance/Metadata/Coordinate_System/Map_Info']).item().decode()
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", transform_code)
            # 解析六参数
            a = float(numbers[4])  # X 方向像元大小
            b = 0.0  # 旋转项（通常为 0）
            d = 0.0  # 旋转项（通常为 0）
            e = float(numbers[6])  # Y 方向像元大小（通常为负值）
            x = float(numbers[2])  # 左上角 X 坐标（UTM 东坐标）
            y = float(numbers[3])  # 左上角 Y 坐标（UTM 北坐标）
            transform = from_origin(x, y,a,e)
            crs = CRS.from_epsg(np.array(ds['DELA/Reflectance/Metadata/Coordinate_System/EPSG Code']).item().decode())
            save_as_geotif_NEON(image,bands,transform, crs, output_path)


