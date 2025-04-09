# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:11:09 2024

@author: jmen
"""
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pyproj
from rasterio.transform import from_origin

def check_encoding(path_df):
    import chardet
    with open(path_df, "rb") as f:
        raw_data = f.read(10000)  # 只读取部分数据，提高检测速度
        result = chardet.detect(raw_data)
        detected_encoding = result["encoding"]
        print(f"Detected encoding: {detected_encoding}")
    return detected_encoding

def geomatching(path_df, path_image, path_out, invalid=0, head_name=None):
    detected_encoding = check_encoding(path_df)
    print("Encoding: ",detected_encoding)
    # Load in-situ measurement data
    df = pd.read_csv(path_df, encoding=detected_encoding)

    # Create an empty list to store matching results
    results = []

    # Define coordinate systems
    utm_crs = pyproj.CRS.from_epsg(32616)  # WGS 84 / UTM zone 16N
    wgs84_crs = pyproj.CRS.from_epsg(4326)  # WGS 84
    # Create a coordinate transformer
    transformer = pyproj.Transformer.from_crs(wgs84_crs, utm_crs)

    # Load satellite image data
    with rasterio.open(path_image) as src:
        # Get the original coordinate system of the image
        src_crs = src.crs
        print("src_crs: ",src_crs)
        # If the image's coordinate system is not UTM, reproject it
        if src_crs != utm_crs:
            print(f"Reprojecting image from {src_crs} to {utm_crs}...")
            # Compute the reprojected image extent and resolution
            transform, width, height = rasterio.warp.calculate_default_transform(
                src_crs, utm_crs, src.width, src.height, *src.bounds
            )
            # Create an in-memory file to store the reprojected image
            reprojected_image = np.zeros((src.count, height, width), dtype=src.dtypes[0])

            # Reproject the image
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=reprojected_image[i - 1],
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest
                )

            # Update the image's affine transformation parameters and coordinate system
            transform = transform
            image_crs = utm_crs
        else:
            # If the image is already in UTM coordinates, use the original data
            reprojected_image = src.read()
            transform = src.transform
            image_crs = src.crs
            width, height = src.width, src.height

        # Get the number of image bands
        num_bands = reprojected_image.shape[0]

        # Iterate through each in-situ measurement point
        for index, row in df.iterrows():
            # Get the latitude and longitude of the in-situ point
            lon, lat = row['Lon'], row['Lat']

            print('Lon: %f; lat: %f' % (lon, lat))

            # Convert latitude and longitude to UTM coordinates
            utm_easting, utm_northing = transformer.transform(lat, lon)  # Note: latitude first
            print(f"UTM Easting: {utm_easting}; UTM Northing: {utm_northing}")

            # Convert UTM coordinates to image pixel coordinates
            pixel_row, pixel_col = ~transform * (utm_easting, utm_northing)

            # Check for NaN values
            if np.isnan(pixel_row) or np.isnan(pixel_col):
                print(f"--Error: Invalid coordinates for point {index} (Lon: {lon}, Lat: {lat}). Skipping this point.")
                continue  # Skip this point

            pixel_row, pixel_col = int(pixel_row), int(pixel_col)  # Convert to integers
            print(f"x: {pixel_row}; y: {pixel_col}")

            # Check if the in-situ point is within the image extent
            if 0 <= pixel_row < height and 0 <= pixel_col < width:
                print('--This point is in the image range!')

                # Read a 3x3 pixel box centered at the matched pixel
                window = ((pixel_row - 1, pixel_row + 2), (pixel_col - 1, pixel_col + 2))

                try:
                    data_B3 = reprojected_image[2, window[0][0]:window[0][1], window[1][0]:window[1][1]]

                    data_B3 = np.where(data_B3 == invalid, np.nan, data_B3)
                    # Compute the number of valid pixels and coefficient of variation
                    valid_pixels = data_B3[~np.isnan(data_B3)]
                    if valid_pixels.size > data_B3.size * 0.5:
                        print('---Valid pixels > 0.5!')
                        cv = np.std(valid_pixels) / np.mean(valid_pixels)
                        if cv < 0.15:
                            print('----CV < 0.15!')
                            data = reprojected_image[:, window[0][0]:window[0][1], window[1][0]:window[1][1]]
                            data = np.where(data == invalid, np.nan, data)
                            # Compute the mean of all valid values within the box
                            mean_value = np.nanmean(data, axis=(1, 2))
                            # Append the results to the list
                            results.append(np.concatenate((row.values, mean_value)))
                except Exception as e:
                    print(f"Error reading window data: {e}")
            else:
                print('--This point is NOT in the image range!')

    # Save to CSV
    if head_name is None:
        df_result = pd.DataFrame(results)
    else:
        df_result = pd.DataFrame(results, columns=head_name)
    df_result.to_csv(path_out, index=False)

    print('Save CSV to %s' % (path_out))

if __name__ == '__main__':
    # Input CSV file
    path_df = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeTuscaloosa\LakeTuscaloosa_Mar_1_2025\YSI_manual\YSI_matching\raw\250301-145615-laketuscaloosa_area1-1.csv'
    # Input satellite/drone .tif image
    path_image = r'I:\headwall_20250301_LT.tif'
    # Output path
    path_out = r'C:\Users\jmen\Box\ERSL_FieldDatabase\LakeTuscaloosa\LakeTuscaloosa_Mar_1_2025\YSI_manual\YSI_matching\Headwall\Headwall_YSI_F1_20250301_area1-1.csv'
    # Column headers
    head_name = ['FID', 'Id', 'NTU', 'Chl', 'pH', 'fDOM', 'DO', 'NitraLED', 'SpCond', 'Lat', 'Lon',
                 'AR_B1', 'AR_B2', 'AR_B3', 'AR_B4', 'AR_B5', 'AR_B6', 'AR_B7', 'AR_B8', 'AR_B9', 'AR_B10']
        
    geomatching(path_df, path_image, path_out)
