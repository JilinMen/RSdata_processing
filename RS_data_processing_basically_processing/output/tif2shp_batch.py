# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:18:53 2024

@author: jmen
"""
import cv2
import numpy as np
from osgeo import gdal, osr
import h5py as h5
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.ndimage import zoom
from glob import glob
import os

def resample_image_with_geolocation(
        image,
        latitude,
        longitude,
        original_resolution=30,
        target_resolution=300,
        interpolation_method='bicubic'):
    """
    将遥感影像从原始分辨率重采样到目标分辨率

    参数:
    - image: 原始图像数组
    - latitude: 纬度数组
    - longitude: 经度数组
    - original_resolution: 原始分辨率(米)
    - target_resolution: 目标分辨率(米)
    - interpolation_method: 插值方法
        可选: 'nearest', 'bilinear', 'bicubic', 'area', 'lanczos4'

    返回:
    重采样后的图像和对应的地理坐标
    """
    import scipy.ndimage as ndimage
    # 插值方法映射
    interpolation_map = {
        'nearest': {
            'scipy_order': 0,
            'cv2_flag': cv2.INTER_NEAREST
        },
        'bilinear': {
            'scipy_order': 1,
            'cv2_flag': cv2.INTER_LINEAR
        },
        'bicubic': {
            'scipy_order': 3,
            'cv2_flag': cv2.INTER_CUBIC
        },
        'area': {
            'scipy_order': 1,
            'cv2_flag': cv2.INTER_AREA
        },
        'lanczos4': {
            'scipy_order': 4,
            'cv2_flag': cv2.INTER_LANCZOS4
        }
    }

    # 验证插值方法
    if interpolation_method not in interpolation_map:
        raise ValueError(f"不支持的插值方法: {interpolation_method}. "
                         f"支持的方法包括: {list(interpolation_map.keys())}")

    # 计算缩放比例
    scale_factor = target_resolution / original_resolution

    # 选择插值参数
    interp_params = interpolation_map[interpolation_method]

    # 处理多波段和单波段图像
    if len(image.shape) == 3:
        # 多波段图像
        resampled_channels = []
        for channel in range(image.shape[0]):
            channel_data = ndimage.zoom(
                image[channel, :, :],
                1 / scale_factor,
                order=interp_params['scipy_order']
            )
            resampled_channels.append(channel_data)
        resampled_image = np.stack(resampled_channels, axis=0)
    else:
        # 单波段图像
        resampled_image = ndimage.zoom(
            image,
            1 / scale_factor,
            order=interp_params['scipy_order']
        )

    # 重采样经纬度
    resampled_latitude = ndimage.zoom(
        latitude,
        1 / scale_factor,
        order=interp_params['scipy_order']
    )
    resampled_longitude = ndimage.zoom(
        longitude,
        1 / scale_factor,
        order=interp_params['scipy_order']
    )

    return resampled_image, resampled_latitude, resampled_longitude

def write_bands(im_data, banddes=None):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    # 数据类型必须有，因为要计算需要多大内存空间
    driver = gdal.GetDriverByName("MEM")
    dataset = driver.Create("", im_width, im_height, im_bands, datatype)

    # 写入数组数据
    if im_bands == 1:
        # dataset.GetRasterBand(1).SetNoDataValue(65535)
        try:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入
        except:
            dataset.GetRasterBand(1).WriteArray(im_data[:,:,0])
    else:
        # if banddes==None:
        # banddes = ['Rrs_412', 'Rrs_443', 'Rrs_490', 'Rrs_520', 'Rrs_565', 'Rrs_670', 'chlor_a']
        for i in range(im_bands):
            try:
                # dataset.GetRasterBand(i + 1).SetNoDataValue(65535)
                RasterBand = dataset.GetRasterBand(i + 1)
                # RasterBand.SetDescription(banddes[i])
                RasterBand.WriteArray(im_data[:, :, i])
            except IndentationError:
                print('band:'+i)

    return dataset


def save_tif_UTM(image, output_path, lat, lon, bands, set_projection=True):
    """
    保存图像数据为UTM投影的GeoTIFF文件。

    参数:
    -----------
    image : numpy.ndarray
        2D 或 3D 图像数组 (bands, height, width)
    output_path : str
        GeoTIFF 文件保存路径
    lat : numpy.ndarray
        纬度值的二维数组
    lon : numpy.ndarray
        经度值的二维数组
    set_projection : bool
        是否设置UTM投影，默认为 True
    """
    from pyproj import Transformer
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import rasterio

    # 将波段数据写入临时内存文件
    image: gdal.Dataset = write_bands(image)

    # 控制点列表, 设置7*7个控制点
    gcps = []
    x_arr = np.linspace(0, lon.shape[1] - 1, num=30, endpoint=True, dtype='int')
    y_arr = np.linspace(0, lon.shape[0] - 1, num=30, endpoint=True, dtype='int')
    for x in x_arr:
        for y in y_arr:
            if abs(lon[y, x]) > 180 or abs(lat[y, x]) > 90:
                continue
            gcps.append(gdal.GCP(np.float64(lon[y, x]), np.float64(lat[y, x]),
                                 0,
                                 np.float64(x), np.float64(y)))

    # 设置空间参考
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.MorphToESRI()

    # 给数据及设置控制点及空间参考
    image.SetGCPs(gcps, sr.ExportToWkt())

    dst = gdal.Warp(output_path, image, format='GTiff', tps=True, xRes=0.01, yRes=0.01, dstNodata=np.nan,
                    resampleAlg=gdal.GRA_NearestNeighbour)  # dstNodata=65535

    for i, bandname in enumerate(bands):
        band = dst.GetRasterBand(i + 1)
        band.SetMetadata({'bandname': bandname})
        band.SetDescription(bandname)

    image: None
    return output_path

if __name__ == '__main__':
    path = r'D:\10-Landsat_Aquatic_Reflectance\Timeseries\VIIRS\requested_files_1'
    out_path = r'D:\10-Landsat_Aquatic_Reflectance\Timeseries\VIIRS\tif'
    files = glob(os.path.join(path,'*L2.OC.x.nc'))
    for file in files:
        '''load data'''
        output_path = os.path.join(out_path, os.path.basename(file).split('.')[0] +os.path.basename(file).split('.')[1]+ '.tif')
        ds = h5.File(file, 'r')
        sensor = 'VIIRS'  # Choose the sensor type
        AC = 'seadas'  # acolite, seadas, ocsmart

        resample_option = False

        # Select bands based on sensor type
        if sensor == 'L8':
            bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_655']
        elif sensor == 'L9':
            bands = ['Rrs_443', 'Rrs_482', 'Rrs_561', 'Rrs_654', 'Rrs_865']
        elif sensor == 'OLCI':
            bands = ['Rrs_443', 'Rrs_490', 'Rrs_560', 'Rrs_665']
        elif sensor == 'VIIRS':
            bands = ['Rrs_445', 'Rrs_489', 'Rrs_556', 'Rrs_667']
        elif sensor == 'MODISA':
            bands = ['Rrs_443', 'Rrs_488', 'Rrs_555', 'Rrs_667']
        else:
            raise ValueError("Unrecognized sensor type")

        if AC == 'seadas':
            # Create an empty 3D array to store all bands
            image = np.array([np.array(np.array(
                ds['geophysical_data/' + band] * ds['geophysical_data/' + band].attrs['scale_factor'] +
                ds['geophysical_data/' + band].attrs['add_offset'])) for band in bands])

            image[image<=0] = np.nan
            image = np.transpose(image, (1,2,0))
            # image[image==-0.015534002] = 0
            # Get latitude and longitude data
            lat, lon = np.array(ds['navigation_data/latitude']), np.array(ds['navigation_data/longitude'])

            # image = image[:,::-1,::-1]
        elif AC == 'acolite':
            image = np.array(ds['Rrs_561'])

            # Get latitude and longitude data
            lat, lon = np.array(ds['lat']), np.array(ds['lon'])

        elif AC == 'ocsmart':
            image = np.array(ds['Rrs/Rrs_561nm'])

            # Get latitude and longitude data
            lat, lon = np.array(ds['Latitude']), np.array(ds['Longitude'])

        # save_tif_gdal(image, output_path, lat, lon)

        # image = image[:,::-1,::-1]  # 上下翻转影像数据

        '''resampling'''
        if resample_option == True:
            original_resolution = 30
            target_resolution = 750
            image_resample, lat_resample, lon_resample = resample_image_with_geolocation(image, lat, lon,
                                                                                         original_resolution=original_resolution,
                                                                                         target_resolution=target_resolution,
                                                                                         interpolation_method='bicubic')
            # # Save as GeoTIFF file
            save_tif_UTM(image_resample, output_path, lat_resample, lon_resample)
        else:
            save_tif_UTM(image, output_path, lat, lon, bands)