# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:31:59 2024

@author: jmen
"""
# import packages 
import xarray as xr 
import rioxarray as rio 

nc_file = xr.open_dataset(r'H:\Satellite_processing_ERSL\L2\S2B_MSIL2_20241007T161049_N0511_R140_T17SKT_20241007T195344.SAFE\S2B_MSI_2024_10_07_16_24_05_T17SKT_L2R.nc')

bT = nc_file['rhot_943']
lat = nc_file['lat']
lon = nc_file['lon']

bT = bT.rio.set_spatial_dims(x_dim=lon, y_dim=lat)
bT.rio.crs

# Define the CRS projection
bT.rio.write_crs("epsg:4326", inplace=True)

bT.rio.to_raster(r"medsea_bottomT_raster.tif")