# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 17:39:03 2022

@author: rdi420
"""

#from mpl_toolkits.basemap import Basemap
from mpl_toolkits import basemap
#https://matplotlib.org/basemap/api/basemap_api.html
import netCDF4 as nc
#from netCDF4 import nc
import numpy as np
#import pdb # package for debugging
from matplotlib import pyplot as plt

# precipitation 
#path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg.nc"
# soil moisture
path1 = "C:/MultiHazard/Data/soil_moisture/dataset-satellite-soil-moisture-1c23027a-eac9-42c7-bd11-73662486cf06/C3S-SOILMOISTURE-L3S-SSMV-PASSIVE-DEKADAL-19781101000000-TCDR-v202012.0.0.nc"
ds = nc.Dataset(path1)
ds.set_auto_mask(False)
#filename = '/Users/r/global_aug4.region.nc'
#pdb.set_trace()
#with Dataset(path1, mode='r') as fh:
lons = ds['lon'][:]
lats = ds['lat'][:]
# dset.variables.keys() 
content = ds['sm'][0].squeeze()
#lons_sub, lats_sub = np.meshgrid(lons[::4], lats[::4])


lats_fine = np.flipud(np.arange(lats[0], lats[-1], -0.0025)) # 0.25 degree fine grid
#lats_fine = np.arange(lats[0], lats[-1], 0.25) # 0.25 degree fine grid
lons_fine = np.arange(lons[0], lons[-1], 0.0025)
lons_sub, lats_sub = np.meshgrid(lons_fine, lats_fine)

# order 
# 0 for nearest-neighbor interpolation, 1 for bilinear interpolation, 3 for cublic spline (default 1)
coarse = basemap.interp(content, lons, np.flipud(lats), lons_sub, lats_sub, order=0)

# figure

plt.imshow(coarse, interpolation='nearest')
plt.show()