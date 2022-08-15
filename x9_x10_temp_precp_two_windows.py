# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:24:20 2022

@author: rdi420
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
#%
ds = nc.Dataset("C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg.nc")
ds.set_auto_mask(False)
#%%
print(ds['pr'][0,90:98, 205:208])
print(ds['pr'][1,90:98, 205:208])
print(ds['pr'][2,90:98, 205:208])

#%%

plt.plot(u[:, exper_index, lat_index, lon_index], time)
#https://unidata.github.io/netcdf4-python/

#%% input: date and coordinates pairs, return: tensors of images
from datetime import datetime, timedelta
#%%
import numpy as np
def pad(list_of_list, pad_num):
    # pad_num: the number of column you want to add on one side.
             # We have 2 sides to pad in total.
    for idx, row in enumerate(list_of_list):
        last_bit = row[-pad_num:]
        last_bit.reverse()
        last_bit.extend(row)
        last_bit.extend(row[:pad_num])
        list_of_list[idx] = last_bit
        
        
    return np.array(list_of_list)
    

#%%
def get_tensor(var, row_of_input, length):
    #%%
    var_code  = 'pr'
    if var != 'precipitation':
        var_code = []
    # get the file of the right year
    time1 = datetime.strptime(row_of_input[1], '%m/%d/%y')
    time2 = datetime.strptime(row_of_input[2], '%m/%d/%y')
    year1 = time1.year
    year2 = time2.year
    path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1)
    ds1 = nc.Dataset(path1)
    tensor = []
    coord = row_of_input[3][1:-1].split(',')
    # convert the center coordinates to integer
    lat1 = float(coord[0])
    lon1 = float(coord[1])
    lat2 = float(coord[2])
    lon2 = float(coord[3])
    dist_ny = time1 - datetime(year1, 1,1)
    lat1b = int(lat1 + 0.5)-1
    lon1b = int(lon1 + 0.5)-1
#%%
    for day in range(length):
        delta = timedelta(days = day)
        target_day = time1 - delta
        # distance to new year day (January 1st) = index
        dist = (dist_ny - delta).days
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1-1)
            ds = nc.Dataset(path1)
        img = ds[var_code][dist].tolist()
        img = pad(img, 2)# pad 2 degree in longituabs(n: Suppode
        img = img[abs(lat1b-90)-2: abs(lat1b- 90) + 3, abs(lon1b + 180)-2:abs(lon1b+ 180)+3]
        tensor.append(img)

#%%
    # Now we start to extract info for event 2    
    if year2 != year1:
        dist_ny = (time2 - datetime(year1, 1,1)).years
    lat2b = int(lat2 + 0.5)-1
    lon2b = int(lon2 + 0.5)-1
    for day in range(length):
        delta = timedelta(days = day)
        target_day = time2 - delta
        # distance to new year day (January 1st) = index
        dist = (dist_ny - delta).days
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path2 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1-1)
            ds = nc.Dataset(path1)
        img = ds[var_code][dist].tolist()
        img = pad(img, 2)# pad 2 degree in longituabs(n: Suppode
        img = img[abs(lat1b-90)-2: abs(lat1b- 90) + 3, abs(lon1b + 180)-2:abs(lon1b+ 180)+3]
        tensor.append(img)
        
        
#%% read in date-coor file
date_coor = pd.read_csv('C:/MultiHazard/Data/emdat/trial/_30day_100km_date_coord.csv')