# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 17:47:29 2022

@author: rdi420

# Count missing data
# and also output complete samples

"""

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
    
def change_into_chanel_first(matrix):
    chanel_total = len(matrix)
    row_total = len(matrix[0])
    col_total = len(matrix[0])
    result = np.zeros([row_total, col_total, chanel_total])
 #   matrix = matrix.tolist()
    for chanel in range(chanel_total):
        for row in range(row_total):
            for col in range(col_total):
                result[row][col][chanel] = matrix[chanel][row][col]
            # row_value = matrix[chanel][row]
            # if row_value != None:
            #     for col in range(col_total):
            #         result[row][col][chanel] = row_value[col]
            # else:
            #     for col in range(col_total):
            #         result[row][col][chanel] = None
    return(result)
    
#%%
def get_tensor(var, row_of_input, length):
    pad_size = 5
    var_code  = 'pr'
    if var != 'precipitation':
        var_code = []
    # get the file of the right year
    time1 = datetime.strptime(row_of_input[1], '%m/%d/%Y')
    time2 = datetime.strptime(row_of_input[2], '%m/%d/%Y')
    year1 = time1.year
    year2 = time2.year

    path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1)
    ds = nc.Dataset(path1)
    tensor = []
    coord = row_of_input[3][1:-1].split(',')
    # convert the center coordinates to integer
    lat1 = float(coord[0])
    lon1 = float(coord[1])
    lat2 = float(coord[2])
    lon2 = float(coord[3])
    lat_mid = (lat1 + lat2)/2
    lon_mid = (lon1 + lon2)/2
    dist_ny = time1 - datetime(year1, 1,1)
    lat_midb = int(lat1 + 0.5)-1
    lon_midb = int(lon1 + 0.5)-1
    
    missing = 0

# days before event 1
    for day in range(length):
        delta = timedelta(days = day)
        target_day = time1 - delta
        # distance to new year day (January 1st) = index
        dist = (dist_ny - delta).days
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1-1)
            ds = nc.Dataset(path1)
        img = ds[var_code][dist].tolist()
        img.reverse()
        img = pad(img, pad_size)# pad 2 degree in longitude
        img = img[abs(lat_midb-90)-pad_size : abs(lat_midb- 90) + pad_size + 1, abs(lon_midb + 180)-pad_size +pad_size:abs(lon_midb+ 180)+2*pad_size+1]
        if None in img:
            missing = None
        tensor.append(img)
 

# days before event 2

    for day in range(length):
        delta = timedelta(days = day)
        target_day = time2 - delta
        # distance to new year day (January 1st) = index
        dist = (dist_ny - delta).days
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1-1)
            ds = nc.Dataset(path1)
        img = ds[var_code][dist].tolist()
        img.reverse()
        img = pad(img, pad_size)# pad 2 degree in longitude
        img = img[abs(lat_midb-90)-pad_size: abs(lat_midb- 90) + pad_size + 1, abs(lon_midb + 180)-pad_size:abs(lon_midb+ 180)+pad_size+1]
        if None in img:
            missing = None
        tensor.append(img)
        
    if missing == None:
        chanel_first = None
    else:
        chanel_first = change_into_chanel_first(tensor)
    
    return chanel_first
        
#%% read in date-coor file
import pandas as pd
#date_coor = pd.read_csv('C:/MultiHazard/Data/emdat/trial/_30day_100km_emdat_gdis_match_by_disasterno_cut.csv')
date_coor = pd.read_csv('C:/MultiHazard/Data/emdat/trial/_over_1kd_1kkm_date_coord.csv')
#%%
tensor = []
count_miss = 0
for idx, row in date_coor.iterrows():
    try:
        one_tensor = get_tensor('precipitation', row, length = 1)
        if one_tensor != None:
            tensor.append(one_tensor)
        else:
            count_miss += 1
            print('One pair is missed.')
    except:
        print('One pair is skipped.')
   
    print(idx)
#%%
import pickle
#with open ('C:/MultiHazard/Data/emdat/trial/_30day_100km_prcp_list3.pkl','wb') as handle:
with open('C:/MultiHazard/Data/emdat/trial/_over_1kd_1kkm_cut.pkl', 'wb') as handle:
    pickle.dump(tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #%%
def trial(a):
    if a==3:
        result = 4
        break
    result = 5
    return result
    