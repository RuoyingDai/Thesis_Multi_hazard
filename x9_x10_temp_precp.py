# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:24:20 2022

@author: rdi420
"""
#%% Import packages
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import pandas as pd
#%%
#https://unidata.github.io/netcdf4-python/

# input: date and coordinates pairs, return: tensors of images
#from datetime import datetime, timedelta
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

def replace_none_with_0_also_round(list_of_list):
    new = []
    for row in list_of_list:
        new_row = []
        for col in row:
            if col != None:
                new_row.append(round(col, 3))
            else:
                new_row.append(0)
        new.append(new_row)
    return new
                
       
def get_tensor(date, x , y, dist_ny):
    length = 5 # five days of accumulated precipitation
# get tensor for one event pair
    year = date.year

#    time1 = datetime.strptime(row_of_input[1], '%Y-%m-%d')
#    time2 = datetime.strptime(row_of_input[2], '%Y-%m-%d')
    #year2 = time2.year
    path1 = "C:/MultiHazard/Data/precipitation_1988-2018/p{}.nc".format(year)
    ds = nc.Dataset(path1)
    tensor = np.zeros((2,2))
    # convert the center coordinates to integer

    for daydif in range(length):
#        target_day = time1 - delta
        # distance to new year day (January 1st) = index
        dist = dist_ny - daydif
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path1 = "C:/MultiHazard/Data/precipitation_1988-2018/p{}.nc".format(year-1)
            ds = nc.Dataset(path1)
            year_length = len(ds['pr'])
            dist = year_length + dist
        daily_img = ds['pr'][dist].tolist()
        daily_img = replace_none_with_0_also_round(daily_img)
        daily_img.reverse()
        #daily_img = pad(daily_img, pad_size)# pad 2 degree in longitude
        daily_img = np.array(daily_img)
        daily_img = daily_img[y:y+2,x:x+2]
        #print(year)
        tensor += daily_img # element of the same position are summed


    #chanel_first = change_into_chanel_first(tensor)
    
    return tensor

       
       
def get_tensor_temp(date, x , y, dist_ny):
    year = date.year
    path1 = "C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/t{}.nc".format(year)
    ds = nc.Dataset(path1)
    tensor = np.zeros((2,2))

    dist = dist_ny
    if dist <0 : # If the previous year is concerned, we need to open another nc file
        path1 = "C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/t{}.nc".format(year-1)
        ds = nc.Dataset(path1)
        year_length = len(ds['tas'])
        dist = year_length + dist
    daily_img = ds['tas'][dist].tolist()
    daily_img = replace_none_with_0_also_round(daily_img)
    daily_img.reverse()
    #daily_img = pad(daily_img, pad_size)# pad 2 degree in longitude
    daily_img = np.array(daily_img)
    daily_img = daily_img[y:y+2,x:x+2]
    #print(year)
    tensor = daily_img # element of the same position are summed


    #chanel_first = change_into_chanel_first(tensor)
    
    return tensor

#%%
# read in date-coor file
date_coor = pd.read_excel('C:/MultiHazard/Data/emdat/trial/pair_range.xlsx')
#date_coor = pd.read_excel('C:/MultiHazard/Data/emdat/trial/final_v2_pos_neg_match_30km_3104pairs.xlsx')
#date_coor = pd.read_csv('C:/MultiHazard/Data/emdat/trial/_30day_100km_emdat_gdis_match_by_disasterno_cut.csv')
#date_coor = pd.read_csv('C:/MultiHazard/Data/emdat/trial/_over_1kd_1kkm_date_coord.csv')
#
#tensor_p = []
tensor_t = []
count =0

all_date = date_coor['date']
all_y = date_coor['y_up']
all_x = date_coor['x_left']
all_dist_ny = date_coor['band1']
all_info = zip(all_date, all_x, all_y, all_dist_ny)

for info in all_info:

    try:
#        tensor_p.append(get_tensor(info[0], info[1], info[2], info[3]))
        tensor_t.append(get_tensor_temp(info[0], info[1], info[2], info[3]))
    except:
#        tensor_p.append([])
        tensor_t.append([])
        print('Pair {} is skipped'.format(count))
    count += 1
    print('pair {}'.format(count))
#
#tensor_p_large = np.zeros((len(date_coor),240,240))
tensor_t_large = np.zeros((len(date_coor),240,240))
count_error = 0
for row in range(len(date_coor)):

    try:
#        tensor_p_large[row,0:120,0:120] = tensor_p[row][0][0]
#        tensor_p_large[row,120:240,0:120] = tensor_p[row][1][0]
#        tensor_p_large[row,0:120,120:240] = tensor_p[row][0][1]
#        tensor_p_large[row,120:240,120:240] = tensor_p[row][1][1]
        tensor_t_large[row,0:120,0:120] = tensor_t[row][0][0]
        tensor_t_large[row,120:240,0:120] = tensor_t[row][1][0]
        tensor_t_large[row,0:120,120:240] = tensor_t[row][0][1]
        tensor_t_large[row,120:240,120:240] = tensor_t[row][1][1]
    except:
        count_error += 1
#        tensor_p_large[row][0:120][0:120] = 0
#        tensor_p_large[row][120:240][0:120] = 0
#        tensor_p_large[row][ 0:120][120:240] = 0
#        tensor_p_large[row][120:240][120:240] = 0
        tensor_t_large[row][0:120][0:120] = 0
        tensor_t_large[row][120:240][0:120] = 0
        tensor_t_large[row][ 0:120][120:240] = 0
        tensor_t_large[row][120:240][120:240] = 0
    print('row {} enlarged.'.format(row))
        
        




#%%
import pickle
#with open ('C:/MultiHazard/Data/emdat/trial/_30day_100km_prcp_list3.pkl','wb') as handle:
with open('C:/MultiHazard/Data/processed/temp3104.pkl', 'wb') as handle:
    pickle.dump(tensor_t_large, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%    
with open('C:/MultiHazard/Data/processed/prcp3104.pkl', 'wb') as handle:
    pickle.dump(tensor_p_large, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
with open('C:/MultiHazard/Data/emdat/trial/_label_pos_neg_match_30km_4001pairs.pkl', 'wb') as handle:
    pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#%%
import pickle 
with open('C:/MultiHazard/Data/processed/temp3104.pkl', 'rb') as handle:
    x = pickle.load(handle)   
#%%
import numpy as np 
import matplotlib.pyplot as plt


plt.imshow(daily_img, interpolation='none')
plt.show()