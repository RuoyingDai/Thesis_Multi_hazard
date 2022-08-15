# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 12:10:19 2022

@author: rdi420
"""

a ='asv.av'
print(a.endswith('.sv'))
import os 

arr_txt = [x for x in os.listdir('C:\MultiHazard\Data\BERKEARTH_mean_temperature-anomaly_day') if x.endswith(".nc")]

for file in arr_txt:
    year = file[50:54]
    file_full = "C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/" + file
    os.rename(file_full, 'C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/t{}.nc'.format(year))
    
#%%
arr_txt = [x for x in os.listdir('C:/MultiHazard/Data/precipitation_1988-2018') if x.endswith(".nc")]

for file in arr_txt:
    year = file[40:44]
    file_full = "C:/MultiHazard/Data/precipitation_1988-2018/" + file
    os.rename(file_full, 'C:/MultiHazard/Data/precipitation_1988-2018/p{}.nc'.format(year))
