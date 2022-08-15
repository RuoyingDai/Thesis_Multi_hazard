# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:36:04 2021

@author: Ruoying
"""
#%%
# package
import pandas as pd
import numpy as np
import os
import timeit
import csv

#%%
start_runtime = timeit.default_timer()

path2 = 'C:/MultiHazard/Data/usgs_earthquake_1900-1979.csv'
fields = ['time', 'latitude', 'longitude',
          'depth', 'mag', 'magType', 'nst',
          'gap','dmin','rms', 'net', 'id', 'updated',
          'place', 'type', 'horizontalError', 'depthError',
          'magError', 'magNst', 'status','locationSource','magSource']

path = 'C:/MultiHazard/Data/usgs_earthquake/'
folder_file_list = os.listdir(path)

with open(path2, 'w', newline='',encoding='utf-8') as outfile:
   #reader = csv.reader(infile)
   #next(reader, None)  # skip the headers
   writer = csv.writer(outfile)
   writer.writerow(fields)
   for file in folder_file_list:
       with open(path + file,'r',encoding='utf-8') as dest_f:
           data_iter = csv.reader(dest_f,
                                   delimiter = ',')
           next(data_iter)  # skip the headers
           #data = [data for data in data_iter]  
           for row in data_iter:
               # process each row
               writer.writerow(row)

# Return the run time of this program
stop_runtime = timeit.default_timer()

print('Time: ', stop_runtime - start_runtime) 
