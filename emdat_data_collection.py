# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:48:43 2021

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
#%% Get the field/Column names
path0 = 'C:/MultiHazard/Data/emdat/csv/emdat_public_2021_11_19_africa.csv'
with open(path0,'r',encoding='utf-8') as dest_f:
    csv_reader = csv.reader(dest_f)
    header = next(csv_reader)
# variable 'header' stores the column in em-dat database
# 50 of them in total
#%%
path = 'C:/MultiHazard/Data/emdat/csv/'
folder_file_list = os.listdir(path)
path2 = 'C:/MultiHazard/Data/emdat/emdat_Nov19.csv'

with open(path2, 'w', newline='',encoding='utf-8') as outfile:
   #reader = csv.reader(infile)
   #next(reader, None)  # skip the headers
   writer = csv.writer(outfile)
   writer.writerow(header)
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
