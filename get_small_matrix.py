# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:21:15 2022

@author: rdi420
"""


def cut_matrix(lat, lon, map_name, pair_id):

    extend = 1 # previously: 3 degree to four directions
    bottom = str(lat - extend)
    left = str(lon - extend)
    top = str(lat + extend)
    right = str(lon + extend)
    prjwin = '{}, {}, {}, {}'.format(left, right, bottom, top)
    
    file_name = 'C:/MultiHazard/Data/processed/x_img3/{}_{}.tif'.format(pair_id, map_name)  
    output_name = 'C:/MultiHazard/Data/processed/x_img4/{}_{}.tif'.format(pair_id, map_name)
    processing.run("gdal:cliprasterbyextent", {'INPUT':file_name,'PROJWIN':prjwin,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':output_name})
    return

#%%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

hazard_list = pd.read_csv('C:/MultiHazard/Data/emdat/trial/v2_isolate_730days_200km_1535eve.csv',header=0)

for index, row in hazard_list.iterrows():
    year = row['year']
    pair_id = row['pair_id']
    lat = row['latitude']
    lon = row['longitude']
    cut_matrix(lat, lon, 'elevation', pair_id)
    cut_matrix(lat, lon, 'ppp', pair_id)
    cut_matrix(lat, lon, 'road', pair_id)
    cut_matrix(lat, lon, 'slope', pair_id)
    cut_matrix(lat, lon, 'fault', pair_id)
    print('pair {} done.'.format(index))