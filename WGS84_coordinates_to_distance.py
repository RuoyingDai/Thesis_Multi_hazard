# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:57:35 2022

@author: rdi420

"""

import math

lat_1 = 42.02094846	
lat_2 = 41.95929412
lon_1 = 19.4183173
lon_2 = 19.51430874

# def coor2dis(lat1, lon1, lat2, lon2):
#     R = 6371e3 # in meters
#     phi_1 = lat_1 * math.pi/180 # in radians
#     phi_2 = lat_2 * math.pi/180  # in radians
#     delta_phi = (lat_2 - lat_1) * math.pi/180 # in radians
#     delta_lambda = (lon_2 - lon_1) * math.pi/180 # in radians
#     a = math.sin(delta_phi/2) **2
#     a += math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda/2)**2 
#     c = 2* math.atan2(math.sqrt(a), math.sqrt(1-a))
#     d = R * c # in meters
#     return round(d/1000,3) # in km

#coor2dis(lat_1, lon_1, lat_2, lon_2)
#%%
from pyproj import Geod
lat_1 = 23.1291 # guangzhou
lat_2 = 52.3676 # amsterdam
lon_1 = 113.2644 # guangzhou
lon_2 = 4.9041 # amsterdam
def coor2dis(lat1, lon1, lat2, lon2):
    g = Geod(ellps='WGS84')
    az12,az21,dist = g.inv(lon1, lat1, lon2, lat2)
    return round(dist/1000,3)

#%%




