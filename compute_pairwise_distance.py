# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:49:36 2022

@author: rdi420
"""

import pandas as pd
import numpy as np

gdis = pd.read_csv("C:/MultiHazard/Data/emdat/gdis-1960-2018.csv")

arr = np.array(gdis)

import math

def coor2dis(lat_1, lon_1, lat_2, lon_2):
    R = 6371e3 # in meters
    phi_1 = lat_1 * math.pi/180 # in radians
    phi_2 = lat_2 * math.pi/180  # in radians
    delta_phi = (lat_2 - lat_1) * math.pi/180 # in radians
    delta_lambda = (lon_2 - lon_1) * math.pi/180 # in radians
    a = math.sin(delta_phi/2) **2
    a += math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda/2)**2 
    c = 2* math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c # in meters
    return round(d/1000,3) # in km

dist = [[coor2dis(arr[row, -2], arr[row, -1], arr[col, -2], arr[col, -1])for col in range(len(arr))] for row in range(len(arr))]
