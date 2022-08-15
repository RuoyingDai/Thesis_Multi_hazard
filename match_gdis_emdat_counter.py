# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:23:32 2022

@author: rdi420

# match emdat and gdis
# by the column 'disaster no'
# select the non multi-hazard pairs

# new csv wiil have one pair of hazard per row
"""

import pandas as pd
# 
emdat = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_Nov19.csv', 
                 header = 0)
gd = pd.read_csv('C:/MultiHazard/Data/emdat/gdis-1960-2018.csv',
                    header = 0)
#%%
# Get 
e_no = [row[:9] for row in emdat['Dis No']]
g_no = gdis['disasterno']
length = len(g_no)
useful = emdat[[ 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month',
       'End Day']]
use = useful.values.astype(int).tolist()
#use = [[int(item) for item in row if not pd.isna(item)] for row in use]
s1 = [[] for i in range(length)]
s2 = [[] for i in range(length)]
s3 = [[] for i in range(length)]
e1 = [[] for i in range(length)]
e2 = [[] for i in range(length)]
e3 = [[] for i in range(length)]
for row in range(length):
    print(row)
    try:
        no = g_no[row]
        idx = e_no.index(no)
        s1[row] = use[idx][0]        
        s2[row] = use[idx][1]        
        s3[row] = use[idx][2] 
        e1[row] = use[idx][3]        
        e2[row] = use[idx][4]        
        e3[row] = use[idx][5]       
    except:
        continue
#%%
gdis['Start Year'] = s1
gdis['Start Month'] = s2
gdis['Start Day'] = s3
gdis['End Year'] = e1
gdis['End Month'] = e2
gdis['End Day'] = e3
#%%
gdis.to_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno.csv')

#%% matching process

from datetime import datetime
import random

def days_between(d1, d2):
    d1b = datetime.strptime(d1, "%Y-%m-%d")
    d2b = datetime.strptime(d2, "%Y-%m-%d")
    return (d2b - d1b).days

import pandas as pd
# 32670 events
# gd = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno.csv',
#                  header = 0)
# useful = gd[[ 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month',
#        'End Day']]
# coor = gd[['latitude','longitude']].values.tolist()
# day = useful.values.tolist()

def new_csv(match_list):
    return 0
 
from pyproj import Geod
lat_1 = 23.1291 # guangzhou
lat_2 = 52.3676 # amsterdam
lon_1 = 113.2644 # guangzhou
lon_2 = 4.9041 # amsterdam
def coor2dis(lat1, lon1, lat2, lon2):
    g = Geod(ellps='WGS84')
    az12,az21,dist = g.inv(lon1, lat1, lon2, lat2)
    return round(dist/1000,3)   

# August 3rd: make sure one event has no event in 720 days afterwards within 30 days
def get_one_isolate_event(day_threshold, dist_threshold):
    # in day and in kilometer(km)
    gd = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno_90-17.csv',
                 header = 0)
    gd = gd.sample(frac=1).reset_index(drop=True)
    row_index = gd['row']
    useful = gd[[ 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month',
       'End Day']]
    no = gd['disasterno'].values.tolist()
    coor = gd[['latitude','longitude']].values.tolist()
    day = useful.values.tolist()
    count = 0
    isolate_list = []
    for i1 in range(len(gd)):
        check = 0
        for i2 in range(len(gd)):
            if i2>i1:
                d1 = '{}-{}-{}'.format(day[i1][0], day[i1][1], day[i1][2])
                d2 = '{}-{}-{}'.format(day[i2][3], day[i2][4], day[i2][5])
                try:
                    mid = days_between(d1, d2) # d2-d1 
                except:
                    break
                # time threshold
                if mid< day_threshold and no[i1] != no[i2]:
                    dist = coor2dis(coor[i1][0], coor[i1][1], coor[i2][0], coor[i2][1])
                    # distance threshold
                    if dist < dist_threshold:
                        check = 1
                        break
            if i2<i1:
                d1 = '{}-{}-{}'.format(day[i1][0], day[i1][1], day[i1][2])
                d2 = '{}-{}-{}'.format(day[i2][3], day[i2][4], day[i2][5])
                try:
                    mid = days_between(d2, d1) # d1-d2 
                except:
                    break
                # time threshold
                if mid< day_threshold and no[i1] != no[i2]:
                    dist = coor2dis(coor[i1][0], coor[i1][1], coor[i2][0], coor[i2][1])
                    # distance threshold
                    if dist < dist_threshold:
                        check = 1
                        break                
        if check ==0:
            isolate_list.append(i1)
            #day_dif_list.append(mid)
            #dist_list.append(dist)
            count += 1


        print('Event {} checked.'.format(i1))
        print('Count {}.\n'.format(count))
        
        if count > 2000:
            break
        
    return(isolate_list)


import pickle #credits to stack overflow user= blender

#%% Search
# amsterdam to warsaw is around 1000 km in distance
# 1000 days is more than 2 years
a = get_one_isolate_event(day_threshold = 730, dist_threshold = 200)
with open('C:/MultiHazard/Data/emdat/v3_isolate_730days_200km.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
with open('C:/MultiHazard/Data/emdat/match_30d_100km.pkl', 'rb') as handle:
    b = pickle.load(handle)

#%%
col_name = list(gd.columns)[1:]
#col_name2 = [col + str(2) for col in col_name]
new_col = col_name #+ col_name2
df = pd.DataFrame(columns = new_col) 
# Loop through the 'pair' list
row = 0
for p in list(a):
    row_info = gd.iloc[p]
    df.loc[row] = list(row_info[1:]) 
    row += 1
    print("Row {} is added.".format(p))
#%%
df.to_csv("C:/MultiHazard/Data/emdat/trial/isolate_730days_200km_2001eve.csv")


                    
                
    
    
