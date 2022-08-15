# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:23:32 2022

@author: rdi420

# match emdat and gdis
# by the column 'disaster no'

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

#%% You better check the accuracy of the matching!! Checked

from datetime import datetime

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


def match_two_events(day_threshold, dist_threshold):
    # in day and in kilometer(km)
    gd = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno.csv',
                 header = 0)
    useful = gd[[ 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month',
       'End Day']]
    no = gd['disasterno'].values.tolist()
    coor = gd[['latitude','longitude']].values.tolist()
    day = useful.values.tolist()
    count = 0
    match_list = []
    day_dif_list = []
    dist_list = []
    issue_list = []
    for i1 in range(len(gd)):
        for i2 in range(len(gd)):
            try:
                d1 = '{}-{}-{}'.format(day[i1][0], day[i1][1], day[i1][2])
                d2 = '{}-{}-{}'.format(day[i2][3], day[i2][4], day[i2][5])
                mid = days_between(d1, d2)
                if mid<day_threshold and mid>-1 and no[i1] != no[i2]:
                    dist = coor2dis(coor[i1][0], coor[i1][1], coor[i2][0], coor[i2][1])
                    if dist < dist_threshold:
                        match_list.append(list([i1, i2]))
                        day_dif_list.append(mid)
                        dist_list.append(dist)
                        count += 1
                        print(count)
            except:
                    issue_list.append(list([i1, i2]))
        print('Event {} checked.'.format(i1))
        result = [match_list, day_dif_list, dist_list, issue_list]
        with open('C:/MultiHazard/Data/emdat/match_30d_100km2.pkl', 'wb') as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return(result)


import pickle #credits to stack overflow user= blender

#%%
a = match_two_events(30, 100)
#with open('C:/MultiHazard/Data/emdat/match_60d_100km2.pkl', 'wb') as handle:
#    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
with open('C:/MultiHazard/Data/emdat/match_60d_100km.pkl', 'rb') as handle:
    b = pickle.load(handle)


#return(list(match_list, day_dif_list, dist_list, issue_list))

#TypeError: list expected at most 1 argument, got 4
                    
                
    
    
