# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 17:37:53 2022

@author: rdi420

This script will link events and their distance to tectonic plates
"""
#%% Load Data
# Package
import pandas as pd
import numpy as np

# load hazard record
df = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_Nov19.csv')  
df.rename(columns = {'Total Damages (\'000 US$)': 'loss'},inplace = True)

# load city/population/coordinate list
city = pd.read_csv('C:/MultiHazard/Data/simplemaps_worldcities_basic/worldcities.csv')

# city list in lower case/coordiante/country iso name
low = [one.lower() for one in city.city.to_list()]
lat = city.lat.to_list()
lon = city.lng.to_list()
iso = city.iso3.to_list()
pop = city.population.to_list()

# Cleaning


# Some inspection
#df.groupby(['Disaster Group']).sum().loss2
# ~700 million loss in natural disaster
# ~30 million loss in thechnological disaster

#df.groupby(['Disaster Type']).sum().loss2
# Both earthquake and flood have over 200 million loss

# Geocoding
#low = [one.lower() for one in city.city]

def getCORD_POP (names):# Input: string of location
    result = []
    # If the location is not in the list , 
    # the capital will be taken into consideration
    lat_sum = 0
    lon_sum = 0
    pop_sum = 0
    location_count = 0
    #for one_city in low: # low: list of city, written in lowered case
    if names[1] != []:
        for idx, one_city in enumerate(low):   
            if one_city in names[1] and iso[idx] ==names[0]:
            # The second condition is to avoid cities in different countries
            # but have the same name. (e.g., 'Beja' in Portugal and Tunisia)
                location_count += 1
                lat_sum += lat[idx]
                lon_sum += lon[idx]
                pop_sum += pop[idx]
                #print(idx)
            #print(one_city)
            #print(location_count)
    if location_count !=0:
        lat_mean = lat_sum /location_count
        lon_mean = lon_sum /location_count
        result = [lat_mean, lon_mean, pop_sum]
    else:
    # cannot find the city and check the country instead
        row = city[(city.iso3 == names[0]) & (city.capital == 'primary')]
        if row.empty:
            result = []
        else:
            result = [row.lat.values[0], row.lng.values[0], row.population.values[0]]
    return result
#
# Add location (lat and lon) to events
lat2 = []
lon2 = []
pop2 = []
count = 0
for row in df.iterrows():
    #print(row)
    print(count)
    count +=1
    try:
        names = [row[1][11], row[1][14].lower()]
    except:
        names = [row[1][11], []]
    # iso:11, location:14
    result = getCORD_POP(names)
    if result != []:
        if isinstance(row[1]['Latitude'], str):
        #if row[1]['Latitude']!= np.nan:
            new_lat = str(row[1]['Latitude'])
            new_lat = new_lat.replace('N', '')
            if 'S' in new_lat:
                new_lat = new_lat.replace('S', '')
                new_lat = new_lat.replace('.','')
                new_lat = -float(new_lat)
                try:
                    new_lat = -float(new_lat)
                except: 
                    new_lat = np.nan             
            new_lon = str(row[1]['Longitude'])
            new_lon = new_lon.replace('W', '')
            if 'E' in new_lon:
                new_lon = new_lon.replace('E', '')
                new_lon = new_lon.replace('.', '')
                try:
                    new_lon = -float(new_lon)
                except: 
                    new_lon = np.nan
            lat2.append(new_lat)
            lon2.append(new_lon)
        else:
            lat2.append(result[0])
            lon2.append(result[1])
        pop2.append(result[2])
    else:
        lat2.append(np.nan);lon2.append(np.nan);pop2.append(np.nan)
    print ('lat is:' + str(lat2[-1]))
    
df['lat'] = lat2
df['lon'] = lon2
df['pop'] = pop2

#%% Delete empty rows
df0 =df
def replace_empty_with_nan(subject):
    column = []
    for val in subject:
#        if ((val == "") | (val == list()) | (val ==[])):
        if (len(str(val)) == 0):
            column.append(np.nan) 
        else:
            column.append(val)
    return column

# Delete empty rows
#df['lat'] = replace_empty_with_nan(lat2)
#df['lon'] = replace_empty_with_nan(lon2)
#df['pop'] = replace_empty_with_nan(pop2)

df = df.dropna(subset=['lat', 'lon','pop'])
               #inplace = True)
# 801 rows reduces to 731 rows

# Took away events without a starting/ending month
#df.dropna(subset=['Start Month', 'End Month'], inplace = True)
# 717 rows left

df.sort_values(by=['Start Year', 'Start Month', 'Start Day'], inplace = True)
#%% Convert time column into datetime format
# https://docs.python.org/3/library/datetime.html
from datetime import date
df.loc[df['Start Day'].isna(), 'Start Day'] = 1
df.loc[df['End Day'].isna(),'End Day'] = 28
df['date_start'] = [date(row[28], int(row[29]), int(row[30])) for row in df.values.tolist()]
df['date_end']  = [date(row[31], int(row[32]), int(row[33])) for row in df.values.tolist()]

df.reset_index(inplace = True)
del df['index']

#%%
df.to_csv('C:/MultiHazard/emdat_2021nov.csv')
