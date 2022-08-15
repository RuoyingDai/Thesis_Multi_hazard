# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:56:27 2022

@author: rdi420
"""

# x6: Distance to fault has been calculated in QGIS,
# The file has all information
import pandas as pd

dist = pd.read_excel('C:/MultiHazard/Data/gem_active_fault/distance2gen_fault_1kd1kkm.xlsx', header = 0)
# urbanization (two methods), population density
# match by both year and country
urban = pd.read_csv('C:/MultiHazard/Data/urbanization1975_2015/rate_ruoying_with_year.csv')
# match by country
culture = pd.read_csv('C:/MultiHazard/Data/culture_hofstedeinsights/6-dimensions-for-website-2015-08-16.csv')

# find the closest year
# https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
takeClosest = lambda num,collection:min(collection,key=lambda x:abs(x-num))
#Example:takeClosest(5,[1975,1990, 2000, 2015])

year = list(dist['year'])
close_year =  [str(takeClosest(i, [1975, 1990, 2000, 2015]))[-2:] for i in year]
country = list(dist['iso3'])
ub_country = list(urban['ISO3'])
cul_country = list(culture['ctr'])

#%%
# urbanization (built-up area/ total area)
x13 = []
# urbanization (urban population/ total population)
x14 = []
# risk avoidance
#x17 = []
# long-term orientation
#x18 = []
missed = 0
row_list = []
for row in range(len(dist)):
    try:
        yr_idx = close_year[row]
        row_country = country[row]
        # x13
        ub1_str = 'built_up_{}'.format(yr_idx)
        ub_idx = ub_country.index(row_country)
        ub1 = round(float(urban[ub1_str][ub_idx]),4)
        # x14
        ub2_str = 'urban_pop_{}'.format(yr_idx) 
        ub2 = round(float(urban[ub2_str][ub_idx]),4)
        # x17
        #cul_idx = cul_country.index(row_country)
        #risk = round(float(culture['uai'][cul_idx]),4)
        # x18
        #long = round(float(culture['ltowvs'][cul_idx]),4)
        # storm them
        x13.append(ub1)
        x14.append(ub2)
        #x17.append(risk)
        #x18.append(long)
        row_list.append(row)
    except:
        missed += 1

#%%
dist2 = dist.iloc[row_list]
dist2['urban1'] = x13
dist2['urban2'] = x14
#dist2['risk'] = x17
#dist2['long'] = x18
#%% save
dist3 = dist2[(dist2['year']>1987)]
dist3.to_csv('C:/MultiHazard/Data/processed/counter_x6_13_14_463pairs.csv')
    
#%% normalize 
df = pd.read_csv('C:/MultiHazard/Data/processed/pos_neg_x6_13_14_to_dam_2186pairs.csv')
sub = df[['multi_single', 'distance', 'urban1', 'urban2', 'HubDist']]
#%% standardize for each feature column
# and save them
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(sub)
new  =scaler.transform(sub)
import numpy as np
np.savetxt('C:/MultiHazard/Data/processed/standard_pos_neg_x6_13_14_to_dam_2186pairs.csv', 
           new, delimiter=",",
           fmt='%1.3f')
#%%
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv('C:/MultiHazard/Data/processed/standard_pos_neg_x6_13_14_to_dam_2186pairs.csv',
                 header = 0)
df = df[['distance_to_fault', 'distance_to_dam',
         'urbanization_area','urbanization_pop']]
corrMatrix = df.corr()
sn.set(font_scale=2)
#labels = ['Distance To Fault', 'Urbanization1', 'Urbanization2', 'Risk Avoidance', 'Long-term Orientation']
labels = ['Distance To Fault', 'Distance To Dam','Urbanization By Area', 'Urbanization By Population' ]


sn.heatmap(corrMatrix, annot=True,cmap="PiYG",
           annot_kws={"size":25},
           xticklabels=labels,
           yticklabels = labels,
           vmin=-1, vmax=1)

plt.show()

    