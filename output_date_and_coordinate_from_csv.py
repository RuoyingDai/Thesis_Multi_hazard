# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 14:27:53 2022

@author: rdi420

Extract the column of date and coordinates
"""
import pandas as pd

df = pd.read_excel('C:/MultiHazard/Data/emdat/trial/_pos_neg_match_30km_24000pairs.xlsx', header = 0)
#%% Extract

import datetime

time_output = lambda y, m, d: datetime.date(y, m, d)

select = df[['Start Year', 'Start Month', 'Start Day']]
date1 = [time_output(select.iloc[i][0], select.iloc[i][1], select.iloc[i][2]) for i in range(len(select))]

select2 = df[['Start Year2', 'Start Month2', 'Start Day2']]
date2 = [time_output(select2.iloc[i][0], select2.iloc[i][1], select2.iloc[i][2]) for i in range(len(select))]

select3 = df[['latitude','longitude','latitude2','longitude2']].values.tolist()
#%% Output
multi = df[['multi']].values.tolist()
multi2 = [item[0] for item in multi]
output = [date1, date2, select3, multi2]
output = list(zip(*output))
pd.DataFrame(output).to_csv('C:/MultiHazard/Data/emdat/trial/_date_coord_pos_neg_match_30km_24000pairs.csv')
