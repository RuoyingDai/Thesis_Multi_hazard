# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:09:42 2022

@author: rdi420
"""
#%% Read in files
import pickle
import pandas as pd
with open('C:/MultiHazard/Data/emdat/match_60d_100km2.pkl', 'rb') as handle:
    mh = pickle.load(handle)
o = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno.csv')
#pair, interval, distance, issue_list = zip(*mh)
#%% Construct the new dataframe
pair = mh[0]
col_name = list(o.columns)[1:]
col_name2 = [col + str(2) for col in col_name]
new_col = col_name + col_name2
df = pd.DataFrame(columns = new_col) 
#%% Loop through the 'pair' list
row = 0
for p in list(pair):
    print(p)
    e1 = p[0]
    e2 = p[1]
    e1_row = o.iloc[e1]
    e2_row = o.iloc[e2]
    df.loc[row] = list(e1_row[1:]) + list(e2_row[1:])
    row += 1
    print("Row {} is added.".format(row))
#%%
df.to_csv("C:/MultiHazard/Data/emdat/trial/_60day_100km_emdat_gdis_match_by_disasterno_no_cut.csv")

#%% Delete the same event (with different locations)
import pandas as pd
df = pd.read_csv("C:/MultiHazard/Data/emdat/trial/_60day_100km_emdat_gdis_match_by_disasterno_no_cut.csv")
#df.drop([1,2,3], inplace =True)
count =0
for idx in range(len(df)):

    try:
        num1 = df.loc[idx]['disasterno']
        num2 = df.loc[idx]['disasterno2']
        to_del = df.index[(df['disasterno2'] == num2) & (df['disasterno'] == num1)].tolist()
        to_del.remove(idx)
        df.drop(to_del, inplace = True)
    except:
        count += 1

df.to_csv("C:/MultiHazard/Data/emdat/trial/_60day_100km_emdat_gdis_match_by_disasterno_cut.csv")

