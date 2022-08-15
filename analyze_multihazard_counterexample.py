# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:09:42 2022

@author: rdi420
"""
#%% Read in files
import pickle
import pandas as pd
with open('C:/MultiHazard/Data/emdat/v3_isolate_730days_200km.pkl', 'rb') as handle:
    mh = pickle.load(handle)
o = pd.read_csv('C:/MultiHazard/Data/emdat/emdat_gdis_match_by_disasterno_90-17.csv')
#pair, interval, distance, issue_list = zip(*mh)
o = o.sample(frac=1).reset_index(drop=True)
#%% Construct the new dataframe
pair = mh
col_name = list(o.columns)[1:]
#col_name2 = [col + str(2) for col in col_name]
new_col = col_name #+ col_name2
df = pd.DataFrame(columns = new_col) 
#%% Loop through the 'pair' list
row = 0
for p in list(pair):
    row_info = o.iloc[p]
    df.loc[row] = list(row_info[1:]) 
    row += 1
    print(df)
    break
    print("Row {} is added.".format(p))
#%%
df.to_csv("C:/MultiHazard/Data/emdat/trial/_v2_over_1kd_1kkm.csv")
# I also deleted events before 1988 in Excel

#%% Delete the same event (with different locations)
# Marked with the 后缀 cut
import pandas as pd
df = pd.read_csv("C:/MultiHazard/Data/emdat/trial/_over_1kd_1kkm.csv")
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

df.to_csv("C:/MultiHazard/Data/emdat/trial/_over_1kd_1kkm_cut.csv")

