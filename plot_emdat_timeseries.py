# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:26:44 2022

@author: rdi420

purpose: plot time series of 5 groups of disasters
"""
import pandas as pd
df = pd.read_csv('C:/MultiHazard/Data/cut_emdat_subgroup_info.csv')
#%%
import plotly.express as px

#df = px.data.stocks(indexed=True)-1
fig = px.area(df, facet_col="Category", facet_col_wrap='Count')
fig.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (10, 5)
plt.style.use('fivethirtyeight')

#df['year'] = pd.to_datetime(df['Year'])

# Set the date column as the index of your DataFrame meat
df = df.set_index('Year')

# Print the summary statistics of the DataFrame
print(df.describe())
ax = df.plot(linewidth=2, fontsize=12);
#%%
import pandas as pd
df1 = pd.read_csv('C:/MultiHazard/Data/distance_to_plate_cut.csv')
df2 = pd.read_csv('C:/MultiHazard/Data/distance_to_fault_essential.csv')
df3 = pd.merge(
    left=df1,
    right=df2,
    left_on="Dis No",
    right_on="Dis No",
    how="inner"
)
