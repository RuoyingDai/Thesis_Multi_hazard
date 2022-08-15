# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 15:56:45 2021

@author: Ruoying
"""
#%% Load Data
# Package
import pandas as pd

# load hazard record
df = pd.read_excel('D:/MultiHazardRisk/europe_hazard.xlsx')  
df.rename(columns = {'Total Damages (\'000 US$)': 'loss'},inplace = True)

# load city/population/coordinate list
city = pd.read_excel('D:/MultiHazardRisk/worldcities.xlsx')

# city list in lower case/coordiante/country iso name
low = [one.lower() for one in city.city.to_list()]
lat = city.lat.to_list()
lon = city.lng.to_list()
iso = city.iso3.to_list()
pop = city.population.to_list()

# Cleaning
# Delete rows without 'Total damage'/'loss' record.
df.dropna(subset=['loss', 'CPI'], inplace = True)
# After deleting, 3155 rows reduces into 801 rows

# Computer current value of loss using CPI
#df[df['CPI'] == np.nan]['CPI'] =100
df['loss2'] = df['loss'] / df['CPI'] * 100

# Some inspection
df.groupby(['Disaster Group']).sum().loss2
# ~700 million loss in natural disaster
# ~30 million loss in thechnological disaster

df.groupby(['Disaster Type']).sum().loss2
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
lat2 = []
lon2 = []
pop2 = []
for row in df.iterrows():
    #print(row)
    try:
        names = [row[1][11], row[1][14].lower()]
    except:
        names = [row[1][11], []]
    # iso:11, location:14
    result = getCORD_POP(names)
    if result != []:
        lat2.append(result[0])
        lon2.append(result[1])
        pop2.append(result[2])
    else:
        lat2.append([]);lon2.append([]);pop2.append([])
        
df['lat'] = lat2
df['lon'] = lon2
df['pop'] = pop2

#
import numpy as np
def replace_empty_with_nan(subject):
    column = []
    for val in subject:
        if ((val == "") | (val == list()) | (val ==[])):
            column.append(np.nan) 
        else:
            column.append(val)
    return column

# Delete empty rows
df['lat'] = replace_empty_with_nan(lat2)
df['lon'] = replace_empty_with_nan(lon2)
df['pop'] = replace_empty_with_nan(pop2)

df.dropna(subset=['lat', 'pop'], inplace = True)
# 801 rows reduces to 731 rows

# Took away events without a starting/ending month
df.dropna(subset=['Start Month', 'End Month'], inplace = True)
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
#%% Linkage 
# Detect linkage(BY TIME)
start = df['date_start'].tolist()
end = df['date_end'].tolist()
lat3 = df['lat'].tolist()
lon3 = df['lon'].tolist()
consec = np.zeros((len(df), len(df)))
overlap_time = np.zeros((len(df), len(df)))
nb = np.zeros((len(df), len(df)))
#overlap_space = np.zeros((len(df), len(df)))
for row in range(len(df)):
    for col in range(row + 1,len(df)):
        if (start[col] - end[row]).days < 3 and (start[col] - end[row]).days >0:
            consec[row][col] = 1
            consec[col][row] = 1
        if (start[col] - end[row]).days <0:
            overlap_time[row][col] = 1
            overlap_time[col][row] = 1
        if abs(lat3[row] - lat3[col]) <1.5 or abs(lon3[row] - lon3[col])<1.5:
            nb[row][col] = 1
            nb[col][row] = 1
#%%
consec2 = [i!=0 for i in np.sum(consec*nb, axis = 0)]
# 1 degree of latitude on the sphere is 111.2 km
# 55 consecutive events out of 703 records
overlap_time2 = [i!=0 for i in np.sum(overlap_time*nb, axis = 0)]
# 249 overlapping events out of 703 records
df['consec'] = consec2
df['overlap'] = overlap_time2
#%% Create dummies for disaster type
dum = pd.get_dummies(df['Disaster Type'])
df0 = df[['lat', 'lon','pop']]
c = pd.Series([int(i) for i in df['consec']])
o = pd.Series([int(i) for i in df['overlap']])
loss = [np.log10(i) for i in df['loss2']]
#loss = df['loss2']
df2 = pd.concat([df0,dum,c, o], axis = 1)
df2.rename(columns= {0:'consec', 1:'overlap'}, inplace = True)


#%% Partial dependence
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline

from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
# Split training and testing 
X_train, X_test, y_train, y_test = train_test_split(df2, loss, 
                                                    test_size=0.2)
plt.rcParams.update({'font.size': 22})

#%%
print("Training MLPRegressor...")
tic = time()
est = make_pipeline(QuantileTransformer(),#‘identity’,0.04 ‘logistic’, 0.0‘tanh’, ‘relu’
                    MLPRegressor(activation='identity',
                        hidden_layer_sizes=(25,20,10),
                                 learning_rate_init=0.001,
                                 #solver{‘lbfgs’, ‘sgd’, ‘adam’},
                                 solver = 'sgd',
                                 alpha = 0.0001,
                                 early_stopping=True,
                                 verbose = True,
                                 validation_fraction=0.6, 
                                 beta_1=0.9, 
                                 beta_2=0.9,))
est.fit(X_train, y_train)
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(est.score(X_test, y_test)))

print('Computing partial dependence plots...')
tic = time()
# We don't compute the 2-way PDP (5, 1) here, because it is a lot slower
# with the brute method.
features = ['Drought', 'Earthquake', 'Extreme temperature']
plot_partial_dependence(est, X_train, features,
                        n_jobs=3, grid_resolution=20)
print("done in {:.3f}s".format(time() - tic))
fig = plt.gcf()
#fig.suptitle('Partial dependence of house value on non-location features\n'
#             'for the California housing dataset, with MLPRegressor')
fig.subplots_adjust(hspace=0.3)
#%%
print('Computing partial dependence plots...')
tic = time()

#features = ['Drought', 'Earthquake', 'Extreme temperature',
#            ['Drought','consec'],
#            ['Earthquake','consec'],
#            ['Extreme temperature', 'consec']]
#features = ['Drought', 'Earthquake', 'Extreme temperature',
#            ['Drought','overlap'],
#            ['Earthquake','overlap'],
#            ['Extreme temperature', 'overlap']]
#features = [['Drought','consec'],
#            ['Earthquake','consec'],
#            ['Flood', 'consec'],
#            ['Drought','overlap'],
#            ['Earthquake','overlap'],
#            ['Flood', 'overlap']]
#features = ['lat','lon','pop',
#            ['lat','consec'],
#            ['lon', 'consec'],
#            ['pop', 'consec']] 

features = ['lat','lon','pop',
            ['lat','overlap'],
            ['lon', 'overlap'],
            ['pop', 'overlap']]   
          
fig, ax = plt.subplots(figsize=(18, 14))
plot_partial_dependence(est, X_train, features,
                        n_jobs=-1, grid_resolution=20,
                        ax = ax)
                        #)
print("done in {:.3f}s".format(time() - tic))
#fig = plt.gcf()
#fig.set_figwidth(8)
#fig.set_figheight(15)
#fig.tight_layout()
fig.subplots_adjust(wspace=0.4, hspace=0.3)
#plt.savefig()

#%% Plot netowrk
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
 
#%%
colors = ['gold','goldenrod','forestgreen',
          'darkgreen','cornflowerblue','mediumblue',
          'darkslategray']
nodecolors_list = ['b', 'g']
nodecolors_list5 = ['b', 'orange', 'g', 'c', 'm']
def draw_graph(adjacency_matrix, my_labels): # my_labels
    gr = nx.Graph()
    for i in range(703):
        for j in range(703):
            wei = int(adjacency_matrix[i][j])
            if wei!= 0:
                gr.add_edge(i, j) #weight =wei/6, 
                            #color = colors[wei])
    print(nx.voterank(gr, 5))
    #labels = nx.get_edge_attributes(gr,'weight')
    #values = [val_map.get(node, 0.25) for node in G.nodes()]
    edges = gr.edges()
    #node_colors = [nodecolors_list5[cluster2[i]] for i in range(29)]
    #colors2 = [gr[u][v]['color'] for u,v in edges]
    #pos=nx.spring_layout(gr,scale=2)
    plt.figure(figsize=(16,10))
    nx.draw(gr, 
            pos=nx.random_layout(gr),
            #pos = nx.fruchterman_reingold_layout(gr),
            #pos=nx.circular_layout(gr),
            node_size=1200, 
            font_size=16,
            labels=mylabels, 
            with_labels=True,
            #####edge_labels=labels, 
            #edge_color=colors2,
            #node_color=node_colors, 
            font_color='white')
    #nx.draw_networkx_edge_labels(gr,)
    plt.show()
draw_graph(overlap_time, {i:start[i] for i in range(703)})
 