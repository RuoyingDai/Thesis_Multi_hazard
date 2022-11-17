#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:59:25 2022

@author: pika
"""
# conda activate tfenv
# spyder
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from matplotlib import rc
rc('text', usetex=False)
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
#!pip install shap
import shap
from sklearn.preprocessing import MinMaxScaler
#!pip uninstall keras-nightly
from tensorflow import keras
import re
from bayes_opt import BayesianOptimization
import pickle
import tensorflow as tf

#from keras import regularizers
#from keras.callbacks import EarlyStopping


# Data split
df = pd.read_csv('/Users/pika/Documents/GitHub/Thesis_Multi_hazard/main/data/v27.csv')
        # intensity proxy
X = df[['int_drought',
       'int_earthquake', 'int_extreme_temp',
       'int_flood', 'int_landslide',
       'int_tropical', 'int_unknown_storm', 'int_conv_storm',
        # background variables
        'mm_fault_density', 'mm_slope',
          'mm_road_density',
        # vulnerability variables
        'multi_phdi', 'multi_ppp','single_phdi', 'single_ppp']]
y = df['ln10_Total_Damage']

#split the data 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state = 4)
# COMPUTE R SQUARED
def R_squared(x, y, name = 'R_squared'):
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x-mx, y-my
    r_num = tf.math.reduce_mean(tf.multiply(xm,ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den
# from
# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras

#
# PLOT R2 AND MSE
plt.rcParams.update({'font.size': 16})
def plot_2_metric_mean(history, name_str, p1,p2):
  #def plot_loss(history, name_str):
    #val_loss_mean = []


    plt.figure(dpi=120)
    plt.plot(history['mean_loss'], label='Training Loss',
             #color = (32/255,56/255,100/255), linewidth = 2)
             color = (225/255, 190/255, 106/255), linewidth = 2)
    plt.plot(history['mean_val_loss'], label='Validation Loss',
             #color = (125/255,125/255,191/255), linewidth = 2)
             color = (64/255, 176/255, 166/255), linewidth = 2)

    plt.title(name_str)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    plt.savefig('/Users/pika/Documents/multihazard/bo_nn_png/mse_drop{}_mom{}.png'.format(p1[:5],p2[:5]),
                dpi=120,
                bbox_inches='tight')
    plt.show()

    plt.figure(dpi=120)
    plt.plot(history['mean_R_squared'], label='Training R Squared',
             color = (32/255,56/255,100/255),  linewidth = 2)
    plt.plot(history['mean_val_R_squared'], label='Validation R Squared',
             color = (136/255,204/255,238/255), linewidth = 2)

    plt.title(name_str)
    plt.xlabel('Epoch')
    plt.ylabel('Coefficient of Determination')
    plt.legend()
    plt.savefig('/Users/pika/Documents/multihazard/bo_nn_png/r2_drop{}_mom{}.png'.format(p1[:5],p2[:5]),
                dpi=120,
                bbox_inches='tight')
    plt.show()

# NN TRAINING MODEL
def my_model0(lr, reg, lay,run_idx):

    lay_idx =round(lay)
    print(lay_idx)
    lay2 =['tanh', 'relu', 'sigmoid']
    lay3 = ['relu', 'sigmoid', 'tanh']
    lay4 = ['sigmoid', 'tanh', 'relu']
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu',
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
      #tf.keras.layers.Dense(int(hp_neuron1), activation='relu'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(256, activation=lay2[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
      #tf.keras.layers.Dense(int(hp_neuron2), activation='tanh',activity_regularizer=regularizers.l1(hp_reg)),
      tf.keras.layers.Dense(256, activation=lay3[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
      #tf.keras.layers.Dense(int(hp_neuron3), activation='sigmoid'),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(128, activation=lay4[lay_idx],
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
      tf.keras.layers.Dense(64, activation = 'tanh',
                            activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-0)),
      tf.keras.layers.Dense(32)
  ])

    #Compile the model
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),# it does not work if () is not here.
        #optimizer = tf.keras.optimizers.SGD(learning_rate= 0.005,
        #                                    momentum = momentum),# 0.05/0.02
        optimizer = tf.keras.optimizers.Adagrad(
          learning_rate = lr,
          #learning_rate=0.001,# default is 0.001
          initial_accumulator_value=0.1,
          epsilon=1e-07,
          name='Adagrad',
          ),
        metrics = [
            R_squared,
        ]
    )

    overfitCallback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                   min_delta=0.00001,
                                                   patience = 500,
                                                   mode = 'min')
    history = model.fit(np.array(X_train), np.array(y_train),
            batch_size=5,
            epochs=20,
            verbose=True,
            #verbose =False,
            validation_split = 0.2,
    callbacks=[overfitCallback])
    model.predict(X_test, y_test)
    return history.history

# Save and plot mean of 3 runs
def plot_history_dict(drop_out_rate, momentum,
                      history_list, num_run):
    # hyperparameter 1
    p1 = str(drop_out_rate).replace(".", "_" )# drop out rate
    # hyperparameter 2
    p2 = str(momentum).replace(".", "_")# momentum
    keys0 = history_list[0].keys()
    keys = [key for key in keys0]
    key_mean_series = {}

    for key in keys:
        # possible problem: cannot calculate the mean
        key_mean_series['mean_'+key] = list(np.mean([history_list[i][key] for i in range(len(history_list))], axis = 0))
    with open('/Users/pika/Documents/multihazard/bo_nn/mean_drop{}_mom{}'.format(p1[:5],p2[:5]), "wb") as fp:   #Pickling
        pickle.dump(key_mean_series, fp)
    # Actually plotting
    #plot_2_metric_mean(key_mean_series,
    #                   'Drop out: {}/ Momentum: {}'.format(round(drop_out_rate,4),round(momentum,4)),
    #                   p1, p2)
    # return the validation loss of the last epoch
    val_loss_mean = key_mean_series['mean_val_loss'][-1]
    print("val_loss_mean is : {}".format(round(val_loss_mean, 2)))
    return val_loss_mean

# #%% RUN NN TRAINING MODEL my_model0
# learning_rate = 2e-3
# regularization = 6
# layer = 1
# num_run = 1
# history = [[] for i in range(num_run)]
# val_metric = 0
# for run_idx  in range(num_run):
#     print('Same setting run: {}'.format(run_idx + 1))
#     history_run = my_model0(learning_rate, regularization,layer, run_idx)
#     history[run_idx] = history_run
#     val_metric += history_run['val_loss'][-1]
# p1 = str(learning_rate).replace(".", "_" )# learning rate
# # hyperparameter 2
# p2 = str(regularization).replace(".", "_")# regularization
# p3 = round(layer) # activation function
# with open('/Users/pika/Documents/multihazard/bo_nn/adagrad_lr{}_reg{}_lay{}'.format(p1[:5],p2[:5],p3), "wb") as fp:   #Pickling
#     pickle.dump(history, fp)

# RUN MODEL (The same content as model_0)
# Learning rate
lr =  1e-4
reg = 1
lay = 2
lay_idx =round(lay)
print(lay_idx)
lay2 =['tanh', 'relu', 'sigmoid']
lay3 = ['relu', 'sigmoid', 'tanh']
lay4 = ['sigmoid', 'tanh', 'relu']
model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu',
                      activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
#tf.keras.layers.Dense(int(hp_neuron1), activation='relu'),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(256, activation=lay2[lay_idx],
                      activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
#tf.keras.layers.Dense(int(hp_neuron2), activation='tanh',activity_regularizer=regularizers.l1(hp_reg)),
tf.keras.layers.Dense(256, activation=lay3[lay_idx],
                      activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-1, l2=reg*1e-0)),
#tf.keras.layers.Dense(int(hp_neuron3), activation='sigmoid'),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(128, activation=lay4[lay_idx],
                      activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-1)),
tf.keras.layers.Dense(64, activation = 'tanh',
                      activity_regularizer = tf.keras.regularizers.L1L2(l1=reg*1e-2, l2=reg*1e-0)),
tf.keras.layers.Dense(32)
])

  #Compile the model
model.compile(
    loss = tf.keras.losses.MeanSquaredError(),# it does not work if () is not here.
    #optimizer = tf.keras.optimizers.SGD(learning_rate= 0.005,
    #                                    momentum = momentum),# 0.05/0.02
    optimizer = tf.keras.optimizers.Adagrad(
      learning_rate = lr,
      #learning_rate=0.001,# default is 0.001
      initial_accumulator_value=0.1,
      epsilon=1e-07,
      name='Adagrad',
      ),
    metrics = [
        R_squared,
    ]
)

overfitCallback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                               min_delta=0.00001,
                                               patience = 500,
                                               mode = 'min')
history = model.fit(np.array(X_train), np.array(y_train),
        batch_size=5,
        epochs=2,
        verbose=True,
        #verbose =False,
        validation_split = 0.2,
callbacks=[overfitCallback])
# This function is also not working
#model.predict(X_test.values, y_test.values)
#%% SHAP value of my_model0
#
X_train_summary = shap.kmeans(X_train, 10)
ex = shap.KernelExplainer(model.predict, X_train_summary)

#%% SHAP value based on test data
# Dimension: last layer neuron # * sample # * feature #
shap_values = ex.shap_values(X_test.iloc[0:50,:])
# average over all neurons in the last layer

#%% overall SHAP value plot
s = np.mean(shap_values, axis = 0)
shap.summary_plot(s, X_test.iloc[0:50,:])

#%% SHAP interaction term
plt.rcParams.update({'font.size': 25})
shap.dependence_plot("multi_phdi", s, X_test.iloc[0:50,:], interaction_index="mm_slope", show=False,cmap=plt.get_cmap("bwr"),dot_size=40)
#plt.savefig('/Users/pika/Documents/multihazard/bo_nn_png/trial_shap2.png', format='png', dpi=150, bbox_inches='tight')


    
 
    


#%% Bayes Optimization
pbounds = {'learning_rate': (1e-3, 5e-2),
           'regularization': (10,30),
           'layer':(-0.5, 2.5),
           #'drop_out_rate': (0.2, 0.5),
           #'momentum': (0.1, 0.5)
           }
optimizer = BayesianOptimization(
    f = my_model,
    pbounds = pbounds,
    random_state = 42)

optimizer.maximize(
    init_points = 1,
    n_iter = 4)
#%%
for i, res in enumerate(optimizer.res):
    print('Iteration {}: \n\t{}'.format(i, res))
    break

#%%

#file_name = "adagrad_lr1_985_reg41_44_lay1"#1.48/1.39/crazy r2
#file_name = "adagrad_lr0_061_reg5_541_lay1"#1.45/0.75/0.6/0.4
aaa = pickle.load(open( "/Users/pika/Documents/multihazard/bo_nn/"+file_name, "rb"))
#%%
plt.plot(aaa[0]['loss'])
#%%
plt.plot(aaa[0]['val_loss'])
#%%
plt.plot(aaa[0]['R_squared'])
#%%
plt.plot(aaa[0]['val_R_squared'])
