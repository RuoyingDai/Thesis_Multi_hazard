#!/usr/bin/env python
# coding: utf-8

# ### Example of training neural network for multi hazard project

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
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
import re
#from bayes_opt import BayesianOptimization
import pickle


# In[1]:


from platform import python_version

print(python_version())


# In[ ]:


from tensorflow import keras


# In[2]:


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)


# In[3]:


from sklearn.linear_model import LinearRegression


# In[4]:


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)


# In[5]:


# Moved all helper functions (such as plot functions) to utils/utils.py
# Here we import them
from utils.utils import (R_squared, plot_2_metric_mean)


# In[6]:


# Moved all model functions to utils/model.py
from utils.model import (my_model0, my_model)


# ### Load the data from csv file

# In[3]:


df = pd.read_csv('data/v27.csv')


# In[4]:


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


# In[5]:


#split the data 80% training and 20% testin
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state = 4)


# In[6]:


data_train = X_train.copy()


# In[7]:


data_train['y_train'] = y_train.to_numpy().reshape(-1,1)


# ### Let's have a look at the features

# In[13]:


#sns.pairplot(X_train[X_train.columns.to_list()], diag_kind='kde')
sns.pairplot(data_train[data_train.columns.to_list()], diag_kind='kde')


# In[293]:


#data_train.describe().transpose()


# ### Fit linear regression

# In[294]:


model = LinearRegression().fit(X_train, y_train)


# In[295]:


yhat = model.predict(X_test)


# In[296]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# In[297]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[ ]:





# ### Model equivalent to linear regression

# In[198]:


linear_model = tf.keras.Sequential([
    layers.Dense(units=1)
])


# In[199]:


linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# In[200]:


history_linear = linear_model.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history_linear)


# In[201]:


yhat = linear_model.predict(X_test)


# In[202]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[204]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# In[ ]:





# ### Model 2

# In[217]:


tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
                               tf.keras.layers.Dense(1),
                               tf.keras.layers.Dense(1)
])
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
history2 = model_2.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history2)


# In[218]:


yhat = model_2.predict(X_test)


# In[219]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[220]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# ### Model 3 (adam optimizer)

# In[233]:


tf.random.set_seed(42)
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])
history3 = model_3.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history3)


# In[234]:


yhat = model_3.predict(X_test)


# In[235]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[236]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# ### Model 4

# In[1]:


tf.random.set_seed(42)

model_4 = tf.keras.Sequential([

  tf.keras.layers.Dense(100),

  tf.keras.layers.Dense(10),

  tf.keras.layers.Dense(1)
])

model_4 = tf.keras.Sequential([
tf.keras.layers.Dense(100),
tf.keras.layers.Dense(10),
tf.keras.layers.Dense(1)
])
model_4.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.Adam(),
metrics=['mae'])

model_4.compile(loss=tf.keras.losses.mae,

                optimizer=tf.keras.optimizers.Adam(),

                metrics=['mae'])

model_4.fit(X_train, y_train, epochs=100, verbose=0)

history4 = model_4.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history4)


# In[ ]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[ ]:


yhat = model_4.predict(X_test)


# In[ ]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# ### Model 5

# In[222]:


tf.random.set_seed(42)
model_5 = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation = tf.keras.activations.relu),
  tf.keras.layers.Dense(1)
])
model_5.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])
#model_5.fit(X_train, y_train, epochs=100, verbose=0)


# In[223]:


history5 = model_5.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history5)


# In[224]:


yhat = model_5.predict(X_test)


# In[225]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[ ]:





# In[226]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# ### Model 6

# In[8]:


tf.random.set_seed(42)
model_6 = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation = tf.keras.activations.relu),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Dense(1)
])
model_6.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])
#model_6.fit(X_train, y_train, epochs=100, verbose=0)


# In[299]:


history6 = model_6.fit(
    X_train,
    y_train,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
plot_loss(history6)


# In[300]:


plt.plot(y_test.to_numpy().reshape(-1,1), 'kx')
plt.plot(yhat, 'rx')


# In[301]:


yhat = model_6.predict(X_test)


# In[302]:


mae = tf.metrics.mean_absolute_error(y_true=y_test, 
                                     y_pred=yhat.squeeze()).numpy()
mse = tf.metrics.mean_squared_error(y_true = y_test,
                                      y_pred=yhat.squeeze()).numpy()
print(mae)
print(mse)


# ### Shapley values (you can do this for each model to check how consistent this is)

# In[9]:


e = shap.KernelExplainer(model_6, X_train)
shap_values = e.shap_values(X_test)


# In[10]:


shap.initjs()
# visualize the first prediction's explanation with a force plot
shap.force_plot(e.expected_value[0], shap_values[0][0], features = features)


# In[11]:


#%% overall SHAP value plot
s = np.mean(shap_values, axis = 0)
shap.summary_plot(s, X_test)


# In[ ]:





# In[ ]:


import gpflow
k0 = gpflow.kernels.RBF(active_dims=[0])
k1 = gpflow.kernels.RBF(active_dims=[1])
k2 = gpflow.kernels.RBF(active_dims=[2])
k3 = gpflow.kernels.RBF(active_dims=[3])
k4 = gpflow.kernels.RBF(active_dims=[4])
k5 = gpflow.kernels.RBF(active_dims=[5])
k6 = gpflow.kernels.RBF(active_dims=[6])
k7 = gpflow.kernels.RBF(active_dims=[7])
k8 = gpflow.kernels.RBF(active_dims=[8])
k9 = gpflow.kernels.RBF(active_dims=[9])
k10 = gpflow.kernels.RBF(active_dims=[10])
k11 = gpflow.kernels.RBF(active_dims=[11])
k12 = gpflow.kernels.RBF(active_dims=[12])
k13 = gpflow.kernels.RBF(active_dims=[13])
k14 = gpflow.kernels.RBF(active_dims=[14])

k = k0+k1+k2+k3+k4+k5+k6+k7+k8+k9+k10+k11+k12+k13+k14
m = gpflow.models.GPR((X_train, y_train), kernel=k)
# m.likelihood.variance.assign(1e-6)

opt = gpflow.optimizers.Scipy()
opt.minimize(m.training_loss, variables=m.trainable_variables)


# In[ ]:




