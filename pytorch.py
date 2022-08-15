# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:00:29 2022

@author: rdi420
"""
#%% Import packages

# importing the libraries
import pandas as pd
import numpy as np

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
#%% Load datasets

import pickle
with open('C:/MultiHazard/Data/emdat/trial/x.pkl', 'rb') as handle:
    x = pickle.load(handle)
y = np.ones(3195)
y[:1626] = 0
# loading dataset
#train = pd.read_csv('train_LbELtWX/train.csv')
#test = pd.read_csv('test_ScVgIM0/test.csv')

#sample_submission = pd.read_csv('sample_submission_I5njJSF.csv')

#train.head()
#full = np
#%%
# loading training images
train_img = []
for img_name in tqdm(train['id']):
    # defining the image path
    image_path = 'train_LbELtWX/train/' + str(img_name) + '.png'
    # reading the image
    img = imread(image_path, as_gray=True)
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
# defining the target
train_y = train['label'].values
train_x.shape