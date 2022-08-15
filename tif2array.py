# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 14:18:48 2022

@author: rdi420
"""

from PIL import Image
import numpy
startx = 24.0416666666666572
starty = 30.0416666666666572
stopx = 40.9500000000000028
stopy = 46.9500000000000028

Image.open("C:/MultiHazard/Data/processed/x_img/p1_elevation.tif")
cropped_image = image.crop((startx,starty,stopx,stopy))
cropped_image.save("C:/MultiHazard/Data/processed/trial.tif")
#%%
print("ipp"+str(startfile)+" cropped")
    image1 = cropped_image.rotate(180,0,1)
    image1.save("Filepath"+str(startfile),".tif")

#%%
import numpy
#from osgeo import gdal
import gdal
#%%
ds = gdal.Open("C:/MultiHazard/Data/processed/trai_sat.tif")
dem1 = ds.GetRasterBand(1).ReadAsArray()
dem2 = ds.GetRasterBand(2).ReadAsArray()
dem = dem1 + dem2

#%%
#%%
import numpy as np
#from osgeo import gdal
import gdal
import os 
import torch
folder = "C:/MultiHazard/Data/processed/x_img/"
#arr_txt = [x for x in os.listdir(folder) if x.endswith(".nc")]
import pickle 
with open('C:/MultiHazard/Data/processed/temp3104.pkl', 'rb') as handle:
    all_temp = pickle.load(handle) 
with open('C:/MultiHazard/Data/processed/prcp3104.pkl', 'rb') as handle:
    all_prcp = pickle.load(handle)

#%%
def change_into_chanel_first(matrix):
    chanel_total = len(matrix)
    row_total = len(matrix[0])
    col_total = len(matrix[0])
    result = np.zeros([row_total, col_total, chanel_total])
    for chanel in range(chanel_total):
        for row in range(row_total):
            for col in range(col_total):
                result[row][col][chanel] = matrix[chanel][row][col]
    return(result)

def get_x_img_one_pair(pair_id, feature_seq, row_in_file):
    x = []
    for feature in feature_seq:
        file_name = folder + 'p{}_{}.tif'.format(pair_id, feature)
        handle = gdal.Open(file_name)
        file = handle.GetRasterBand(1).ReadAsArray()
        x.append(file)
    x.append(all_prcp[row])
    x.append(all_temp[row])
        
   
    return change_into_chanel_first(np.array(x))

feature_seq = ['elevation', 'ppp', 'slope', 'road']

tensor = []
p0 = ['p{}'.format(i) for i in range(1, 1536)]
p1 = ['p{}'.format(i) for i in range(1627,3196)]
all_pair_id = p0 + p1

for row, pair_id in enumerate(all_pair_id):
    print(pair_id)
    tensorpt = torch.from_numpy(get_x_img_one_pair(pair_id, feature_seq, row))
    torch.save(tensorpt, 'C:/MultiHazard/Data/emdat/trial/x_pt/x_p{}.pt'.format(pair_id))
    if row > 3:
        break
#%%
tensorpt = torch.from_numpy(np.array(x))
torch.save(tensorpt, 'C:/MultiHazard/Data/emdat/trial/x.pt')

#%%
filename = 'C:/MultiHazard/Data/emdat/trial/x.pkl'
infile = open(filename, 'rb')
x = pickle.load(infile)
infile.close()
#%%


