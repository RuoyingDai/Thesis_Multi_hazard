# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:47:22 2022

@author: rdi420
"""
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, 
                       QgsProcessingParameterFeatureSource, 
                       QgsProcessingParameterVectorDestination)
import processing 
#%%
import sys
#sys.path.append("C:\Program Files\QGIS 3.16\apps\Python27\Lib")
sys.path.append("C:\Program Files\QGIS 3.16\apps\Python37\Lib\site-packages\nose2\plugins")
#import qgis.core

def get_1_matrix(lat, lon, map_name, pair_id):

    extend = 1 # 3 degree to four directions
    bottom = str(lat - extend)
    left = str(lon - extend)
    top = str(lat + extend)
    right = str(lon + extend)
    prjwin = '{}, {}, {}, {}'.format(left, right, bottom, top)
    
   # year = date
    
    # Assign address by map_name:
    # 'elevation', 'slope', 'road', 'prcp', 'temp', 'smois', 'vegt', 'ppp', 'human'
    
    # Upscale (by raster calculator)
    # Add reference system (EPSG:4326)
    # Convert it to tif file (Since it is easier to open in python numpy array)
    
    # The following 3 varaibles are constant through time.
    if map_name == 'elevation': # file format: asc
        file_name = 'C:/MultiHazard/Data/Glo30_tif.tif'
    if map_name == 'slope': # file format: tif
        # 43200， 16800
        file_name = 'C:/MultiHazard/Data/slope_1KMmd_GMTEDmd.tif'
    if map_name == 'road': #file format: asc
        file_name = 'C:/MultiHazard/Data/grip4_total_dens_m_km2_tif.tif'
    if map_name == 'fault':
        file_name = 'C:/MultiHazard/Data/gem_active_fault/v2_fault_density.tif'
        
    # The following 6 variables are changing through time
    # and therefore we have to find the file with nearest time available.
    if map_name == 'prcp': # file format: netcdf
        prefix = 'C:/MultiHazard/Data/precipitation_1988-2018/'
        file_name = prefix + 'GPCC_total_precipitation_day_1x1_global_{}_v2020.0-v6.0-fg.nc'.format(year)
    if map_name == 'temp': # file format: netcdf
        prefix = 'C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/'
        file_name = prefix + 'BERKEARTH_mean_temperature-anomaly_day_1x1_global_{}_v1.0.nc'.format(year)
    #if map_name == 'smois': # file format: netcdf  
    #    prefix = 'C:/MultiHazard/Data/soil_moisture/data/'
    #    file_name = 
    #if map_name == 'vegt': # file format: 
    # still need to download it in batch
    if map_name == 'ppp': # file format: netcdf
        file_name = 'C:/MultiHazard/Data/PPP/GDP_PPP_30arcsec_v3.nc'
    #if map_name == 'human': # file format: tif
    #    file_name = 'C:/MultiHazard/Data/population_GHS/GHS_POP_E{}_GLOBE_R2019A_4326_9ss_V1_0/GHS_POP_E{}_GLOBE_R2019A_4326_9ss_V1_0.tif'.format(year, year)
    
    output_name = 'C:/MultiHazard/Data/processed/x_img2/{}_{}.tif'.format(pair_id, map_name)
    processing.run("gdal:cliprasterbyextent", {'INPUT':file_name,'PROJWIN':prjwin,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':output_name})
    return
# covert format



def get_1_matrix_with_time(lat, lon, map_name, year, pair_id):

    extend = 1 # 3 degree to four directions
    bottom = str(lat - extend)
    left = str(lon - extend)
    top = str(lat + extend)
    right = str(lon + extend)
    prjwin = '{}, {}, {}, {}'.format(left, right, bottom, top)
    
   # year = date
    
    # Assign address by map_name:
    # 'elevation', 'slope', 'road', 'prcp', 'temp', 'smois', 'vegt', 'ppp', 'human'
    
    # Upscale (by raster calculator)
    # Add reference system (EPSG:4326)
    # Convert it to tif file (Since it is easier to open in python numpy array)
        
    # The following 6 variables are changing through time
    # and therefore we have to find the file with nearest time available.
    if map_name == 'prcp': # file format: netcdf
        prefix = 'C:/MultiHazard/Data/precipitation_1988-2018/'
        file_name = prefix + 'GPCC_total_precipitation_day_1x1_global_{}_v2020.0-v6.0-fg.nc'.format(year)
    if map_name == 'temp': # file format: netcdf
        prefix = 'C:/MultiHazard/Data/BERKEARTH_mean_temperature-anomaly_day/'
        file_name = prefix + 'BERKEARTH_mean_temperature-anomaly_day_1x1_global_{}_v1.0.nc'.format(year)
    #if map_name == 'smois': # file format: netcdf  
    #    prefix = 'C:/MultiHazard/Data/soil_moisture/data/'
    #    file_name = 
    #if map_name == 'vegt': # file format: 
    # still need to download it in batch
    if map_name == 'ppp': # file format: netcdf
        file_name = 'C:/MultiHazard/Data/PPP/ppp{}.tif'.format(year)
    if map_name == 'human': # file format: tif
        file_name = 'C:/MultiHazard/Data/population_GHS/GHS_POP_E{}_GLOBE_R2019A_4326_9ss_V1_0/GHS_POP_E{}_GLOBE_R2019A_4326_9ss_V1_0.tif'.format(year, year)
    
    output_name = 'C:/MultiHazard/Data/processed/x_img3/{}_{}.tif'.format(pair_id, map_name)
    processing.run("gdal:cliprasterbyextent", {'INPUT':file_name,'PROJWIN':prjwin,'NODATA':None,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':output_name})
    return


def get_acc_matrix(lat, lon, date):
# days before event 1
    input_raster = processing.runandload("saga:rastercalculator", Input_Raster, '', 'a / ' + str(Number), False, 7, None)
    for day in range(length):
        delta = timedelta(days = day)
        target_day = time1 - delta
        # distance to new year day (January 1st) = index
        dist = (dist_ny - delta).days
        if dist <0 : # If the previous year is concerned, we need to open another nc file
            path1 = "C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_{}_day_1x1_global_{}_v2020.0-v6.0-fg.nc".format(var, year1-1)
            ds = nc.Dataset(path1)
        img = ds[var_code][dist].tolist()
        img.reverse()
        img = pad(img, pad_size)# pad 2 degree in longitude
        img[img == None] = 0
        img = img[abs(lat_midb-90)-pad_size : abs(lat_midb- 90) + pad_size + 1, abs(lon_midb + 180)-pad_size +pad_size:abs(lon_midb+ 180)+2*pad_size+1]
        tensor.append(img)    
    
    return

"C:/MultiHazard/Data/precipitation_1988-2018/GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg.nc"
# coordiantes of amsterdam
# 52.3676° N, 4.9041° E
# 2.9041, 6.9041, 50.3676, 54.3676 [EPSG:4326]
# 2.9041, 50.3676, 6.9041,54.3676  [EPSG:4326]
#%% Function to collect 7 matrix

def get_constant_matrix(lat, lon, pair_id, year = 999):
    #one_event_matrix = []

    #get_1_matrix(lat, lon, 'elevation', pair_id)
    #get_1_matrix(lat, lon, 'slope', pair_id)
    #get_1_matrix(lat, lon, 'road', pair_id)
    get_1_matrix(lat, lon, 'fault', pair_id)
    #get_1_matrix_with_time(lat, lon, 'ppp', year, pair_id)
    return_str = 'Pair {} is cut.'.format(pair_id)
    
    return return_str

#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

hazard_list = pd.read_excel('C:/MultiHazard/Data/emdat/trial/final_v2_pos_neg_match_30km_3104pairs.xlsx',header=0)
#hazard_list = pd.read_csv('C:/MultiHazard/Data/emdat/trial/v2_isolate_730days_200km_1535eve.csv', header = 0)
for index, row in hazard_list.iterrows():
    if index > 1534:
        year = row['year']
        pair_id = row['pair_id']
        lat = row['latitude']
        lon = row['longitude']
        get_constant_matrix(lat, lon, pair_id, year)
    # match using the coordinates + time 
    # from the 1st event!
    #if year > 2009:#< 2001
    

    
    
    #if index >10:
    #    break
    #print("Event pair {} is matched".format(count))

#%% Get the list of right path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

hazard_list0 = pd.read_excel('C:/MultiHazard/Data/emdat/trial/_pos_neg_match_30km_3195pairs.xlsx',header=0)
hazard_list = hazard_list0.values.tolist()
for row in hazard_list.iterrows():
    year = row['Start Year']
    month = row['Start Month']
    day = row['Start Day']
    pair_id = row['pair_id']
    break
    
    
    

#%%
path = 'C:/test/'
os.chdir(path)
file_list = os.listdir(path)

#%%
#for file in file_list:
lyr_dict = {}


file_name=file[:-3]
lyr_dict[file_name] = [QgsRasterLayer(file), file_name+'@366']
#%%
file = "C:/MultiHazard/Data/slope/slope_1KMmd_GMTEDmd.tif"
output = 'C:/MultiHazard/Data/processed/event3.tif'
entries = []
# PROBLEM: I cannot get one specific band
#for lyr in lyr_dict:

ras = QgsRasterCalculatorEntry()
ras.ref = 'e3@1'
ras.raster = QgsRasterLayer(file)
ras.bandNumber = 1
entries.append(ras)
#index+=1
    # https://nasirlukman.github.io/2021-08-08-Raster-Calculator/
#calc = QgsRasterCalculator('MAX( MAX ( "GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg@1","GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg@2"),"GPCC_total_precipitation_day_1x1_global_1988_v2020.0-v6.0-fg@3")', 
#                           output, 'GTiff', topo.extent(), int(topo.width()/10), int(topo.height()/10), entries)
layer = QgsRasterLayer(file)

col_num = 43200
row_num = int(43200/layer.width()*layer.height())
#QgsCoordinateReferenceSystem("EPSG:4326")
calc = QgsRasterCalculator("e3@1", output, 'GTiff',layer.extent(),col_num, row_num, entries)
calc.processCalculation()
# Success = 0 , CreateOutputError = 1 , InputLayerError = 2 , Canceled = 3 ,
# ParserError = 4 , MemoryError = 5 , BandError = 6 , CalculationError = 7



#%%
import numpy as np
import PIL
from PIL import Image

img1 = Image.open('C:/MultiHazard/Data/processed/friday001.tif')
#img1.show()
img1_arr = np.array(img1)

#%%
img2 = Image.open('C:/MultiHazard/Data/processed/friday002.tif')
img2.show()
img2_arr = np.array(img1)
    
