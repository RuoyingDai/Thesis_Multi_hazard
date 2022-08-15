# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 20:29:37 2022

@author: rdi420
"""
path = 'C:/MultiHazard/Data/population_GHS/GHS_POP_E1975_GLOBE_R2019A_4326_9ss_V1_0/GHS_POP_E1975_GLOBE_R2019A_4326_9ss_V1_0.tif'
#from PIL import Image
#im = Image.open()
#im.show()

import matplotlib.pyplot as plt
I = plt.imread(path)

import numpy
imarray = numpy.array(I)