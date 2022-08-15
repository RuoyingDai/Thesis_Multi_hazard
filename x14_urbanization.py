# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:36:40 2022

@author: rdi420
"""

import pandas as pd
import numpy as np


df = pd.read_csv("C:/MultiHazard/Data/urbanization1975_2015/rate_ruoying.csv")

co =np.corrcoef(df['normal area rate'].tolist(), df['normal pop rate'].tolist())
# the pearson correlation coefficient is 0.3193
print(co)
