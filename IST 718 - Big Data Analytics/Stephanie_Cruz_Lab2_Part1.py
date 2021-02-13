# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:26:11 2020

@author: thecr
"""

import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt

import timeit
from fbprophet import Prophet

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from math import sqrt
from pandas import DataFrame
from scipy.stats import boxcox
import itertools

seriesDF = pd.read_csv('ZillowBaseData.csv')
#print(series.describe())

#=========================================================================================

hotSpringsDF = seriesDF[seriesDF['Metro'] == 'Hot Springs']
hotSpringsDF = hotSpringsDF.drop(columns = ['RegionID', 'SizeRank', 'RegionType',
                                            'StateName','State','City','Metro',
                                            'CountyName','RegionName'])

hot_springs_mean = hotSpringsDF.mean()
hot_springs_mean.plot()
#plt.suptitle("Hot Springs")


#==========================================================================================

littleRockDF = seriesDF[seriesDF['Metro'] == 'Little Rock-North Little Rock-Conway']
littleRockDF = littleRockDF.drop(columns = ['RegionID', 'SizeRank', 'RegionType',
                                            'StateName','State','City','Metro',
                                            'CountyName','RegionName'])

little_rock_mean = littleRockDF.mean() 

little_rock_mean.plot()
#plt.suptitle("Little Rock")
 
#=========================================================================================

fayettevilleDF = seriesDF[seriesDF['Metro'] == 'Fayetteville']
fayettevilleDF = fayettevilleDF.drop(columns = ['RegionID', 'SizeRank', 'RegionType',
                                            'StateName','State','City','Metro',
                                            'CountyName','RegionName'])

fayetteville_mean = fayettevilleDF.mean() 

fayetteville_mean.plot()
#plt.suptitle("Fayetteville")

#=========================================================================================
searcyDF = seriesDF[seriesDF['Metro'] == 'Searcy']
searcyDF = searcyDF.drop(columns = ['RegionID', 'SizeRank', 'RegionType',
                                            'StateName','State','City','Metro',
                                            'CountyName','RegionName'])

searcy_mean = searcyDF.mean() 

searcy_mean.plot()
#plt.suptitle("Searcy")
#=======================================================================================