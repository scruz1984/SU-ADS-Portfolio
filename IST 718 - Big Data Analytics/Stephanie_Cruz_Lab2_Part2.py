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

def projectValue(zipFrames, filename):
    results = pd.DataFrame()
    for k,v in zipFrames.items():
        df = zipFrames[k]
        df = df.reset_index()
        df.columns = ['ds','y']
        prop_model = Prophet(interval_width=.095)
        prop_model.fit(df)
        future_dates = prop_model.make_future_dataframe(periods = 60, freq='M')
        forecast = prop_model.predict(future_dates)
        metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        tempDF = {'zipcode':k, 'proj_value': forecast['yhat'].iloc[-1], 
                  'r2':r2_score(metric_df.y, metric_df.yhat),
                  'meanAbsError':mean_absolute_error(metric_df.y, metric_df.yhat)}
        results = results.append(tempDF, ignore_index=True)
    
    results.to_csv(filename, index=False)

# Load data
seriesDF = pd.read_csv('ZillowBaseData.csv')
part2Series = seriesDF

# Drop columns we don't need
part2Series = part2Series.drop(columns = ['RegionID', 'SizeRank', 'RegionType',
                                            'StateName','State','City','Metro',
                                            'CountyName'])

# Transpose the data
transposed = part2Series.transpose()

zipFrames = {}

# Create a dataframe for each zip code
for(columnName, columnData) in transposed.iteritems():
    zipFrames[columnData[0]] = columnData[1:]


# Process the data, 10 percent at a time

#Entries first 10%
#firstTen = dict(list(zipFrames.items())[0:3046])
#projectValue(firstTen, 'firstTen.csv')

#Entries 10-20%
#secondTen = dict(list(zipFrames.items())[3047:6092])
#projectValue(secondTen, 'secondTen.csv')

#Entries 20-30%
#thirdTen = dict(list(zipFrames.items())[6093:9168])
#projectValue(thirdTen, 'thirdTen.csv')

#Entries 30-40%
#fourthTen = dict(list(zipFrames.items())[9169:12184])
#projectValue(fourthTen, 'fourthTen.csv')

#Entries 40-50%
#fifthTen = dict(list(zipFrames.items())[12185:15230])
#projectValue(fifthTen, 'fifthTen.csv')

#Entries 50-60%
#sixthTen = dict(list(zipFrames.items())[15230:18276])
#projectValue(sixthTen, 'sixthTen.csv')

#Entries 60-70%
#seventhTen = dict(list(zipFrames.items())[18277:21322])
#projectValue(seventhTen, 'seventhTen.csv')

#Entries 70-80%
#eigthTen = dict(list(zipFrames.items())[21323:24368])
#projectValue(eigthTen, 'eigthTen.csv')

#Entries 80-90%
#ninthTen = dict(list(zipFrames.items())[24369:27414])
# projectValue(ninthTen, 'ninthTen.csv')

#Entries 90-100%
tenthTen = dict(list(zipFrames.items())[27415:])
projectValue(tenthTen, 'tenthTen.csv')
