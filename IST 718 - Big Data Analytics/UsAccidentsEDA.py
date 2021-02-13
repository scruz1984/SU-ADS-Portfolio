# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:32:51 2020

@author: thecr
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform
import statsmodels.formula.api as smf

import plotly.express as px
from plotly.offline import plot

import gmaps
import gmaps.datasets
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

import re

#Read csv file
accidents = pd.read_csv("US_accidents_June20.csv")

accidents['Wind_Speed(mph)'] = accidents['Wind_Speed(mph)'].fillna(0)

# Friendlier Names
cleanAccidents = accidents
accidents = accidents.loc[accidents['Start_Time'].str.contains("2018")]

#test = accidents

accidents['WindSpeed'] = accidents['Wind_Speed(mph)']
accidents['Pressure'] = accidents['Pressure(in)']
accidents['WindDirection'] = accidents['Wind_Direction']
accidents['Visibility'] = accidents['Visibility(mi)']
accidents['Temperature'] = accidents['Temperature(F)']
accidents['WindChill'] = accidents['Wind_Chill(F)']
accidents['Distance'] = accidents['Distance(mi)']

accidents = accidents.drop(['ID', 'Source', 'TMC','End_Lat',
                                'End_Lng', 'Description',
                                'Number', 'Street', 'Zipcode','Airport_Code',
                                'Weather_Timestamp','Amenity',
                                'Bump',
                                'Junction','No_Exit', 'Railway',
                                'Turning_Loop', 'Start_Time','End_Time',
                                'Start_Lat','Start_Lng','Distance(mi)',
                                'Side','County','Country','Timezone',
                                'Temperature(F)','Wind_Chill(F)','Humidity(%)',
                                'Pressure(in)','Visibility(mi)','Wind_Direction',
                                'Wind_Speed(mph)','Precipitation(in)',
                                'Weather_Condition','Crossing','Give_Way',
                                'Roundabout','Station','Stop',
                                'Traffic_Calming','Traffic_Signal',
                                'Sunrise_Sunset','Civil_Twilight',
                                'Nautical_Twilight','Astronomical_Twilight'], axis=1)

# Correlation Chart
corr = accidents.corr()

mask = np.zeros_like(corr, dtype=np.int)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220,10,as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})

#Regression Model
laAccidents = accidents

#print("TEST")
#print(laAccidents.columns)
#print("DONE")

laAccidents = laAccidents.loc[laAccidents['State'] == 'TX']
laAccidents = laAccidents.loc[laAccidents['City'] == 'Amarillo']

np.random.seed(1234)
laAccidents['runiform'] = uniform.rvs(loc=0, scale=1, size=len(laAccidents))
laAccidents_train = laAccidents[laAccidents['runiform'] >= .33]
laAccidents_test = laAccidents[laAccidents['runiform'] < .33]

#Specify a model
laAccidents_model = str('Severity ~ Visibility + WindSpeed + Pressure + WindDirection + Temperature + WindChill')

#Fit the model
laAccidents_model_fit = smf.ols(laAccidents_model, data=laAccidents).fit()
print(laAccidents_model_fit.summary())

# Create the diagram
fig = px.bar_polar(laAccidents, r="Severity", theta="WindDirection", color="WindSpeed", template="plotly_dark", title="Amarillo, TX",
            color_discrete_sequence= px.colors.sequential.Plasma_r)
fig.show()

plot(fig)

cleanAccidents['Start_Time'] = pd.to_datetime(cleanAccidents['Start_Time'])
cleanAccidents['year'] = pd.DatetimeIndex(cleanAccidents['Start_Time']).year
cleanAccidents['month'] = pd.DatetimeIndex(cleanAccidents['Start_Time']).month

accidents2018 = cleanAccidents.loc[cleanAccidents['year'] == 2018]
accidents2018 = accidents2018.loc[accidents2018['State'] == 'MN']

jan2018Accidents = accidents2018.loc[accidents2018['month'] == 1]
feb2018Accidents = accidents2018.loc[accidents2018['month'] == 2]
mar2018Accidents = accidents2018.loc[accidents2018['month'] == 3]
apr2018Accidents = accidents2018.loc[accidents2018['month'] == 4]
may2018Accidents = accidents2018.loc[accidents2018['month'] == 5]
jun2018Accidents = accidents2018.loc[accidents2018['month'] == 6]
jul2018Accidents = accidents2018.loc[accidents2018['month'] == 7]
aug2018Accidents = accidents2018.loc[accidents2018['month'] == 8]
sep2018Accidents = accidents2018.loc[accidents2018['month'] == 9]
oct2018Accidents = accidents2018.loc[accidents2018['month'] == 10]
nov2018Accidents = accidents2018.loc[accidents2018['month'] == 11]
dec2018Accidents = accidents2018.loc[accidents2018['month'] == 12]

accidents2018 = pd.DataFrame(columns = ['Month', 'Accidents'])
accidents2018 = accidents2018.append({'Month': 'Jan', 'Accidents': len(jan2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Feb', 'Accidents': len(feb2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Mar', 'Accidents': len(mar2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Apr', 'Accidents': len(apr2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'May', 'Accidents': len(may2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Jun', 'Accidents': len(jun2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Jul', 'Accidents': len(jul2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Aug', 'Accidents': len(aug2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Sep', 'Accidents': len(sep2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Oct', 'Accidents': len(oct2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Nov', 'Accidents': len(nov2018Accidents)}, ignore_index=True)
accidents2018 = accidents2018.append({'Month': 'Dec', 'Accidents': len(dec2018Accidents)}, ignore_index=True)
print(accidents2018)

ax = accidents2018.plot.bar(x='Month',y='Accidents', rot=0)
