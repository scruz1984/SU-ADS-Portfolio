# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:17:43 2020

@author: thecr
"""

import pandas as pd
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

import seaborn as sns

# Load initial data
zillow = seriesDF = pd.read_csv('ZillowBaseData.csv')

# Create a new dataframe with just the zip code and base value (latest home value)
zillowNew = pd.DataFrame()
zillowNew['zipcode'] = zillow['RegionName']
zillowNew['base_value'] = zillow['3/31/2020']

# Load Zip Code 5 year projections
firstTen = pd.read_csv('firstTen.csv')
secondTen = pd.read_csv('secondTen.csv')
thirdTen = pd.read_csv('thirdTen.csv')
fourthTen = pd.read_csv('fourthTen.csv')
fifthTen = pd.read_csv('fifthTen.csv')
sixthTen = pd.read_csv('sixthTen.csv')
seventhTen = pd.read_csv('seventhTen.csv')
eighthTen = pd.read_csv('eigthTen.csv')
ninthTen = pd.read_csv('ninthTen.csv')

frames = [firstTen, secondTen, thirdTen, fourthTen, fifthTen, sixthTen, seventhTen, eighthTen, ninthTen]

# Concatenante the dataframes
concatFrames = pd.concat(frames)

# Load supplemental employment data
supplementalData = pd.read_csv('zbp18totals.csv')
supplementalData = supplementalData.drop(columns = ['name', 'emp_nf', 'emp', 'qp1_nf', 'qp1',
                                                    'ap_nf','est','city','stabbr', 'cty_name'])

# Merge data with supplemental data
mergeStepOne = pd.merge(concatFrames, supplementalData, on=['zipcode'])

# Merge remainder of data
mergedData = pd.merge(mergeStepOne, zillowNew, on=['zipcode'])

# Calculate new fields
mergedData['adj_value'] = mergedData['proj_value'] - mergedData['meanAbsError']
mergedData['inc_in_val'] = mergedData['adj_value'] - mergedData['base_value']

# Plot no labels
x = mergedData['ap']
y = mergedData['inc_in_val']
labels = mergedData['zipcode']

plt.scatter(mergedData['ap'], mergedData['proj_value'])
plt.title("Zip Code Investment")
plt.xlabel("Annual Payroll")
plt.ylabel("Projected Increase In Home Value")


# Plot with labels
fig, ax = plt.subplots(1, figsize=(10,6))
fig.suptitle('Zip Code Investment')

ax.scatter(mergedData['ap'], mergedData['inc_in_val'])

for x_pos, y_pos, label in zip(x,y,labels):
    ax.annotate(label,             # The label for this point
                xy=(x_pos, y_pos), # Position of the corresponding point
                xytext=(7, 0),     # Offset text by 7 points to the right
                textcoords='offset points', # tell it to use offset points
                ha='left',         # Horizontally aligned to the left
                va='center')       # Vertical alignment is centered