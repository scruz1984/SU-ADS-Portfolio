# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 12:13:12 2020

@author: thecr
"""

import pandas
import numpy
import csv
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm


#Read our data from csv
electionInfoData = pandas.read_csv('election_info.csv', header='infer', nrows=436)

gunViolenceData = pandas.read_csv('gun_violence_data.csv', header='infer', nrows=239678)

stateFirearmLawData = pandas.read_csv('state_firearms.csv', header='infer', nrows=1351)

#Trim our data to only include the columns we want
electionInfoData = electionInfoData[['District', 'rep_party_2012', 'winning_party_2012', 'rep_party_2014',
                                     'winning_party_2014', 'rep_party_2016', 'winning_party_2016', 'rep_party_2018',
                                     'winning_party_2018','white','black','latino','asian and pacific island',
                                     'native','other','bach degree or higher among 25up','whiteBA','median_income',
                                     'noncollege_white', 'CITYLAB_CDI']]

gunViolenceData = gunViolenceData[['date','state','n_killed','n_injured','congressional_district',
                                   'gun_type','incident_characteristics','participant_age_group','participant_gender',
                                   'participant_status','participant_type']]

#Drop data that doesn't include the congressional district
gunViolenceData = gunViolenceData[gunViolenceData['congressional_district'] != 0]
gunViolenceData = gunViolenceData.dropna(subset=['congressional_district'])

#split gunViolenceData into 4 data frames based on the year
gunViolenceData2014 = gunViolenceData[gunViolenceData['date'].str.contains('2014')]
gunViolenceData2015 = gunViolenceData[gunViolenceData['date'].str.contains('2015')]
gunViolenceData2016 = gunViolenceData[gunViolenceData['date'].str.contains('2016')]
gunViolenceData2017 = gunViolenceData[gunViolenceData['date'].str.contains('2017')]

#Trim state firearm law data to only include the columns we want
stateFirearmLawData = stateFirearmLawData[['state', 'year','lawtotal']]

#Remove years from state firearm law data that we aren't looking at for this project
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1991].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1992].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1993].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1994].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1995].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1996].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1997].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1998].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==1999].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2000].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2001].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2002].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2003].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2004].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2005].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2006].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2007].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2008].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2009].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2010].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2011].index
stateFirearmLawData.drop(indexNames, inplace=True)
indexNames = stateFirearmLawData[stateFirearmLawData['year']==2012].index
stateFirearmLawData.drop(indexNames, inplace=True)

#Define column names for aggregated gun violence data
columnNames = ['state','district','numIncidents','numKilled','numInjured']

#Create aggregated gun violence dataframes for each year that we are looking at
aggregated2014gunViolenceData = pandas.DataFrame(columns = columnNames)
aggregated2015gunViolenceData = pandas.DataFrame(columns = columnNames)
aggregated2016gunViolenceData = pandas.DataFrame(columns = columnNames)
aggregated2017gunViolenceData = pandas.DataFrame(columns = columnNames)

#Group our gun violence data by state and congressional district
grouped2014gunViolenceData = gunViolenceData2014.groupby(['state', 'congressional_district'])
grouped2015gunViolenceData = gunViolenceData2015.groupby(['state', 'congressional_district'])
grouped2016gunViolenceData = gunViolenceData2016.groupby(['state', 'congressional_district'])
grouped2017gunViolenceData = gunViolenceData2017.groupby(['state', 'congressional_district'])

#TODO Wordcloud -- see below
#for name, group in grouped2014gunViolenceData:
#    for i in group['incident_characteristcs']:
#        wordcloud = WordCloud().generate(i)
#        plt.imshaw(wordcloud,interpolation='bilinear')

#Loop through our grouped gun violence data for 2014

incidentCharString2014 = ""

for name, group in grouped2014gunViolenceData:
    state = name[0]
    
    district = name[1]
    numIncidents = group['n_killed'].count()
    numKilled = group['n_killed'].sum()
    numInjured = group['n_injured'].sum()
    
    massShooting = 0
    officerInvolvedIncident = 0
    domesticViolence = 0
    murderSuicide = 0
    spreeShooting = 0
    schoolIncident = 0
    
    #Look for key words in our incident characteristics column
    for i in group['incident_characteristics']:
        try:
            incidentCharString2014 = incidentCharString2014 + i
            if(i.find('Mass Shooting') > -1):
                massShooting +=1
            if(i.find('Officer Involved Incident') > -1):
                officerInvolvedIncident +=1
            if(i.find('Domestic Violence') > -1):
                domesticViolence +=1
            if(i.find('Murder/Suicide') > -1):
                murderSuicide +=1
            if(i.find('Spree Shooting') > -1):
                spreeShooting +=1
            if(i.find('School Incident') > -1):
                schoolIncident +=1
        except AttributeError:
            print('error')
        except TypeError:
            print('error')
        #Do this outside, concatenate and then create the wordcloud
        #wordcloud = WordCloud().generate(i)
        #plt.imshow(wordcloud,interpolation='bilinear')

    #create an array of our data         
    data = [[state, district, numIncidents, numKilled, numInjured, massShooting, officerInvolvedIncident, domesticViolence, murderSuicide, spreeShooting, schoolIncident]]
    
    #Create a temp dataframe
    tempDf = pandas.DataFrame(data, columns = ['state','district','numIncidents','numKilled','numInjured', 'mass', 'officer', 'domestic','ms','spree','school'])
    
    #Append it to our existing dataframe
    aggregated2014gunViolenceData = aggregated2014gunViolenceData.append(tempDf, True)
    
#print(incidentCharString2014)
#wordcloud = WordCloud().generate(incidentCharString2014)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.show()
    
#Loop through our grouped gun violence data for 2015 
for name, group in grouped2015gunViolenceData:
    state = name[0]
    
    district = name[1]
    numIncidents = group['n_killed'].count()
    numKilled = group['n_killed'].sum()
    numInjured = group['n_injured'].sum()
    
    massShooting = 0
    officerInvolvedIncident = 0
    domesticViolence = 0
    murderSuicide = 0
    spreeShooting = 0
    schoolIncident = 0
    
    #Look for key words in our incident characteristics column
    for i in group['incident_characteristics']:
        try:
            if(i.find('Mass Shooting') > -1):
                massShooting +=1
            if(i.find('Officer Involved Incident') > -1):
                officerInvolvedIncident +=1
            if(i.find('Domestic Violence') > -1):
                domesticViolence +=1
            if(i.find('Murder/Suicide') > -1):
                murderSuicide +=1
            if(i.find('Spree Shooting') > -1):
                spreeShooting +=1
            if(i.find('School Incident') > -1):
                schoolIncident +=1
        except AttributeError:
            print('error')
        
    #create an array of our data
    data = [[state, district, numIncidents, numKilled, numInjured, massShooting, officerInvolvedIncident, domesticViolence, murderSuicide, spreeShooting, schoolIncident]]
    
    #Create a temp dataframe
    tempDf = pandas.DataFrame(data, columns = ['state','district','numIncidents','numKilled','numInjured', 'mass', 'officer', 'domestic','ms','spree','school'])
    
    #Append it to our existing dataframe
    aggregated2015gunViolenceData = aggregated2015gunViolenceData.append(tempDf, True)
    
#Loop through our grouped gun violence data for 2016    
for name, group in grouped2016gunViolenceData:
    state = name[0]
    
    district = name[1]
    numIncidents = group['n_killed'].count()
    numKilled = group['n_killed'].sum()
    numInjured = group['n_injured'].sum()
    
    massShooting = 0
    officerInvolvedIncident = 0
    domesticViolence = 0
    murderSuicide = 0
    spreeShooting = 0
    schoolIncident = 0
    
    #Look for key words in our incident characteristics column
    for i in group['incident_characteristics']:
        try:
            if(i.find('Mass Shooting') > -1):
                massShooting +=1
            if(i.find('Officer Involved Incident') > -1):
                officerInvolvedIncident +=1
            if(i.find('Domestic Violence') > -1):
                domesticViolence +=1
            if(i.find('Murder/Suicide') > -1):
                murderSuicide +=1
            if(i.find('Spree Shooting') > -1):
                spreeShooting +=1
            if(i.find('School Incident') > -1):
                schoolIncident +=1
        except AttributeError:
            print('error')
      
    #create an array of our data
    data = [[state, district, numIncidents, numKilled, numInjured, massShooting, officerInvolvedIncident, domesticViolence, murderSuicide, spreeShooting, schoolIncident]]
    
    #Create a temp dataframe
    tempDf = pandas.DataFrame(data, columns = ['state','district','numIncidents','numKilled','numInjured', 'mass', 'officer', 'domestic','ms','spree','school'])
    
    #Append it to our existing dataframe
    aggregated2016gunViolenceData = aggregated2016gunViolenceData.append(tempDf, True)

#Loop through our grouped gun violence data for 2017    
for name, group in grouped2017gunViolenceData:
    state = name[0]
    
    district = name[1]
    numIncidents = group['n_killed'].count()
    numKilled = group['n_killed'].sum()
    numInjured = group['n_injured'].sum()
    
    massShooting = 0
    officerInvolvedIncident = 0
    domesticViolence = 0
    murderSuicide = 0
    spreeShooting = 0
    schoolIncident = 0
    
    #Look for key words in our incident characteristics column
    for i in group['incident_characteristics']:
        try:
            if(i.find('Mass Shooting') > -1):
                massShooting +=1
            if(i.find('Officer Involved Incident') > -1):
                officerInvolvedIncident +=1
            if(i.find('Domestic Violence') > -1):
                domesticViolence +=1
            if(i.find('Murder/Suicide') > -1):
                murderSuicide +=1
            if(i.find('Spree Shooting') > -1):
                spreeShooting +=1
            if(i.find('School Incident') > -1):
                schoolIncident +=1
        except AttributeError:
            print('error')
      
    #create an array of our data
    data = [[state, district, numIncidents, numKilled, numInjured, massShooting, officerInvolvedIncident, domesticViolence, murderSuicide, spreeShooting, schoolIncident]]
    
    #Create a temp dataframe
    tempDf = pandas.DataFrame(data, columns = ['state','district','numIncidents','numKilled','numInjured', 'mass', 'officer', 'domestic','ms','spree','school'])
    
    #Append it to our existing dataframe
    aggregated2017gunViolenceData = aggregated2017gunViolenceData.append(tempDf, True)
   
#Print our reports of aggregated gun violence data for each year from 2014-2017
aggregated2014gunViolenceData.to_csv("2014AggregatedGunViolenceData.csv")
aggregated2015gunViolenceData.to_csv("2015AggregatedGunViolenceData.csv")
aggregated2016gunViolenceData.to_csv("2016AggregatedGunViolenceData.csv")
aggregated2017gunViolenceData.to_csv("2017AggregatedGunViolenceData.csv")

#Convert column numIncidents to be an integer
aggregated2014gunViolenceData['mass'] = aggregated2014gunViolenceData['mass'].astype(int)

#Yearly Breakdown Plots
aggregated2014gunViolenceData.plot.box(figsize=(15,12)) 
plt.title("2014 Gun Violence Overview")
aggregated2015gunViolenceData.plot.box(figsize=(15,12))  
plt.title("2015 Gun Violence Overview")
aggregated2016gunViolenceData.plot.box(figsize=(15,12)) 
plt.title("2016 Gun Violence Overview") 
aggregated2017gunViolenceData.plot.box(figsize=(15,12))  
plt.title("2017 Gun Violence Overview")

#Histogram Plots
#aggregated2014gunViolenceData.hist(column='mass')
#plt.title("2014 Mass Shooting Breakdown")
#aggregated2015gunViolenceData.hist(column='mass')
#plt.title("2015 Mass Shooting Breakdown")
#aggregated2016gunViolenceData.hist(column='mass')
#plt.title("2016 Mass Shooting Breakdown")
#aggregated2017gunViolenceData.hist(column='mass')
#plt.title("2017 Mass Shooting Breakdown")

#aggregated2014gunViolenceData.hist(column='officer')
#plt.title("2014 Officer Involved Shooting Breakdown")
#aggregated2015gunViolenceData.hist(column='officer')
#plt.title("2015 Officer Involved Shooting Breakdown")
#aggregated2016gunViolenceData.hist(column='officer')
#plt.title("2016 Officer Involved Shooting Breakdown")
#aggregated2017gunViolenceData.hist(column='officer')
#plt.title("2017 Officer Involved Shooting Breakdown")

#aggregated2014gunViolenceData.hist(column='domestic')
#plt.title("2014 Domestic Violence Shooting Breakdown")
#aggregated2015gunViolenceData.hist(column='domestic')
#plt.title("2015 Domestic Violence Shooting Breakdown")
#aggregated2016gunViolenceData.hist(column='domestic')
#plt.title("2016 Domestic Violence Shooting Breakdown")
#aggregated2017gunViolenceData.hist(column='domestic')
#plt.title("2017 Domestic Violence Shooting Breakdown")

#aggregated2014gunViolenceData.hist(column='school')
#plt.title("2014 School Shooting Breakdown")
#aggregated2015gunViolenceData.hist(column='school')
#plt.title("2015 School Shooting Breakdown")
#aggregated2016gunViolenceData.hist(column='school')
#plt.title("2016 School Shooting Breakdown")
#aggregated2017gunViolenceData.hist(column='school')
#plt.title("2017 School Shooting Breakdown")

#Function to map state codes to state name so that we can merge our data sets
def extractStateFromCode(code):
    switcher = {
        "AL":"Alabama",
        "AK":"Alaska",
        "AZ":"Arizona",
        "AR":"Arkansas",
        "CA":"California",
        "CO":"Colorado",
        "CT":"Connecticut",
        "DE":"Deleware",
        "FL":"Florida",
        "GA":"Georgia",
        "HI":"Hawaii",
        "ID":"Idaho",
        "IL":"Illinois",
        "IN":"Indiana",
        "IA":"Iowa",
        "KS":"Kansas",
        "KY":"Kentucky",
        "LA":"Loisiana",
        "ME":"Maine",
        "MD":"Maryland",
        "MA":"Massachusetts",
        "MI":"Michigan",
        "MN":"Minnesota",
        "MS":"Mississippi",
        "MO":"Missouri",
        "MT":"Montana",
        "NE":"Nebraska",
        "NV":"Nevada",
        "NH":"New Hampshire",
        "NJ":"New Jersey",
        "NM":"New Mexico",
        "NY":"New York",
        "NC":"North Carolina",
        "ND":"North Dakota",
        "OH":"Ohio",
        "OK":"Oklahoma",
        "OR":"Oregon",
        "PA":"Pennsylvania",
        "RI":"Rhode Island",
        "SC":"South Carolina",
        "SD":"South Dakota",
        "TN":"Tennessee",
        "TX":"Texas",
        "UT":"Utah",
        "VT":"Vermont",
        "VA":"Virginia",
        "WA":"Washington",
        "WV":"West Virginia",
        "WI":"Wisconsin",
        "WY":"Wyoming"
    }
    return switcher.get(code, "Unk")

#Function to extract the district fomr our code
def extractDistrictFromCode(code):
    if(code == 'AL'):
        return "1"
    else:
        return code
    
#electionInfoData.info()

#Function to determine if a congressional seat changed political parties
def determineIfSeatChangedParties(partyA, partyB):
    if(partyA == 'Open Post-Redistrict'):
        return "false"
    elif(partyB == 'NA'):
        return "false"
    elif(partyA == 'Open-Used to be Dem' and partyB == 'D'):
        return "false"
    elif(partyA == 'Open-Used to be GOP' and partyB == 'R'):
        return "false"
    elif(partyA == 'GOP' and partyB == 'R'):
        return "false"
    elif(partyA == 'DEM' and partyB == 'D'):
        return "false"
    elif(partyA == partyB):
        return "false"
    else:
        return "true"
        
#Funciton to determine the party if it's not a "swing" party
def calculatePartyIfNotSwing(repParty2014, winParty2014):
    if(repParty2014 == "Open-Used to be Dem" or repParty2014 == 'Open-Used to be GOP'):
        return winParty2014
    else:
        return repParty2014
 
#Define column names for our data frame to map the districts to their political parties
columnNames = ['state','district','party']

#Create our new dataframe
congressionalDistrictPartyData = pandas.DataFrame(columns = columnNames)

#Loop through our election info data
for index, row in electionInfoData.iterrows():
    
    #Extract our state and district
    state = extractStateFromCode(row['District'][0:2])
    district = extractDistrictFromCode(row['District'][2:])
    
    #Determine if it's a "Swing" district
    swing = determineIfSeatChangedParties(row['rep_party_2014'], row['winning_party_2014'])
    if(swing == 'false'):
        swing = determineIfSeatChangedParties(row['rep_party_2016'], row['winning_party_2016'])
    if(swing == 'false'):
        swing = determineIfSeatChangedParties(row['rep_party_2018'], row['winning_party_2018'])
    #Calculate the political party affiliated with this district
    party = ""
    if(swing == "true"):
        party = "Swing"
    else:
        party = calculatePartyIfNotSwing(row['rep_party_2014'], row['winning_party_2014'])
     
    #Create an array of our data
    electionData = [[state, district, party]]
    
    #Create a temporary dataframe
    tempElectionDf = pandas.DataFrame(electionData, columns=['state','district','party'])
    
    #Append it to our dataframe
    congressionalDistrictPartyData = congressionalDistrictPartyData.append(tempElectionDf, True)

#Write our our congressional district party data report
congressionalDistrictPartyData.to_csv("congressionalDistrictPartyData.csv")

#Convet numIncidents to numeric
aggregated2014gunViolenceData['numIncidents'] = pandas.to_numeric(aggregated2014gunViolenceData['numIncidents'])

#Calculate the total number of incidents by state for 2014
total2014IncidentsByState = aggregated2014gunViolenceData.groupby('state').sum()
total2014IncidentsByState = total2014IncidentsByState.reset_index()
total2014IncidentsByState = total2014IncidentsByState[['state','numIncidents']]
total2014IncidentsByState.columns = ['state','2014 Incidents']

#Calculate the total number of incidents by state for 2015
aggregated2015gunViolenceData['numIncidents'] = pandas.to_numeric(aggregated2015gunViolenceData['numIncidents'])
total2015IncidentsByState = aggregated2015gunViolenceData.groupby('state').sum()
total2015IncidentsByState = total2015IncidentsByState.reset_index()
total2015IncidentsByState = total2015IncidentsByState[['state','numIncidents']]
total2015IncidentsByState.columns = ['state','2015 Incidents']

#Calculate the total number of incidents by state for 2016
aggregated2016gunViolenceData['numIncidents'] = pandas.to_numeric(aggregated2016gunViolenceData['numIncidents'])
total2016IncidentsByState = aggregated2016gunViolenceData.groupby('state').sum()
total2016IncidentsByState = total2016IncidentsByState.reset_index()
total2016IncidentsByState = total2016IncidentsByState[['state','numIncidents']]
total2016IncidentsByState.columns = ['state','2016 Incidents']

#Calculate the total number of incidents by state for 2017
aggregated2017gunViolenceData['numIncidents'] = pandas.to_numeric(aggregated2017gunViolenceData['numIncidents'])
total2017IncidentsByState = aggregated2017gunViolenceData.groupby('state').sum()
total2017IncidentsByState = total2017IncidentsByState.reset_index()
total2017IncidentsByState = total2017IncidentsByState[['state','numIncidents']]
total2017IncidentsByState.columns = ['state','2017 Incidents']

#Create four dataframes to hold the state firearms law data for each year from 2014-2017
stateLawData2014 = stateFirearmLawData[stateFirearmLawData['year'] == 2014]
stateLawData2015 = stateFirearmLawData[stateFirearmLawData['year'] == 2015]
stateLawData2016 = stateFirearmLawData[stateFirearmLawData['year'] == 2016]
stateLawData2017 = stateFirearmLawData[stateFirearmLawData['year'] == 2017]

#Rename column names to be more descriptive
stateLawData2014 = stateLawData2014[['state','lawtotal']]
stateLawData2014.columns = ['state','lawtotal2014']

stateLawData2015 = stateLawData2015[['state','lawtotal']]
stateLawData2015.columns = ['state','lawtotal2015']

stateLawData2016 = stateLawData2016[['state','lawtotal']]
stateLawData2016.columns = ['state','lawtotal2016']

stateLawData2017 = stateLawData2017[['state','lawtotal']]
stateLawData2017.columns = ['state','lawtotal2017']

#Merge our incident dataframes and statelaw dataframes
mergedIncidentDataByState = pandas.merge(total2014IncidentsByState, stateLawData2014, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, total2015IncidentsByState, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, stateLawData2015, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, total2016IncidentsByState, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, stateLawData2016, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, total2017IncidentsByState, on='state',how='inner')
mergedIncidentDataByState = pandas.merge(mergedIncidentDataByState, stateLawData2017, on='state',how='inner')

mergedIncidentWithoutLawInformation = mergedIncidentDataByState[['state','2014 Incidents','2015 Incidents','2016 Incidents', '2017 Incidents']]

#Write out our mergedincidents by state report
mergedIncidentDataByState.to_csv("violenceCountsByStateAndYear.csv")
mergedIncidentWithoutLawInformation.to_csv("violenceCountsByStateAndYearWithoutLawInfo.csv")

#Convert district column to float type
congressionalDistrictPartyData['district'] = congressionalDistrictPartyData['district'].astype(float) 

#Merge our gun violence data and congressional party data for 2014-2017
mergedIncidentsByCongressionalDistricts2014 = pandas.merge(aggregated2014gunViolenceData, congressionalDistrictPartyData, on=['state','district'], how='inner')
mergedIncidentsByCongressionalDistricts2015 = pandas.merge(aggregated2015gunViolenceData, congressionalDistrictPartyData, on=['state','district'], how='inner')
mergedIncidentsByCongressionalDistricts2016 = pandas.merge(aggregated2016gunViolenceData, congressionalDistrictPartyData, on=['state','district'], how='inner')
mergedIncidentsByCongressionalDistricts2017 = pandas.merge(aggregated2017gunViolenceData, congressionalDistrictPartyData, on=['state','district'], how='inner')

#Write out our reports of merged incidents by congressional districts
mergedIncidentsByCongressionalDistricts2014.to_csv("violenceIncidentsByCongressionalDistrictType2014.csv")
mergedIncidentsByCongressionalDistricts2015.to_csv("violenceIncidentsByCongressionalDistrictType2015.csv")
mergedIncidentsByCongressionalDistricts2016.to_csv("violenceIncidentsByCongressionalDistrictType2016.csv")
mergedIncidentsByCongressionalDistricts2017.to_csv("violenceIncidentsByCongressionalDistrictType2017.csv")

groupedAndMergedIncidentsByCongressionalDistricts2014 = mergedIncidentsByCongressionalDistricts2014.groupby('party')

#Calculate the average number of firearm laws between 2014-2017 per state
averageStateFirearmLawData = stateFirearmLawData.groupby('state').mean()
averageStateFirearmLawData = averageStateFirearmLawData.reset_index()
averageStateFirearmLawData = averageStateFirearmLawData[['state','lawtotal']]

#Count the number of rep, dem and swing seats per state
republicanRepsInState=mergedIncidentsByCongressionalDistricts2014.groupby('state')['party'].apply(lambda x: (x=='R').sum()).reset_index(name='count')
democraticRepsInState=mergedIncidentsByCongressionalDistricts2014.groupby('state')['party'].apply(lambda x: (x=='D').sum()).reset_index(name='count')
swingRepsInState=mergedIncidentsByCongressionalDistricts2014.groupby('state')['party'].apply(lambda x: (x=='Swing').sum()).reset_index(name='count')

#Merge our counts of dems, reps and swings with the average amount of firearm laws between 2014-2017
mergedFirearmLawsWithCongressionInformation = pandas.merge(averageStateFirearmLawData, republicanRepsInState, on=['state'], how='outer')
mergedFirearmLawsWithCongressionInformation = pandas.merge(mergedFirearmLawsWithCongressionInformation, democraticRepsInState, on=['state'], how='outer')
mergedFirearmLawsWithCongressionInformation = pandas.merge(mergedFirearmLawsWithCongressionInformation, swingRepsInState, on=['state'], how='outer')
mergedFirearmLawsWithCongressionInformation.columns = ['state','lawtotal','rep count','dem count','swing count']

#Print our report of firearm law relationship with number of congressional districts by party
mergedFirearmLawsWithCongressionInformation.to_csv("firearmsWithCongressionalInformation.csv")

#Correlation of officer involved shootings
aggregated2014Correlation = aggregated2014gunViolenceData.corr(method='pearson')
aggregated2014Correlation.to_csv('aggregated2014CorrelationInfo.csv')
aggregated2015Correlation = aggregated2015gunViolenceData.corr(method='pearson')
aggregated2015Correlation.to_csv('aggregated2015CorrelationInfo.csv')
aggregated2016Correlation = aggregated2016gunViolenceData.corr(method='pearson')
aggregated2016Correlation.to_csv('aggregated2016CorrelationInfo.csv')
aggregated2017Correlation = aggregated2017gunViolenceData.corr(method='pearson')
aggregated2017Correlation.to_csv('aggregated2017CorrelationInfo.csv')

OIX2014 = aggregated2017gunViolenceData['officer']
OIY2014 = aggregated2017gunViolenceData['numIncidents']

OI2014Reg = linear_model.LinearRegression()
OI2014Model = sm.OLS(OIX2014,OIY2014).fit()
#print(OI2014Model.summary())

mergedIncidentDataByStateCorrelation = mergedIncidentDataByState.corr(method='pearson')
mergedIncidentDataByStateCorrelation.to_csv('mergedIncidentDataByStateCorrelation.csv')

#TODO - LINEAR REGRESSION
#print(mergedIncidentDataByState)

X = mergedIncidentDataByState['lawtotal2017']
Y = mergedIncidentDataByState['2017 Incidents']

regr = linear_model.LinearRegression()
model = sm.OLS(X,Y).fit()
print(model.summary())

