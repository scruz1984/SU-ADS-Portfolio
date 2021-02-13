import nltk
import pandas as pd
import sklearn
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import pyLDAvis
import pyLDAvis.sklearn as LDAvis
from sklearn.svm import LinearSVC
import string
import numpy
import seaborn
from textblob import TextBlob


ayottePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Ayotte"
barassoPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Barasso"
bluntPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Blunt"
boozmanPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Boozman"
burrPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Burr"
corkerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Corker"
crapoPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Crapo"
cruzPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Cruz"
fischerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Fischer"
flakePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Flake"
grassleyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Grassley"
hatchPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Hatch"
hellerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Heller"
hoevenPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Hoeven"
isaksonPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Isakson"
johnsonPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Johnson"
kirkPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Kirk"
lankfordPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Lankford"
leePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Lee"
mcCainPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\McCain"
moranPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Moran"
murkowskiPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Murkowski"
paulPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Paul"
portmanPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Portman"
rubioPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Rubio"
scottPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Scott"
shelbyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Shelby"
thunePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Thune"
toomeyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Toomey"
vitterPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Vitter"
wickerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Republican\\Wicker"


sentimentDF = pd.DataFrame(columns = ['Party', 'Statement'])

repListOfCompleteFiles = []
demListOfCompleteFiles = []

for name in os.listdir(ayottePath):
    nextL= ayottePath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(barassoPath):
    nextL= barassoPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(bluntPath):
    nextL= bluntPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(boozmanPath):
    nextL= boozmanPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(burrPath):
    nextL= burrPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(corkerPath):
    nextL= corkerPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

# for name in os.listdir(crapoPath):
#     nextL= crapoPath + "\\" + name
#     repListOfCompleteFiles.append(nextL)
    
#     f = open(nextL, encoding="ISO-8859-1")
#     text = f.read()
#     sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(cruzPath):
    nextL= cruzPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(fischerPath):
    nextL= fischerPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(flakePath):
    nextL= flakePath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(grassleyPath):
    nextL= grassleyPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(hatchPath):
    nextL= hatchPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(hellerPath):
    nextL= hellerPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(hoevenPath):
    nextL= hoevenPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(isaksonPath):
    nextL= isaksonPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(johnsonPath):
    nextL= johnsonPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(kirkPath):
    nextL= kirkPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(lankfordPath):
    nextL= lankfordPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(leePath):
    nextL= leePath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(mcCainPath):
    nextL= mcCainPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(moranPath):
    nextL= moranPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(murkowskiPath):
    nextL= murkowskiPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(paulPath):
    nextL= paulPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(portmanPath):
    nextL= portmanPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(rubioPath):
    nextL= rubioPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(scottPath):
    nextL= scottPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(shelbyPath):
    nextL= shelbyPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(thunePath):
    nextL= thunePath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(toomeyPath):
    nextL= toomeyPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(vitterPath):
    nextL= vitterPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)

for name in os.listdir(wickerPath):
    nextL= wickerPath + "\\" + name
    repListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'R', 'Statement': text}, ignore_index=True)
    
# Democrat Statements
baldwinPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Baldwin"
bennetPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Bennet"
blumenthalPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Blumenthal"
boxerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Boxer"
brownPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Brown"
cantwellPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Cantwell"
cardinPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Cardin"
carperPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Carper"
caseyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Casey"
donnellyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Donnelly"
feinsteinPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Feinstein"
gillibrandPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Gillibrand"
heinrichPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Heinrich"
heitkampPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\heitkamp"
hironoPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Hirono"
kainePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Kaine"
klobucharPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Klobuchar"
leahyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Leahy"
manchinPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Manchin"
mcCaskillPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\McCaskill"
menendezPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Menendez"
mikulskiPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Mikulski"
murphyPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Murphy"
murrayPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Murray"
nelsonPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Nelson"
reidPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Reid"
schatzPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Schatz"
schumerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Schumer"
stabenowPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Stabenow"
testerPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Tester"
warrenPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Warren"
whitehousePath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Whitehouse"
wydenPath = "C:\\Users\\thecr\\Desktop\\Data Science\\Fifth Quarter\\NLP\\Project\\Press Release Data\\Press Release Data\\Democrat\\Wyden"

for name in os.listdir(baldwinPath):
    nextL= baldwinPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(bennetPath):
    nextL= bennetPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(blumenthalPath):
    nextL= blumenthalPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(boxerPath):
    nextL= boxerPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(brownPath):
    nextL= brownPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(cantwellPath):
    nextL= cantwellPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(cardinPath):
    nextL= cardinPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(carperPath):
    nextL= carperPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(caseyPath):
    nextL= caseyPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(donnellyPath):
    nextL= donnellyPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(feinsteinPath):
    nextL= feinsteinPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(gillibrandPath):
    nextL= gillibrandPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(heinrichPath):
    nextL= heinrichPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(heitkampPath):
    nextL= heitkampPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(hironoPath):
    nextL= hironoPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(kainePath):
    nextL= kainePath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(klobucharPath):
    nextL= klobucharPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(leahyPath):
    nextL= leahyPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(manchinPath):
    nextL= manchinPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(mcCaskillPath):
    nextL= mcCaskillPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(menendezPath):
    nextL= menendezPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(mikulskiPath):
    nextL= mikulskiPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(murphyPath):
    nextL= murphyPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(murrayPath):
    nextL= murrayPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(nelsonPath):
    nextL= nelsonPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(reidPath):
    nextL= reidPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(schatzPath):
    nextL= schatzPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(schumerPath):
    nextL= schumerPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(stabenowPath):
    nextL= stabenowPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(testerPath):
    nextL= testerPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(warrenPath):
    nextL= warrenPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(whitehousePath):
    nextL= whitehousePath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)

for name in os.listdir(wydenPath):
    nextL= wydenPath + "\\" + name
    demListOfCompleteFiles.append(nextL)
    
    f = open(nextL, encoding="utf8")
    text = f.read()
    sentimentDF = sentimentDF.append({'Party' : 'D', 'Statement': text}, ignore_index=True)




#col_one_list = df['one'].tolist()


sentimentClassificationVectorizer = CountVectorizer(input='content',analyzer='word',stop_words='english',lowercase=True)
sentimentClassificationDF = pd.DataFrame()

allStatements = sentimentDF['Statement'].tolist()
allPartyAffiliations = sentimentDF['Party'].tolist()

statementCountVectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',lowercase=True, max_features=100)
statementTfidfVectorizer = TfidfVectorizer(input='content',stop_words='english',max_features=100)

statementCVFitTransform = statementCountVectorizer.fit_transform(allStatements)
statementTfidfFitTransform = statementTfidfVectorizer.fit_transform(allStatements)

colNames = statementTfidfVectorizer.get_feature_names()

sentimentDataFrame = pd.DataFrame(statementCVFitTransform.toarray(), columns=colNames)

for nextCol in sentimentDataFrame.columns:
    if(any(char.isdigit() for char in nextCol)):
        sentimentDataFrame.drop([nextCol], axis=1)
    elif(len(str(nextCol)) <= 3):
        sentimentDataFrame.drop([nextCol], axis=1)
            
sentimentDataFrame.insert(loc = 0, column = 'LABEL', value = allPartyAffiliations)

sentimentTrainDF, sentimentTestDF = train_test_split(sentimentDataFrame, test_size=.3)

testSentimentLabels = sentimentTestDF["LABEL"]
sentimentTestDF = sentimentTestDF.drop(["LABEL"], axis=1)

trainSentimentLabels = sentimentTrainDF["LABEL"]
sentimentTrainDF = sentimentTrainDF.drop(["LABEL"], axis=1)





sentimentModelNB = MultinomialNB()
sentimentModelNB.fit(sentimentTrainDF, trainSentimentLabels)
sentimentPrediction = sentimentModelNB.predict(sentimentTestDF)
sentimentConfusionMatrix = confusion_matrix(testSentimentLabels, sentimentPrediction)
group_names = ['True Dem','False Rep','False Dem','True Rep']
group_counts = ['{0:0.0f}'.format(value) for value in sentimentConfusionMatrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in (sentimentConfusionMatrix/numpy.sum(sentimentConfusionMatrix)).flatten()]
labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(group_names, group_counts, group_percentages)]
labels = numpy.asarray(labels).reshape(2,2)
seaborn.heatmap(sentimentConfusionMatrix/numpy.sum(sentimentConfusionMatrix), annot=labels, fmt='', cmap='Blues')




# rbfSvmModel = sklearn.svm.SVC(C=10000, kernel='rbf', verbose=True, gamma='auto')
# rbfSvmModel.fit(sentimentTrainDF, trainSentimentLabels)
# rbfSvmPredict = rbfSvmModel.predict(sentimentTestDF)
# rbfSvmMatrix = confusion_matrix(testSentimentLabels, rbfSvmPredict)
# print(rbfSvmMatrix)

# r_group_names = ['True Dem','False Rep','False Dem','True Rep']
# r_group_counts = ['{0:0.0f}'.format(value) for value in rbfSvmMatrix.flatten()]
# r_group_percentages = ['{0:.2%}'.format(value) for value in (rbfSvmMatrix/numpy.sum(rbfSvmMatrix)).flatten()]
# r_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(r_group_names, r_group_counts, r_group_percentages)]
# r_labels = numpy.asarray(r_labels).reshape(2,2)

# seaborn.heatmap(rbfSvmMatrix/numpy.sum(rbfSvmMatrix), annot=r_labels, fmt='', cmap='Blues')




#LINEAR
linearSvmModel = LinearSVC(C=100)

linearSvmModel.fit(sentimentTrainDF, trainSentimentLabels)
linearSvmPredict = linearSvmModel.predict(sentimentTestDF)
print(linearSvmPredict)
linearSvmMatrix = confusion_matrix(testSentimentLabels, linearSvmPredict)
print(linearSvmMatrix)

l_group_names = ['True Dem','False Rep','False Dem','True Rep']
l_group_counts = ['{0:0.0f}'.format(value) for value in linearSvmMatrix.flatten()]
l_group_percentages = ['{0:.2%}'.format(value) for value in (linearSvmMatrix/numpy.sum(linearSvmMatrix)).flatten()]
l_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(l_group_names, l_group_counts, l_group_percentages)]
l_labels = numpy.asarray(l_labels).reshape(2,2)

seaborn.heatmap(linearSvmMatrix/numpy.sum(linearSvmMatrix), annot=l_labels, fmt='', cmap='Blues')




#POLY
polySvmModel = sklearn.svm.SVC(C=50, kernel='poly',degree=3, gamma='auto',verbose=True)
polySvmModel.fit(sentimentTrainDF, trainSentimentLabels)
polySvmPredict = polySvmModel.predict(sentimentTestDF)
polySvmMatrix = confusion_matrix(testSentimentLabels, polySvmPredict)
print(polySvmMatrix)

p_group_names = ['True Dem','False Rep','False Dem','True Rep']
p_group_counts = ['{0:0.0f}'.format(value) for value in polySvmMatrix.flatten()]
p_group_percentages = ['{0:.2%}'.format(value) for value in (polySvmMatrix/numpy.sum(polySvmMatrix)).flatten()]
p_labels = [f'{v1}\n{v2}\n{v3}' for v1,v2,v3 in zip(p_group_names, p_group_counts, p_group_percentages)]
p_labels = numpy.asarray(p_labels).reshape(2,2)

seaborn.heatmap(polySvmMatrix/numpy.sum(polySvmMatrix), annot=p_labels, fmt='', cmap='Blues')





def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
            
ldaVectorizer = CountVectorizer(input='filename', stop_words='english')

ldaVectorizerFitTransform = ldaVectorizer.fit_transform(demListOfCompleteFiles)

columnNames = ldaVectorizer.get_feature_names()

corpusDataFrame = pd.DataFrame(ldaVectorizerFitTransform.toarray(), columns=columnNames)

#print(corpusDataFrame)

ldaModel = LatentDirichletAllocation(n_components=3, max_iter=10, learning_method='online')

ldaModelFitTransform = ldaModel.fit_transform(ldaVectorizerFitTransform)

#print(ldaModelFitTransform[0])

print_topics(ldaModel, ldaVectorizer)

#Visualize The Model
panel = LDAvis.prepare(ldaModel, ldaVectorizerFitTransform, ldaVectorizer, mds='tsne')
pyLDAvis.show(panel)










