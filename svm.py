################
# SVM Assignment 
################

# Import libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import svm


sun_data = pd.read_csv('SunData.csv')
moon_data = pd.read_csv('MoonData.csv')

clf = sk.svm.SVC()
clf.fit(sun_data, moon_data)
clf.support_vectors_



clf.predict(sun_data)
sun_data.to_csv('SunDataPredictions.csv')
clf.predict(moon_data)
moon_data.to_csv('MoonDataPredictions.csv')


