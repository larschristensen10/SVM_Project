################
# SVM Assignment 
################

# Import libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import svm
import matplotlib.pyplot as plt

sun_data = pd.read_csv('SunData.csv',usecols=['Eclipse Type', 'Catalog Number','Delta T (s)','Lunation Number','Saros Number','Gamma','Eclipse Magnitude','Sun Altitude','Sun Azimuth','Path Width (km)'])
moon_data = pd.read_csv('MoonData.csv',usecols=['Catalog Number', 'Delta T (s)', 'Lunation Number', 'Saros Number', 'Gamma', 'Penumbral Magnitude', 'Umbral Magnitude','Penumbral Eclipse Duration (m)', 'Partial Eclipse Duration (m)', 'Total Eclipse Duration (m)'])


# moon_data = moon_data.dropna()

sun_data['Eclipse Type'] = (sun_data['Eclipse Type'] == "T").astype(int)
sun_data = sun_data.fillna(-1)


sun_data_type = []
for t in sun_data['Eclipse Type']:
    sun_data_type.append(t)

# sun_data_type = pd.DataFrame(columns=['Eclipse Type'])
# for i in range(sun_data['Eclipse Type'].size()):
#     sun_data_type['Eclipse Type'][i] = 1 if sun_data_type[i] == 'T' else -1




# print(sun_data_type)
# print(sun_data['Eclipse Type'])
sun_data = sun_data.drop('Eclipse Type', axis=1)

clf_linear = svm.SVC(kernel = "linear")
clf_rbf = svm.SVC(kernel = "rbf")
clf_poly = svm.SVC(kernel = "poly", degree = 3)
clf_sig = svm.SVC(kernel = "sigmoid")

clf_linear.fit(sun_data, sun_data_type)
clf_rbf.fit(sun_data, sun_data_type)
clf_poly.fit(sun_data, sun_data_type)
clf_sig.fit(sun_data, sun_data_type)





clf_linear.predict(sun_data)
sun_data.to_csv('SunDataPredictions.csv')



# # clf.predict(moon_data)
# # moon_data.to_csv('MoonDataPredictions.csv')


