################
# SVM Assignment 
################

# Import libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read in CSV data
moon_data = pd.read_csv('MoonData.csv',usecols=['Eclipse Type','Catalog Number', 'Delta T (s)', 'Lunation Number', 'Saros Number', 'Gamma', 'Penumbral Magnitude', 'Umbral Magnitude','Penumbral Eclipse Duration (m)', 'Partial Eclipse Duration (m)'])

# 
moon_data['Eclipse Type'] = (moon_data['Eclipse Type'] == "T").astype(int)
moon_data = moon_data.fillna(-1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(moon_data.drop(columns='Eclipse Type'), moon_data['Eclipse Type'], test_size=0.2, random_state=42)

moon_data_type = []
for t in moon_data['Eclipse Type']:
    moon_data_type.append(t)

moon_data = moon_data.drop('Eclipse Type', axis=1)

# Different kernels are inputted
clf_linear = svm.SVC(kernel = "linear")
clf_rbf = svm.SVC(kernel = "rbf")
clf_poly = svm.SVC(kernel = "poly", degree = 3)
clf_sig = svm.SVC(kernel = "sigmoid")

# Each fit is created
clf_linear.fit(X_train, y_train)
clf_rbf.fit(X_train, y_train)
clf_poly.fit(X_train, y_train)
clf_sig.fit(X_train, y_train)

# The predictions are tested
prediction_linear = clf_linear.predict(X_test)
prediction_rbf = clf_rbf.predict(X_test)
prediction_poly = clf_poly.predict(X_test)
prediction_sig = clf_sig.predict(X_test)

# Accuracy scores are found
accuracy_linear = accuracy_score(y_test, prediction_linear)
accuracy_rbf = accuracy_score(y_test, prediction_rbf)
accuracy_poly = accuracy_score(y_test, prediction_poly)
accuracy_sig = accuracy_score(y_test, prediction_sig)

# Results are exported
accuracy = pd.DataFrame({'Kernel': ['Linear', 'RBF', 'Poly', 'Sigmoid'], 'Accuracy': [accuracy_linear, accuracy_rbf, accuracy_poly, accuracy_sig]})
accuracy.to_csv('moon_predictions.csv', index=False)