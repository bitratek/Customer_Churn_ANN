#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 09:31:33 2022

@author: sudhirbitra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('/Users/sudhirbitra/Complete-Deep-Learning/ANN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:,13]

#Create dummy variables
geography = pd.get_dummies(X["Geography"],drop_first=True)
gender = pd.get_dummies(X["Gender"],drop_first=True)

# Concatenate the Data Frames

X = pd.concat([X, geography, gender],axis=1)

#Drop unnecessary columns
X = X.drop(['Geography', 'Gender'], axis=1)

# Splitting the dataset into Training set and Test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Part 2 - Now let's make the ANN!

#importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()

#Adding the input Layer and the first hidden Layer
classifier.add(Dense(units=10, kernel_initializer= 'he_normal', activation='relu', input_dim = 11))
classifier.add(Dropout(0.3))


#Adding the second hidden Layer
classifier.add(Dense(units=20, kernel_initializer= 'he_normal', activation='relu'))
classifier.add(Dropout(0.4))

# #Adding another hidden Layer
classifier.add(Dense(units=15, kernel_initializer= 'he_normal', activation='relu'))
classifier.add(Dropout(0.2))

#Adding the output Layer
classifier.add(Dense(units=1, kernel_initializer= 'glorot_uniform', activation='sigmoid'))

#compiling the ANN 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN model to Training set
model_history = classifier.fit(X_train, y_train, batch_size=10, epochs=100)

#List all data in history

print(model_history.history.keys())


#part 3 - Making the predictions and evaluating the model

#predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred, y_test)
























