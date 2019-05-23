#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:33:05 2019

@author: karthikchowdary
"""

# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import pandas as pd
import warnings
warnings.simplefilter("ignore")
# Importing the dataset
dataset = pd.read_csv('/Users/karthikchowdary/Desktop/KarthiK/graduate/spring-19/python/lab-2/diabetes.csv')

dataset.describe()
dataset["Insulin"].value_counts()
dataset.groupby(['Insulin', 'BMI']).mean()

dataset = dataset.fillna(dataset.mean())



#X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
X = dataset.drop(['Pregnancies','Insulin','SkinThickness','BMI'],axis=1)
#dataset = dataset.fillna(dataset.mean())



#df = df_train.drop(['Summary','Daily Summary'],axis=1)

#X = pd.get_dummies(X, columns=["Precip Type"])
#before EDA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
print("Variance score: %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X.values[:, 1] = labelencoder.fit_transform(X.values[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
corr = dataset.corr()

print (corr['Insulin'].sort_values(ascending=False)[:3], '\n')
print (corr['Insulin'].sort_values(ascending=False)[-3:])

# Avoiding the Dummy Variable Trap
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
model = regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


#Evaluating the model

dataset = dataset.fillna(dataset.mean())
#dataset_numeric = dataset.filter(like="_N", axis=1)
from sklearn.metrics import mean_squared_error, r2_score
print("Variance score: %.2f" % r2_score(y_test,y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test,y_pred))