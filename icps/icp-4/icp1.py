#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 23:11:56 2019

@author: karthikchowdary
"""

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

# Importing dataset
data = pd.read_csv("/Users/karthikchowdary/Desktop/KarthiK/graduate/spring-19/python/class-4/iris.csv")
iris=datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.4,random_state=0)
MltiNB= MultinomialNB()
MltiNB.fit(x_train,y_train)
print(MltiNB)
y_expect=y_test
y_pred=MltiNB.predict(x_test)
print(y_pred)
print(y_test)
print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred,average='macro'))
print(metrics.f1_score(y_test,y_pred,average='macro'))
print(metrics.recall_score(y_test,y_pred,average='macro'))