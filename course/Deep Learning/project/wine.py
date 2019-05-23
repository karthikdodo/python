#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:12:14 2019

@author: karthikchowdary
"""
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split

import pandas as pd

data = pd.read_csv("winequalityN.csv", sep = ",") 
data.wine[data.wine == 'white'] = 1
data.wine[data.wine == 'red'] = 2
with open("winequalityN.csv",'r') as f:
    with open("updated_test.csv",'w') as f1:
        next(f) # skip header line
        for line in f:
            f1.write(line)

dataset = pd.read_csv("updated_test.csv", sep = ",").values

print(dataset)

import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,1:12], dataset[:,0],
                                                    test_size=0.25, random_state=87)



np.random.seed(155)

from tensorflow.python.framework import ops
ops.reset_default_graph()

my_first_nn = Sequential()
my_first_nn.add(Dense(105, input_dim=12, activation='relu'))


my_first_nn.add(Dense(125, input_dim=105, activation='relu'))
my_first_nn.add(Dense(1, activation='sigmoid')) 
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100)
print(my_first_nn.summary())