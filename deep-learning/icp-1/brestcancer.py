#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:42:18 2019

@author: karthikchowdary
"""

import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("BreastCancer.csv", header=None).values
# print(dataset)
import numpy as np

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:32], dataset[:,1],
                                                    test_size=0.25, random_state=87)



np.random.seed(155)

my_first_nn = Sequential() # create model
my_first_nn.add(Dense(45, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam')
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100)


print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
