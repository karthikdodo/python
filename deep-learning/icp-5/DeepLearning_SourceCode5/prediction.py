#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:15:06 2019

@author: karthikchowdary
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model


sentence=["A lot of good things are happening. We are respected again throughout the world, and that's a great thing"]

max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(sentence)

X = tokenizer.texts_to_sequences(sentence)

X = pad_sequences(X,maxlen=28)

model=load_model('big.h5')

result=model.predict(X)
print(result)