#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:35:07 2019

@author: karthikchowdary
"""
import pandas as pd

import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
"""
tsv_file='train.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('train.csv',index=False)
tsv_file='test.tsv'
csv_table=pd.read_table(tsv_file,sep='\t')
csv_table.to_csv('test.csv',index=False)"""
 
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
data = pd.read_csv('train.csv',encoding='cp1252')
dataset=pd.read_csv('train.csv',encoding='cp1252')
data = data[['Phrase','Sentiment']]
data['Phrase'] = data['Phrase'].apply(lambda x: x.lower())
data['Phrase'] = data['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

print(data['Sentiment'].size)
for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['Phrase'].values)
X = tokenizer.texts_to_sequences(data['Phrase'].values)

X = pad_sequences(X)

embed_dim = 128
lstm_out = 196
def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return model
    #print(model.summary())

y=pd.get_dummies(dataset['Sentiment']).values 
print(y)
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.25, random_state = 87)  
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
model = createmodel()
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 4)
score,acc = model.evaluate(X_test,Y_test,verbose=4,batch_size=batch_size)
print(score)
print(acc)
 