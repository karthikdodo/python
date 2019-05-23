#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 21:22:39 2019

@author: karthikchowdary
"""

import numpy as np 
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from keras.utils import to_categorical
import random
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.layers import Dense, Activation,Flatten
from keras.layers.convolutional import Conv1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
lemmatizer = WordNetLemmatizer()
set_random_seed(123)
random.seed(123)

train= pd.read_csv("train.tsv", sep="\t")
test = pd.read_csv("test.tsv", sep="\t")

train.head()
train['Phrase'][143]


def clean_sentences(df):
    reviews = []

    for sent in tqdm(df['Phrase']):
        review_text = BeautifulSoup(sent).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        words = word_tokenize(review_text.lower())
        lemma_words = [lemmatizer.lemmatize(i) for i in words]
        reviews.append(lemma_words)

    return(reviews)
    
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print(len(train_sentences))
print(len(test_sentences))
target=train.Sentiment.values
y=to_categorical(target)
num_classes=y.shape[1]


X_train,X_val,y_train,y_val=train_test_split(train_sentences,y,test_size=0.3,stratify=y)
unique_words = set()
len_max = 0

for sent in tqdm(X_train):
    
    unique_words.update(sent)
    
    if(len_max<len(sent)):
        len_max = len(sent)
        
#length of the list of unique_words gives the no of unique words
print(len(list(unique_words)))
print(len_max)
for x in tqdm(X_train[1:10]):
  print(x)
tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)
X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)
print(X_train.shape,X_val.shape,X_test.shape)

early_stopping = EarlyStopping(min_delta = 0.001, mode = 'max', monitor='val_acc', patience = 2)
callback = [early_stopping]

model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(32,dropout=0.3, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.005),metrics=['accuracy'])
model.summary()
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=3, batch_size=66, verbose=1)

