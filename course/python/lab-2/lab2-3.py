# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 15:41:57 2019

@author: karthikchowdary
"""
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk, ngrams, FreqDist, sent_tokenize, word_tokenize
 


file=open("nlp_input.txt","r+")
fileData = file.read()

# Word Tokenization - to extarct each word from the text
tokens = word_tokenize(fileData)

# Applying Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatizerOutput = []
print("Lemmatization Output : \n")
for tok in tokens:
    #itearting through each word and lematizing and appending it to list
    lemmatizerOutput.append(lemmatizer.lemmatize(str(tok)))
print(lemmatizerOutput)

# performing trigram on the  Lemmatizer Output
print("trigrams :\n")
trigramsOutput = []
for tri in ngrams(tokens, 3):
    # Fetching trigrams using 'ngrams' method and Iterating it and appending it to list
    trigramsOutput.append(tri)
print(trigramsOutput)

# triGram- Word Frequency
# Using trigramOutput fetch the WordFreq Details
wordFreq = FreqDist(trigramsOutput)
# Getting Most Common Words and Printing them - Will get the Counts from top to least
mostCommon = wordFreq.most_common()
print("triGrams Frequency (From Top to Least) : \n", mostCommon)
# Fetching the Top 10 trigrams
top10 = wordFreq.most_common(10)
print("Top 10 triGrams : \n", top10)


# Getting Sentences using Sentence Tokenization
sentTokens = sent_tokenize(fileData)
# Creating an Array to append the sentence
concatenatedArray = []
# Iterating the Sentences
for sentence in sentTokens:
    # Iterating the trirams present
    for a, b, c in trigramsOutput:
        # Iterating the Top 10 triGrams
        for ((d, e, f), length) in top10:
            # Comparing the each with each of the Top 10 trigram
            if(a == d, e, f):
                concatenatedArray.append(sentence)

print("Concatenated Array : ",concatenatedArray)
print("Maximum of Concatenated Array : ", max(concatenatedArray))