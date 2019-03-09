#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 17:01:34 2019

@author: karthikchowdary
"""

from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Google').read()


file=open("datafile.txt","r+")
file.write(text_from_html(html))
file.close()


stokens=nltk.tokenize.sent_tokenize(text_from_html(html))
for s in stokens:
    print(s)
    
wtokens=nltk.tokenize.word_tokenize(text_from_html(html))
for s in wtokens:
    print(s)
    
print(nltk.pos_tag(wtokens))

pStemmer=PorterStemmer()
for s in wtokens:
    print(pStemmer.stem(s))
    
lem=WordNetLemmatizer()
for s in wtokens:
    print(lem.lemmatize(s))
    
for s in stokens:
    print(ne_chunk(pos_tag(wordpunct_tokenize(s))))
    

