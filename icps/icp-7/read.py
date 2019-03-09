#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:32:50 2019

@author: karthikchowdary
"""

import urlib.request
from bs4 import BeautifulSoup

wikiURL="https://en.wikipedia.org/wiki/Google"
openURL=urllib.request.urlopen(wikiURL)
soup=BeautifulSoup(openURL.read(),"lxml")
for script in soup(["script","style"]):
    script.extract()
text=soup.body.get_text()

lines=(line.strip() for line in text.splitlines())
chunks=(phrase.strip() for line in lines for phrase in line.split(" ))
text= ''.join(chunk for chunkl in chunks if chunk)

with open('input.txt','w') as text_file:
    
