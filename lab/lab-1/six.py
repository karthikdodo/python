#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:58:44 2019

@author: karthikchowdary
"""
import urllib.request
from bs4 import BeautifulSoup

file1 = open("table_txt", "w+")
wikiurl = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"


openURL = urllib.request.urlopen(wikiurl)


soup = BeautifulSoup(openURL, "html.parser")


for rows in soup.find_all('th'):

    file1.write(str(rows.text))

file1.seek(0,0)
string1 = file1.read()
print(string1)
file1.close()