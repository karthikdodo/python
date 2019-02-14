#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:30:49 2019

@author: karthikchowdary
"""

Py = {"karthik", "santosh", "mourya","sachin", "taylor", "gilly"}

# students list who took web
web = {"karthik", "fire", "gayle", "taylor", "santosh"}



print("who take both python and web::",Py & web)

onlypython = Py-web


onlyweb= web-Py

print("not in unique subjects::",onlypython.union(onlyweb))
i=1
while(i):
    i=input("select python or web or 0 to exit")
    if(i=="python"):
        print(Py)
    elif(i=="web"):
        print(web)
    else:
        break
    
    

"""print("are in python but not in web::",onlypython)

print("only in web",onlyweb)"""
