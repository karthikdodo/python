#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 18:09:41 2019

@author: karthikchowdary
"""
def Con(tuple, dictonary):
    for a, b in tuple:
        dictonary.setdefault(a, []).append(b)
    return dictonary


tuple1 = ('John', ('Physics', 80))
tuple2 = ('Daniel', ('Science', 90))
tuple3 = ('John', ('Chemistry', 60))
tuple4 = ('Mark',('Maths',100))
tuple5 = ('Daniel',('History',75))
tuple6 = ('Mark',('Social', 95))


lt1 = [tuple1,tuple2,tuple3,tuple4,tuple5,tuple6]


dict = {}
dict1 = Con(lt1,dict)


print(dict1)

