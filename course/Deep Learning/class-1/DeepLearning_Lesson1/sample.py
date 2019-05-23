#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:39:10 2019

@author: karthikchowdary
"""

fields=['diagnosis']
target = pd.read_csv('BreastCancer.csv', skipinitialspace=True, usecols=fields)


mapping={'M':0,'B':1}
target.replace({'M':mapping, 'B':mapping})
print(target)