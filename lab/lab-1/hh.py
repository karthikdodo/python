#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:47:11 2019

@author: karthikchowdary
"""

netAmount = 0
while True:
    user_s = input("enter the operation and then amount: ")
    if not user_s:
        break
    values = user_s.split()
    operation = values[0]
    amount = int(values[1])
    if operation == "D":
        netAmount += amount
    elif operation == "W":
        netAmount -= amount
    else:
        pass
    print(netAmount)
