#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:44:22 2019

@author: karthikchowdary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')

# Plot
plt.scatter(x=train['GarageArea'], y=train['SalePrice'])
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


data=train.drop(train[train['GarageArea']>1200].index)
data=train.drop(train[train['SalePrice']>70000].index)
plt.scatter(x=data['GarageArea'], y=data['SalePrice'])
plt.title('Scatter plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()