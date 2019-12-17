#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:46:16 2019

@author: carsonellsworth
"""
import numpy as np
from mlxtend.data import loadlocal_mnist


X, y = loadlocal_mnist(
        images_path='train-images', 
        labels_path='train-labels')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))

np.savetxt(fname='images.csv', 
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname='labels.csv', 
           X=y, delimiter=',', fmt='%d')