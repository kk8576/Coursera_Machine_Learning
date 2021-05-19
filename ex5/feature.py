#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:29:03 2021

@author: wangkun
"""

import numpy as np

def polynomial_features(X, p):
    if p == 1:
        return X
    
    X1 = X
    for i in range(2, p+1):
        X = np.concatenate((X, np.power(X1, i)), axis = 1)
            
    return X

def normalize_features(X):
    """
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    m = np.mean(X, axis = 0)
    s = np.std(X, axis = 0)
    
    X = (X - m) / s
    
    return X, m, s