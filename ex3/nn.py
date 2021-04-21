#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:43:52 2021

@author: wangkun
"""

import numpy as np

def sigmoid(X):
    m = X.shape[0]
    
    for i in range(m):
        X[i, :] = 1.0 / (1.0 + np.exp(-X[i, :]))
        
    return X


def predict(weights, X):
    nlayer = len(weights)
        
    # number of examples
    m = X.shape[0]
    
    for i in range(nlayer):
        X = np.concatenate((np.ones((m, 1)), X), axis = 1)
        
        X = np.matmul(X, weights[i].transpose())
        
        X = sigmoid(X)
                
    return np.argmax(X, axis = 1) + 1
        