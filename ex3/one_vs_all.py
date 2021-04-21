#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:31:12 2021

@author: wangkun
"""

import numpy as np
import lr_cost_function
from scipy.optimize import minimize


def train(X, y, num_labels, parameter_lambda):
    m = X.shape[0]
    n = X.shape[1]
    
    all_theta = np.zeros((num_labels, n + 1))
    
    # add ones to X
    X = np.concatenate((np.ones((m, 1)), X), axis = 1)
    
    for i in range(num_labels):
        print('Number Labels: ', i)
        initial_theta = np.zeros((n + 1, 1))
        
        yt = (y == i + 1) * 1.0
        
        res = minimize(lr_cost_function.cost_function, initial_theta, 
                       method='BFGS',
                       jac=lr_cost_function.cost_function_jac,
                       args = (X, yt, parameter_lambda),
                       options = {'disp': True})
        
        theta = res['x']
        
        all_theta[i, :] = theta.transpose()
                
    return all_theta

def predict(all_theta, X):
    m = X.shape[0]
        
    X = np.concatenate((np.ones((m, 1)), X), axis = 1)
    
    h = lr_cost_function.sigmoid(np.matmul(X, all_theta.transpose()))
    
    p = np.argmax(h, axis = 1) + 1
        
    return p.reshape((m, 1))