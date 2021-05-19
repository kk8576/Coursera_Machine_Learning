#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:30:35 2021

@author: wangkun
"""

import numpy as np
from scipy.optimize import minimize


def cost_func(theta, X, y, lambda_parameter, g_flag = True):
    m = X.shape[0]
    
    X0 = np.concatenate((np.ones((m, 1)), X), axis = 1)
    
    y0 = np.sum(X0 * theta, axis = 1)
    
    error = y0.reshape((-1,1)) - y.reshape((-1,1))
    
    J = 1.0 / (2.0 * m) * np.sum(error * error) + lambda_parameter / (2.0 * m) * np.sum(theta[1:] * theta[1:])
    
    if not g_flag:
        return J
    
    grad = 1.0 / (2.0 * m) * np.sum(2.0 * X0 * error, axis = 0)
    
    grad[1:] += lambda_parameter / (2.0 * m) * np.sum(2.0 * theta[1:], axis = 0)
    
    return J, grad

def train(X, y, lambda_parameter):
    initial_theta = np.zeros((1, X.shape[1] + 1))
    
    res = minimize(cost_func, 
                   initial_theta, 
                   method='BFGS',
                   jac = True,
                   args = (X, y, lambda_parameter),
                   options = {'disp': False, 'maxiter' : 500})
    
    succ = res['success']
    
    if not succ:
        res = minimize(cost_func, 
                   initial_theta, 
                   method='Powell',
                   jac = False,
                   args = (X, y, lambda_parameter, False),
                   options = {'disp': False, 'maxiter' : 500})
        
    theta = res['x']
    c = res['fun']
        
    return theta, c