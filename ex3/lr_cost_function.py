#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:54:12 2021

@author: wangkun
"""

import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost_function(theta, X, y, parameter_lambda):
    # number of training examples
    m = y.shape[0]
    
    # y * log(h(x) + (1 - y) * log(1 - h)
    #print(np.matmul(X, theta))
    h = sigmoid(np.matmul(X, theta)).reshape((m, 1))

    c = y * np.log(h) + (1 - y) * np.log(1 - h)

    # cost function value
    J = -1.0 / float(m) * np.sum(c, axis=0) + parameter_lambda / (2.0 * float(m)) * (np.sum(theta * theta, axis=0) - theta[0] * theta[0])
    
    #grad = np.matmul(X.transpose(), (h - y)) / float(m) + parameter_lambda / float(m) * theta
    #grad[0] -= parameter_lambda / float(m) * theta[0]
    
    return J[0]

def cost_function_jac(theta, X, y, parameter_lambda):
    # number of training examples
    m = y.shape[0]
    
    # y * log(h(x) + (1 - y) * log(1 - h)
    h = sigmoid(np.matmul(X, theta)).reshape((m, 1))
    
    grad = np.matmul(X.transpose(), (h - y)) / float(m)
    grad += (parameter_lambda / float(m) * theta).reshape((theta.shape[0], 1))
    grad[0] -= parameter_lambda / float(m) * theta[0]
    
    return grad.reshape((grad.shape[0],))
    