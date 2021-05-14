#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:16:11 2021

@author: wangkun
"""

import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))    

def sigmoid_gradient(x):
    return 1.0 / ((1.0 + np.exp(-x)) * (1.0 + np.exp(-x))) * np.exp(-x)
    

def cost_func(parameters, input_layer_size, hidden_layer_size, num_labels,
              X, y, lambda_parameter):
    
    theta1 = parameters[0:(input_layer_size + 1) * hidden_layer_size].reshape((hidden_layer_size, input_layer_size + 1)) 
    theta2 = parameters[(input_layer_size + 1) * hidden_layer_size:].reshape((num_labels, hidden_layer_size + 1))

    m = X.shape[0]
    
    #part 1: cost function value
    a1 = np.concatenate((np.ones((m, 1)), X), axis = 1)
    z2 = np.matmul(a1, theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis = 1)
    z3 = np.matmul(a2, theta2.transpose())
    a3 = sigmoid(z3)
    
    c = - y * np.log(a3) - (1.0 - y) * np.log(1.0 - a3)    
    J = np.sum(c) * (1.0 / m) + 0.5 * lambda_parameter / m * (np.sum(theta1[:,1:] * theta1[:,1:]) + np.sum(theta2[:,1:] * theta2[:,1:]))
    
    #part 2: backpropagation
    sigma3 = a3 - y
    sigma2 = np.matmul(sigma3, theta2) * a2 * (1 - a2)
    
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    
    #for i in range(m):
    #    theta2_grad += np.matmul(sigma3[i].reshape((sigma3.shape[1], 1)), a2[i].transpose().reshape((1, a2.shape[1])))
    #    theta1_grad += np.matmul(sigma2[i, 1:].reshape((sigma2.shape[1] - 1, 1)), a1[i].transpose().reshape((1, a1.shape[1])))
        
    theta2_grad += np.sum(np.matmul(sigma3.reshape(sigma3.shape[0], sigma3.shape[1], 1), 
                                    a2.reshape((a2.shape[0], a2.shape[1], 1)).transpose((0, 2, 1))), axis = 0)
    theta1_grad += np.sum(np.matmul(sigma2[:,1:].reshape((sigma2.shape[0], sigma2.shape[1] - 1, 1)),
                                    a1.reshape((a1.shape[0], a1.shape[1], 1)).transpose(0, 2, 1)), axis = 0)
    
    
    
    theta1_grad[:,1:] += lambda_parameter * theta1[:,1:]
    theta2_grad[:,1:] += lambda_parameter * theta2[:,1:]
    
    grad = np.concatenate((theta1_grad.flatten() / m, theta2_grad.flatten() / m))
   
    return J, grad


def rand_initialize_weights(in_size, out_size):
    epsilon_init = np.sqrt(6.0) / np.sqrt(in_size + out_size)
    
    W = np.random.rand(out_size, 1 + in_size) * 2 * epsilon_init - epsilon_init;
    
    return W

def debug_initialize_weights(in_size, out_size):
    W = np.sin(np.arange((in_size + 1) * out_size)).reshape(out_size, in_size + 1)
    
    return W


def check_nn_gradient(lambda_parameter = 0.0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    theta1 = debug_initialize_weights(input_layer_size, hidden_layer_size);
    theta2 = debug_initialize_weights(hidden_layer_size, num_labels);
    
    # Reusing debugInitializeWeights to generate X
    X = debug_initialize_weights(input_layer_size - 1, m);
    y = 1 + np.arange(m, dtype = np.int32) / num_labels
    y = y.reshape((m, 1))
    
    nn_params = np.concatenate((theta1.flatten(), theta2.flatten()), axis = 0)
    
    J, grad = cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_parameter)

    size = nn_params.size
        
    num_grad = np.zeros(size)
    perturb = np.zeros(size)
    eps = 0.0001
    for i in range(size):
        perturb[i] = eps
        
        J1, grad1 = cost_func(nn_params - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_parameter)
        J2, grad2 = cost_func(nn_params + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_parameter)

        num_grad[i] = (J2 - J1) / (2.0 * eps)
        
        perturb[i] = 0.0
    
    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)
    
    if diff < 1e-9:
        print('\n  The backpropagation implementation is correct.')
    else:
        print('\n  The backpropagation implementation is wrong.')
        
    print('\n   The relative difference between analytical and numereical direvative is: ', diff)

    
def train(input_layer_size, hidden_layer_size, num_labels, X, y, initial_nn_params, lambda_parameter):
    res = minimize(cost_func, 
                   initial_nn_params, 
                   method='Newton-CG',
                   jac = True,
                   args = (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_parameter),
                   options = {'disp': True, 'maxiter' : 30})
        
    nn_params = res['x']
    c = res['fun']
        
    return nn_params, c
    
 
def predict(theta1, theta2, X):
    m = X.shape[0]
    
    a1 = np.concatenate((np.ones((m, 1)), X), axis = 1)
    z2 = np.matmul(a1, theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.concatenate((np.ones((m, 1)), a2), axis = 1)
    z3 = np.matmul(a2, theta2.transpose())
    a3 = sigmoid(z3)
    
    return np.argmax(a3, axis = 1).reshape(a3.shape[0], 1) + 1
    
