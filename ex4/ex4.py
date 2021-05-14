#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:50:20 2021

@author: wangkun
"""

import numpy as np
from scipy.io import loadmat
import random
import display_data
import nn

input_layer_size = 400 
hidden_layer_size = 25
num_labels = 10

# Part 1: Loading and visualizing data
print('\nPart1: Loading and Visualizing Data ...\n')

# load matlab mat format data from the file ex3data1.mat
data = loadmat('ex4data1.mat')

X = data['X']
m = X.shape[0]

y = data['y'].reshape((m, 1))

yt = np.zeros((y.shape[0], num_labels))
yt[np.arange(yt.shape[0]), y.reshape((1, y.shape[0])) - 1] = 1

# Randomly select 100 data points to display
print('\n  Rondomly select 100 examples to display')
rand_indices = random.sample(range(m),100)

sel = X[rand_indices, :]

display_data.display_data(sel)

wait = input('\nProgram paused. Press <ENTER> to continue')

# Part 2: Loading parameters
print('\nPart2: Loading Saved Neural Network Parameters ...')

weights = loadmat('ex4weights.mat')

theta1 = weights['Theta1']
theta2 = weights['Theta2']

theta1_shape = theta1.shape
theta2_shape = theta2.shape

# unroll parameters
nn_params = np.concatenate((theta1.reshape(theta1_shape[0] * theta1_shape[1], 1), 
                            theta2.reshape(theta2_shape[0] * theta2_shape[1], 1)), axis = 0)

#print(nn_params.shape)

#part 3: Compute costt function (feedforward)
print('\nPart3: Feedforward Using Neural Network ...')
lambda_parameter = 0.0
J = nn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels,
              X, yt, lambda_parameter)

print('\n  Cost at parameters (loaded from ex4weights): ', J)
print('  (this value should be about 0.287629.)')

#part 4: Implement Regularization
print('\nPart 4: Checking Cost Function (Regularization) ...')
lambda_parameter = 1.0
J = nn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels,
                 X, yt, lambda_parameter)

print('\n  Cost at parameters (loaded from ex4weights): ', J)
print('  (this value should be about 0.383770.)')

#part 5: Sigmoid Gradient
print('\nPart 5: Sigmoid Gradient')
print('\n  Evaluating sigmoid gradient ...')

test_points = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
g = nn.sigmoid_gradient(test_points)

print('\n  Sigmoid gradieent evaluated at [-1, -0.5, 0, 0.5, 1]:', g)

#part 6: Initialiiziing Parameters
print('\nPart 6: Initializiing Neural Network Parameters ...')

initial_theta1 = nn.rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_theta2 = nn.rand_initialize_weights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate((initial_theta1.flatten(), initial_theta2.flatten()), axis = 0)

#part 7: Implement Backpropagation
print('\nPart 7: Checking Backpropagation ...')
nn.check_nn_gradient()

#part 8: Implement Regularization
print('\nPart 8: Checking Backpropagation with Regularization ...')
lambda_parameter = 3.0
nn.check_nn_gradient(lambda_parameter)

debug_J, debug_grad = nn.cost_func(nn_params, input_layer_size, hidden_layer_size, num_labels, X, yt, lambda_parameter)
print('\n  Cost function at (fixed) debugging parameters with lambda = 3.0 is: ', debug_J)
print('\n  This value should be 0.576051.')

#part 9: Training NN
print('\nPart 9: Training Neural Network ...')

lambda_parameter = 1.0
nn_params, cost = nn.train(input_layer_size, hidden_layer_size, num_labels, X, yt, initial_nn_params, lambda_parameter)

theta1 = nn_params[0:(input_layer_size + 1) * hidden_layer_size].reshape((hidden_layer_size, input_layer_size + 1)) 
theta2 = nn_params[(input_layer_size + 1) * hidden_layer_size:].reshape((num_labels, hidden_layer_size + 1))


#part 10: Implement Prediction
pred = nn.predict(theta1, theta2, X)

print('\nPart 10: Training Set Accuracy: ', np.mean((pred == y) * 1.0))                      



