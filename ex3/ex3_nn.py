#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:36:53 2021

@author: wangkun
"""

import numpy as np
from scipy.io import loadmat
import random
import display_data
import nn

input_layer_size = 400
num_labels = 10

print('\n  Loading and Visualizing Data ...')

# load matlab mat format data from the file ex3data1.mat
data = loadmat('/Users/wangkun/reference/ML/Coursera ML/machine-learning-ex3/ex3/ex3data1.mat')

X = data['X']
y = data['y']

print('\n  The number of training examples: ', X.shape[0])
m = X.shape[0]

# Randomly select 100 data points to display
print('\n  Rondomly select 100 examples to display')
rand_indices = random.sample(range(m),100)

sel = X[rand_indices, :]

display_data.display_data(sel)

wait = input('\nProgram paused. Press <ENTER> to continue')

print('Loading Saved Neural Network Parameters ...')

weights = loadmat('/Users/wangkun/reference/ML/Coursera ML/machine-learning-ex3/ex3/ex3weights.mat')
theta1 = weights['Theta1']
theta2 = weights['Theta2']

pred = nn.predict([theta1, theta2], X)

print('\nTraining Set Accuracy (%): ', np.mean((pred.reshape((pred.shape[0],1)) == y)) * 100.0)

rand_example_ind = random.sample(range(m), m)
for i in range(m):
    print('\nDisplay Example Image\n')
    display_data.display_data(X[rand_example_ind[i], :].reshape((1, X.shape[1])))
    
    pred0 = nn.predict([theta1, theta2], X[rand_example_ind[i], :].reshape((1, X.shape[1])))
    print('\nNeural Network Prediction: ', pred0)
    print('\nY lable: ', y[rand_example_ind[i]])
    
    s = input('\nPaused - press enter to continue, q to exit:')
    if s == 'q':
        break
    