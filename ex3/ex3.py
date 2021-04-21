#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python implementation of the Ex3 of the Coursera Machine Learning

@author: wangkun
"""
import numpy as np
from scipy.io import loadmat
import random
import display_data
import lr_cost_function
import one_vs_all

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

## ============ Part 2a: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2]).reshape((4, 1))
X_t = np.concatenate((np.ones((5,1)), np.arange(1, 16).reshape([3, 5]).transpose() /10.0), axis = 1)
#y_t = np.array([1, 0, 1, 0, 1]).reshape(5,1) > 0.5;
y_t = np.array([1.0, 0.0, 1.0, 0.0, 1.0]).reshape(5,1);

lambda_t = 3;
J = lr_cost_function.cost_function(theta_t, X_t, y_t, lambda_t);
grad = lr_cost_function.cost_function_jac(theta_t, X_t, y_t, lambda_t);


print('\nCost: ', J);
print('Expected cost: 2.534819\n');
print('Gradients:\n');
print(grad);
print('Expected gradients:\n');
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

wait = input('Program paused. Press enter to continue.\n');

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

parameter_lambda = 3.0;
all_theta = one_vs_all.train(X, y, num_labels, parameter_lambda);


wait = input('Program paused. Press enter to continue.\n');

# ================ Part 3: Predict for One-Vs-All ================

pred = one_vs_all.predict(all_theta, X)

print('\nTraining Set Accuracy (%): ', np.mean((pred == y)) * 100.0);

rand_example_ind = random.sample(range(m), m)
for i in range(m):
    print('\nDisplay Example Image\n')
    display_data.display_data(X[rand_example_ind[i], :].reshape((1, X.shape[1])))
    
    pred0 = one_vs_all.predict(all_theta, X[rand_example_ind[i], :].reshape((1, X.shape[1])))
    print('\nOne-vs-All Prediction: ', pred0)
    print('\nY lable: ', y[rand_example_ind[i]])
    
    s = input('\nPaused - press enter to continue, q to exit:')
    if s == 'q':
        break





