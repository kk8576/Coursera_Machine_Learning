#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:13:44 2021

@author: wangkun
"""

import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt 
import linear
import feature

input_layer_size = 400 
hidden_layer_size = 25
num_labels = 10

# Part 1: Loading and visualizing data
print('\nPart1: Loading and Visualizing Data ...\n')

# load matlab mat format data from the file ex3data1.mat
data = loadmat('ex5data1.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

m = X.shape[0]

plt.plot(X, y, 'rx')
plt.xlabel('Change in wateer level (x)')
plt.ylabel('Watre flowing cout of th edam (y)')
plt.show()

#Part 2: Regularized Linear Regression Cost
theta = np.array([1.0, 1.0])

J, grad = linear.cost_func(theta, X, y, 1.0)

print('\nPart 2: Regularized Linear Regression Cost ...')
print('  Cost function at [1 1]: ', J)
print('    this value should be 303.993192')
print('  Gradient: ', grad)
print('    this value should be [-15.303016; 598.250744]')

#Part 3: Train Linear Regression
lambda_parameter = 1.0
theta = linear.train(X, y, lambda_parameter)

plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
yy = np.sum(np.concatenate((np.ones((m, 1)), X), axis = 1) * theta[0], axis = 1)
plt.plot(X, yy, 'r-', color='b')
plt.show()

#Part 4: Learning curve: error vs. size of training data
J_train = np.zeros(m-1)
J_cv = np.zeros(m-1)

for i in range(1,m):
    theta, _ = linear.train(X[:i+1], y[:i+1], 0.0)
    
    J_train0, _ = linear.cost_func(theta, X[:i+1], y[:i+1], 0.0)
    J_cv0, _ = linear.cost_func(theta, Xval, yval, 0.0)
    
    J_train[i-1] = J_train0
    J_cv[i-1] = J_cv0
    
plt.plot(np.arange(1,m) + 1, J_train, 'b-', label='Training')
plt.plot(np.arange(1,m) + 1, J_cv, 'g-', label='Cross Validation')
plt.legend()
plt.xlabel('Size of Training Set')
plt.ylabel('Error')
plt.show()

#Part 5: Feature Mapping for Polynomial Regression
p= 8
X_poly = feature.polynomial_features(X, p)
X_poly, mu, sigma = feature.normalize_features(X_poly)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = feature.polynomial_features(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = feature.polynomial_features(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma

print('Normalized Training Example 1:\n', X_poly[0])

#Part 6: Train polynomial regression and plot learning curve
lambda_parameter = 0.0
theta = linear.train(X_poly, y, lambda_parameter)

x = np.arange(X.min() - 15.0, X.max() + 15.0, 0.05).reshape((-1,1))
x_poly = feature.polynomial_features(x, p)
x_poly = (x_poly - mu) / sigma
y_poly = np.sum(np.concatenate((np.ones((x_poly.shape[0], 1)), x_poly), axis = 1) * theta[0], axis = 1)
plt.plot(x, y_poly, 'r--')
plt.plot(X, y, 'b*')
plt.title('Polynomial Fit, lambda = 0')
plt.show()

J_train = np.zeros(m-1)
J_cv = np.zeros(m-1)

for i in range(1,m):
    theta, _ = linear.train(X_poly[:i+1], y[:i+1], 0.0)
    
    J_train0, _ = linear.cost_func(theta, X_poly[:i+1], y[:i+1], 0.0)
    J_cv0, _ = linear.cost_func(theta, X_poly_val, yval, 0.0)
    
    J_train[i-1] = J_train0
    J_cv[i-1] = J_cv0
    
plt.plot(np.arange(1,m) + 1, J_train, 'b-', label='Training')
plt.plot(np.arange(1,m) + 1, J_cv, 'g-', label='Cross Validation')
plt.legend()
plt.xlabel('Size of Training Set')
plt.ylabel('Error')
plt.show()

#Part 7: Select regularization lambda
lambda_parameter = 1.0
theta = linear.train(X_poly, y, lambda_parameter)

y_poly = np.sum(np.concatenate((np.ones((x_poly.shape[0], 1)), x_poly), axis = 1) * theta[0], axis = 1)
plt.plot(x, y_poly, 'r--')
plt.plot(X, y, 'b*')
plt.title('Polynomial Fit, lambda = 1')
plt.show()

lambda_parameter = 100.0
theta = linear.train(X_poly, y, lambda_parameter)

y_poly = np.sum(np.concatenate((np.ones((x_poly.shape[0], 1)), x_poly), axis = 1) * theta[0], axis = 1)
plt.plot(x, y_poly, 'r--')
plt.plot(X, y, 'b*')
plt.title('Polynomial Fit, lambda = 100')
plt.show()

lambda_parameters = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
lambda_size = len(lambda_parameters)
J_train = np.zeros(lambda_size)
J_cv = np.zeros(lambda_size)
i = 0
lambda_op = -1.0
J_cv_min = 1e10
theta_op = np.array([])

for l in lambda_parameters:
    theta, _ = linear.train(X_poly, y, l)
    
    J_train0, _ = linear.cost_func(theta, X_poly, y, 0.0)
    J_cv0, _ = linear.cost_func(theta, X_poly_val, yval, 0.0)
    
    J_train[i] = J_train0
    J_cv[i] = J_cv0
    i = i+1
    
    if J_cv_min > J_cv0:
        J_cv_min = J_cv0
        lambda_op = l
        theta_op = theta
    
plt.plot(lambda_parameters, J_train, 'b-*', label = 'Training')
plt.plot(lambda_parameters, J_cv, 'g-*', label = 'Cross Validation')
plt.ylim([0,20])
plt.xlim([0, 10])
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.legend()
plt.show()
print('  The optimal lambda is:', lambda_op)
print('  The cross validation value is:', J_cv_min)

J_test, _ = linear.cost_func(theta_op, X_poly_test, ytest, 0.0)
print('  The test error is:', J_test)

    