#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:57:36 2021

Display 2D data in a nice grid

@author: wangkun
"""

import matplotlib.pyplot as plt
import numpy as np

def display_data(X):
    
    m = X.shape[0]
    n = X.shape[1]
    
    # Compuate rows, cols
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    dim = int(np.round(np.sqrt(n)))
    
        
    curr_ex = 0
    
    fig, ax = plt.subplots(display_rows, display_cols)

    
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex > m:
                break
                        
            if display_rows == 1 and display_cols == 1:
                ax.imshow(np.transpose(X[curr_ex].reshape(dim, dim)), cmap = 'gray')
                ax.axis('off')
            else:
                ax[i, j].imshow(np.transpose(X[curr_ex].reshape(dim, dim)), cmap = 'gray')
                ax[i, j].axis('off')
                
            curr_ex += 1
    plt.subplots_adjust()

    plt.show()

