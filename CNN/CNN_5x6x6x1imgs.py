#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 29 2018
@author: lucagaegauf

One iteration of forward and backpropagation with 1 observation.

CNN structure -----------------------------------------------------------------
Input layer (IN): 6x6x1 (grey-scale) image
Activation map (AM): 4x4x1 nodes
Fully-connected layer (FC): 3 nodes
Output layer (ON): 1 Node

Variable key ------------------------------------------------------------------
l:    layer
k:    kernel
w:    weight
b:    bias
rf:   receptive field
ix:   index
f:    function
bpe:  backpropagation error
e:    error
d:    (undefined) data structure

Sources -----------------------------------------------------------------------
https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4
https://trinket.io/python3/b195e28e24
"""
#%% Import dependencies -------------------------------------------------------
import numpy as np
from scipy import signal

#%% Define CNN training hyper-parameters --------------------------------------
# Learning rate and max error
alpha = 0.5
maxError = 0.001

#%% Define helper functions ---------------------------------------------------
# Activation function
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_relu(x):
    return x * (x > 0)

# Derivatives of activation functions
def f_sigmoid_derivative(x):
    return x * (1 - x)

def f_relu_derivative(x):
    return (x > 0) * 1

# Quadratic cost function
def f_cost(y, y_hat):
    return np.sum(np.square((y - y_hat))) * 0.5

# Convolution function
def f_convolve(X, kernel, bias):
    
    img_dim  = X.shape[0]
    k_dim    = kernel.shape[0]
    conv_dim = img_dim - k_dim + 1
    
    # Assign memory for the feature map
    convolved_image = np.zeros((conv_dim, conv_dim))
    
    # Convolve image with the feature and add to existing matrix
    for ix_x in range(conv_dim):
        for ix_y in range(conv_dim):
            X_rf = X[ix_y:(ix_y + k_dim), ix_x:(ix_x + k_dim)]
            convolved_image[ix_y, ix_x] = np.sum(X_rf * kernel)
    
    # Take ReLU transform and store
    activation_map = f_relu(convolved_image + bias)
    
    return activation_map 

#f_convolve(X, w_AM, b_AM)

#%% Process data --------------------------------------------------------------
# Set inputs
x1 = np.array([[255, 151, 122,  91,  61,  24],
               [255, 169,  69,  68,  59,   4],
               [255, 235, 203, 156, 111,  11],
               [255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255]])
    
x2 = np.array([[255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255],
               [255, 255, 210, 126,  97,  18],
               [255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255]])

x3 = np.array([[255, 255, 255, 255, 255, 255],
               [255, 255, 255, 255, 255, 255],
               [255, 133,  50, 101, 143, 255],
               [255,  93,  50, 110, 118, 255],
               [255, 175,   4,  64,  94, 255],
               [255, 255, 255, 255, 255, 255]])
    
x4 = np.array([[255, 255, 255, 255, 255, 255],
               [255, 195, 243, 233, 255, 255],
               [255, 159, 236, 227, 255, 255],
               [255, 142, 219, 164, 255, 255],
               [255, 105, 114, 158, 255, 255],
               [255, 102, 107,   9, 255, 255]])
    
x5 = np.array([[255, 255, 255, 255, 255, 255],
               [112,  55, 101, 255, 255, 255],
               [ 86,  22,  27, 255, 255, 255],
               [ 41,  38, 109, 255, 255, 255],
               [227, 143, 170, 255, 255, 255],
               [255, 255, 255, 255, 255, 255]])
    
X = np.array([x1, x2, x3, x4, x5])

# Normalize input
X = np.around(2 * (X / 255) - 1, 2)

# Labels
y = np.array([[1],
              [1],
              [1],
              [0],
              [0]])

#%% Initialize weights --------------------------------------------------------
w_AM = np.array([[-0.8, 0.9, -0.9],
                [0.5, 0.6, 0.1],
                [0.8, -0.2, 0.4]])

print(w_AM)

w_AM2FC = np.array([[-0.9, -0.4,  0.3],
                    [-0.6, -0.1, -0.9],
                    [ 0.1, -0.1,  0.1],
                    [ 0.8,  0.6,  0.6],
                    [ 0.9, -0.9, -0.1],
                    [ 0.3, -0.7, -0.9],
                    [-0.7,  0.1,  0.3],
                    [-0.5,  0.4,  0.2],
                    [-0.1, -0.9,  0.6],
                    [ 0.6,  0.2, -0.8],
                    [-0.5,  0.7,  0.9],
                    [-0.2, -0.5,  1.5],
                    [-0.9, -0.6,  1.2],
                    [ 0.3, -0.1,  0.4],
                    [-0.6,  0.5,  0.9 ],
                    [-0.2,  0.6,  0.4]])
# Initialize bias
b_AM = np.array([[0.7]])
b_FC = np.array([[-0.7, -0.6,  0.2]])
w_FC2ON = np.array([[-0.6], [-0.6], [-0.3]])
b_ON = np.array([[-0.6]])

#%% Train CNN -----------------------------------------------------------------
# Loop 10,000 times
n_iter = 1
for i in range(n_iter):

    # Feed forward ------------------------------------------------------------
    l_IN = X

    l_AM = np.zeros((5, 4, 4))
    for img in range(X.shape[0]):
        l_AM[img] = f_convolve(l_IN[img], w_AM, b_AM)
        
    flat_shape = l_AM.shape[1] * l_AM.shape[2]
    l_AM_flat = np.reshape(l_AM, (l_AM.shape[0], flat_shape))
        
    l_FC = f_sigmoid(np.dot(l_AM_flat, w_AM2FC) + b_FC)
        
    l_ON = f_sigmoid(np.dot(l_FC, w_FC2ON) + b_ON)
    
    # Backpropagation ---------------------------------------------------------
    e_ON = l_ON - y  # Error (ON)
    bpe_ON = e_ON * f_sigmoid_derivative(l_ON)  # Backpropagation error (ON)
    
    e_FC = bpe_ON.dot(w_FC2ON.T)  # Error (FC)
    bpe_FC = e_FC * f_sigmoid_derivative(l_FC)  # Backpropagation error (FC)
    
    e_AM = bpe_FC.dot(w_AM2FC.T)  # Error (K)
    e_AM_reshape = np.reshape(e_AM, (5, 4, 4)) # Reshape error to activation map dims
    bpe_AM = e_AM_reshape * f_relu_derivative(l_AM)  # Backpropagation error (HL)
  
    # Update ------------------------------------------------------------------
    # Update the weights
    w_FC2ON -= alpha * l_FC.T.dot(bpe_ON)
    w_AM2FC -= alpha * l_AM_flat.T.dot(bpe_FC)
    
    w_AM_grad = np.zeros((3, 3))
    for img in range(X.shape[0]):
        w_AM_grad += signal.convolve2d(l_IN[img], np.rot90(bpe_AM[img], 2), 'valid')
    w_AM    -= alpha * w_AM_grad
    
    # Update the bias    
    b_ON -= alpha * np.sum(bpe_ON, axis=0)
    b_FC -= alpha * np.sum(bpe_FC, axis=0)
    b_AM -= alpha * np.sum(bpe_AM)
        
    # Control -----------------------------------------------------------------
    # Print the error to show that we are improving
    if (i% 1000) == 0:
        print("Error after " + str(i) + " iterations: " + str(f_cost(y, l_ON)))
        
    # Exit if the error is less than maxError
    if(f_cost(y, l_ON) < maxError):
        print("Goal reached after " + str(i) + " iterations: " + str(f_cost(y, l_ON)) + " is smaller than the goal of " + str(maxError))
        break
    
# Show results
print("")
print("Weights between IN -> AM")
print(w_AM)
print("")

print("Bias of AM")
print(b_AM)
print("")

print("")
print("Weights between AM -> FC")
print(w_AM2FC)
print("")

print("Bias of FC")
print(b_FC)
print("")

print("Weights between FC -> ON")
print(w_FC2ON)
print("")

print("Bias of ON")
print(b_ON)
print("")

print("Computed probabilities (rounded to 3 decimals)")
print(np.around(l_ON, decimals=3))
print("")

print("labels")
print(y)
print("")

print("Final Error")
print(str(f_cost(y, l_ON)))
print(str(np.around(f_cost(y, l_ON), 2)))

print("Final Kernel Weights")
print(w_AM)



