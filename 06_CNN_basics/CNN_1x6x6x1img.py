#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 29 2018
@author: lpupp

This script implements a simple convolutional neural network with 1
convolutional layer and 1 fully connected layer with numpy and python base
functions. We only pass in a single image to be able to investigate the
mechanics of a CNN without unnecessary complixity. We perform only one
iteration of forward and backpropagation. The activation map is activated with
ReLU and the output node are activated with sigmoid functions.

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
# Import dependencies ---------------------------------------------------------
import numpy as np

# Constants and training hyper-parameters -------------------------------------
# Learning rate
alpha = 0.5

# Helper functions ------------------------------------------------------------
# Activation functions
# Sigmoid function
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Rectified linear unit
def f_relu(x):
    return x * (x > 0)

# Derivatives of activation functions
def f_sigmoid_derivative(x):
    """Derivative of the sigmoid activation.

    Note that the true derivative of the sigmoid function is:
        f_sigmoid(z) * (1 - f_sigmoid(z))
    However, in this function we define:
        x := f_sigmoid(z).
    """
    return x * (1 - x)

def f_relu_derivative(x):
    return (x > 0) * 1

# Quadratic cost function
def f_cost(y, y_hat):
    return np.sum(np.square((y - y_hat))) * 0.5

# Convolution function
def f_convolve(X, kernel, bias):
    """Perform a non-activation CNN convolution on input X.

    Args:
        X: input data.
        kernel: CNN weights.
        bias: CNN bias.

    Returns:
        Pre-activation activation map.

    e.g.: f_convolve(X, w_AM, b_AM)

    Note about the convolution function: The convolution operation used in
    convulational neural networks is not the technically a convolution operation
    used in mathematics but rather a cross-correlation operation. So, if you are
    using the convolution function from scipy.signal you will not get the same
    results. We use:
        from scipy import signal
        signal.convolve2d(X, np.rot90(kernel, 2), 'valid')
    """

    img_dim = X.shape[0]
    k_dim = kernel.shape[0]
    conv_dim = img_dim - k_dim + 1

    # Assign memory for the feature map
    convolved_image = np.zeros((conv_dim, conv_dim))

    # Convolve image with the feature and add to existing matrix
    for ix_x in range(conv_dim):
        for ix_y in range(conv_dim):
            X_rf = X[ix_y:(ix_y + k_dim), ix_x:(ix_x + k_dim)]
            convolved_image[ix_y, ix_x] = np.sum(X_rf * kernel)

    return convolved_image + bias

# Training data ---------------------------------------------------------------
# Set inputs (6x6x1 grey scale image)
X = np.array([[255, 151, 122,  91,  61,  24],
              [255, 169,  69,  68,  59,   4],
              [255, 235, 203, 156, 111,  11],
              [255, 255, 255, 255, 255, 255],
              [255, 255, 255, 255, 255, 255],
              [255, 255, 255, 255, 255, 255]])

# Normalize input
X = 2 * (X / 255) - 1

# Labels (y)
y = np.array([[1]])

# Network initialization ------------------------------------------------------
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
                    [-0.6,  0.5,  0.9],
                    [-0.2,  0.6,  0.4]])
print(w_AM2FC)

w_FC2ON = np.array([[-0.6], [-0.6], [-0.3]])
print(w_FC2ON)

# Initialize bias
b_AM = np.array([[0.7]])
b_FC = np.array([[-0.7, -0.6,  0.2]])
b_ON = np.array([[-0.6]])

# Feed forward ----------------------------------------------------------------
l_IN = X
l_AM_input = f_convolve(l_IN, w_AM, b_AM)
l_AM = f_relu(l_AM_input)
print(np.around(l_AM, 2))

flat_shape = l_AM.shape[0] * l_AM.shape[1]
l_AM_flat = np.reshape(l_AM, (1, flat_shape))
print(np.around(l_AM_flat, 2))

l_FC = f_sigmoid(np.dot(l_AM_flat, w_AM2FC) + b_FC)
print(np.around(l_FC, 2))

l_ON = f_sigmoid(np.dot(l_FC, w_FC2ON) + b_ON)
print(np.around(l_ON, 2))

# CNN cost --------------------------------------------------------------------
cost = f_cost(y, l_ON)

# Backpropagation -------------------------------------------------------------
e_ON = l_ON - y  # Error (ON)
print(np.around(e_ON, 3))
bpe_ON = e_ON * f_sigmoid_derivative(l_ON)  # Backpropagation error (ON)
print(np.around(bpe_ON, 3))

e_FC = bpe_ON.dot(w_FC2ON.T)  # Error (FC)
print(np.around(e_FC, 3))
bpe_FC = e_FC * f_sigmoid_derivative(l_FC)  # Backpropagation error (FC)
print(np.around(bpe_FC, 3))

e_AM = bpe_FC.dot(w_AM2FC.T)  # Error (AM)
e_AM_reshape = np.reshape(e_AM, (4, 4)) # Reshape error to activation map dims
print(np.around(e_AM, 3))
bpe_AM = e_AM_reshape * f_relu_derivative(l_AM_input)  # Backpropagation error (HL)
print(np.around(bpe_AM, 5))

# Update ----------------------------------------------------------------------
# Update the weights
w_FC2ON -= alpha * l_FC.T.dot(bpe_ON)
w_AM2FC -= alpha * l_AM_flat.T.dot(bpe_FC)
w_AM -= alpha * f_convolve(l_IN, bpe_AM, np.array([[0]]))

# Update the bias
b_ON -= alpha * np.sum(bpe_ON, axis=0)
b_FC -= alpha * np.sum(bpe_FC, axis=0)
b_AM -= alpha * np.sum(bpe_AM)
