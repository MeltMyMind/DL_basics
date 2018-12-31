#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2018
@author: lpupp

This script implements a simple neural network with 2 hidden layers with numpy
and python base functions. The hidden layer nodes and the output node are
activated with sigmoid functions. The neural network is trained using gradient
descent.

MLP structure -----------------------------------------------------------------
Input Layer (IN): 2 Nodes
Hidden Layer 1 (HL1): 3 Nodes
Hidden Layer 2 (HL2): 3 Nodes
Output Layer (ON): 1 Node

Variable key ------------------------------------------------------------------
l: layer
w: weight
b: bias
f: function
d: (undefined) data structure
bpe: backpropagation error
e: error

Sources -----------------------------------------------------------------------
https://iamtrask.github.io/2015/07/12/basic-python-network/
"""

# Import dependencies ---------------------------------------------------------
import numpy as np

# Constants and training hyper-parameters -------------------------------------
# Learning rate and max error
alpha = 0.5
maxError = 0.001
n_iter = 1

# Helper functions ------------------------------------------------------------
# Activation function (sigmoid function)
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def f_sigmoid_derivative(x):
    """Derivative of the sigmoid activation.

    Note that the true derivative of the sigmoid function is:
        f_sigmoid(z) * (1 - f_sigmoid(z))
    However, in this function we define:
        x := f_sigmoid(z).
    """
    return x * (1 - x)

# Quadratic cost function
def f_cost(y, y_hat):
    return np.sum(np.square(y - y_hat) * 0.5)

# Training data ---------------------------------------------------------------
# Set inputs (x1, x2)
X = np.array([[-0.9687464, -0.8026618],
              [-0.6869292, -0.9746607],
              [-1.1448821, -1.0893267],
              [-0.9335192, -1.0893267],
              [ 0.2642036,  0.6306628],
              [ 1.1448821,  1.0893267],
              [ 0.7573835,  0.9173277],
              [ 1.3210178,  1.4906575],
              [-0.7926107, -0.5159968],
              [ 1.0392006,  0.3439979]])

# Labels (y)
y = np.array([[0],
              [0],
              [0],
              [0],
              [1],
              [1],
              [1],
              [1],
              [1],
              [0]])

# Network initialization ------------------------------------------------------
# Initialize weights
w_IN2HL1 = np.array([[-0.2,  0.4, -1.0],
                     [-0.4, -0.7, -0.8]])
#print(w_IN2HL1)

w_HL12HL2 = np.array([[0.1, -0.5,  0.8],
                      [0.9,  0.4,  0.2],
                      [0.1,  0.7, -0.7]])
#print(w_HL12HL2)

w_HL22ON = np.array([[-0.6], [-0.6], [-0.3]])
#print(w_HL22ON)

# Initialize bias
b_HL1 = np.array([[0.1, -0.2,  0.4]])
b_HL2 = np.array([[1.0, -0.2,  0.7]])
b_ON = np.array([-0.6])

# Training --------------------------------------------------------------------
# Loop n_iter times
for i in range(n_iter):

    # Feed forward ------------------------------------------------------------
    l_IN = X
    l_HL1 = f_sigmoid(np.dot(l_IN, w_IN2HL1) + b_HL1)
    l_HL2 = f_sigmoid(np.dot(l_HL1, w_HL12HL2) + b_HL2)
    l_ON = f_sigmoid(np.dot(l_HL2, w_HL22ON) + b_ON)

    # Backpropagation ---------------------------------------------------------
    e_ON = (l_ON - y)  # Error (ON)
    bpe_ON = e_ON * f_sigmoid_derivative(l_ON)  # Backpropagation error (ON)

    e_HL2 = bpe_ON.dot(w_HL22ON.T)  # Error (HL2)
    bpe_HL2 = e_HL2 * f_sigmoid_derivative(l_HL2)  # Backpropagation error (HL2)

    e_HL1 = bpe_HL2.dot(w_HL12HL2.T)  # Error (HL1)
    bpe_HL1 = e_HL1 * f_sigmoid_derivative(l_HL1)  # Backpropagation error (HL1)

    # Update ------------------------------------------------------------------
    # Update the weights
    w_HL22ON -= alpha * l_HL2.T.dot(bpe_ON)
    w_HL12HL2 -= alpha * l_HL1.T.dot(bpe_HL2)
    w_IN2HL1 -= alpha * l_IN.T.dot(bpe_HL1)

    # Update the bias
    b_ON -= alpha * np.sum(bpe_ON, axis=0)
    b_HL2 -= alpha * np.sum(bpe_HL2, axis=0)
    b_HL1 -= alpha * np.sum(bpe_HL1, axis=0)

    # Control -----------------------------------------------------------------
    # Print the error to show that we are improving
    if (i % 1000) == 0:
        print('Error after {} iterations: {}'.format(i, f_cost(y, l_ON)))

    # Exit if the error is less than maxError
    if(f_cost(y, l_ON) < maxError):
        print('Goal reached after {} iterations: {} is smaller than the goal of {}'.format(i, f_cost(y, l_ON), maxError))
        break

# Show results ----------------------------------------------------------------
print('\nWeights between IN -> HL1')
print(w_IN2HL1)

print('\nBias of HL')
print(b_HL1)

print('\nWeights between HL1 -> HL2')
print(w_HL12HL2)

print('\nBias of HL2')
print(b_HL2)

print('\nWeights between HL2 -> ON')
print(w_HL22ON)

print('\nBias of ON')
print(b_ON)

print('\nComputed probabilities (rounded to 3 decimals)')
print(np.around(l_ON, decimals=3))

print('\nlabels')
print(y)

print('\nFinal Error')
print(f_cost(y, l_ON))
print(np.around(f_cost(y, l_ON), 2))
