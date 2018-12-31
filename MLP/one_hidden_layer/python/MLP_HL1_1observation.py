#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 14 2018
@author: lpupp

This script implements a simple neural network with 1 hidden layer with numpy
and python base functions. We only pass in a single observation to be able to
investigate the mechanics of a neural network without unnecessary complixity.
We perform only one iteration of forward and backpropagation. The hidden layer
nodes and the output node are activated with sigmoid functions.

MLP structure -----------------------------------------------------------------
Input Layer (IN): 2 Nodes
Hidden Layer (HL): 3 Nodes
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
# Learning rate
alpha = 0.5

# Helper functions ------------------------------------------------------------
# Activation function (sigmoid)
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
    return np.sum(np.square((y - y_hat)))

# Training data ---------------------------------------------------------------
# Set inputs (x1, x2)
X = np.array([[-0.9687464, -0.8026618]])

# Labels
y = np.array([[0]])

# Network initialization ------------------------------------------------------
# Initialize weights and biases
w_IN2HL = np.array([[-0.2, 0.4, -1.0],
                    [-0.4, -0.7, -0.8]])
print(w_IN2HL)

w_HL2ON = np.array([[-0.6], [-0.6], [-0.3]])
print(w_HL2ON)

b_HL = np.array([[0.1, -0.2,  0.4]])
b_ON = np.array([-0.6])

# Feed forward ----------------------------------------------------------------
l_IN = X
l_HL = f_sigmoid(np.dot(l_IN, w_IN2HL) + b_HL)
l_ON = f_sigmoid(np.dot(l_HL, w_HL2ON) + b_ON)

# Backpropagation -------------------------------------------------------------
e_ON = l_ON - y  # Error (ON)
bpe_ON = e_ON * f_sigmoid_derivative(l_ON)  # Backpropagation error (ON)

e_HL = bpe_ON.dot(w_HL2ON.T)  # Error (HL)
bpe_HL = e_HL * f_sigmoid_derivative(l_HL)  # Backpropagation error (HL)

# Update the weights ----------------------------------------------------------
w_HL2ON -= alpha * l_HL.T.dot(bpe_ON)
w_IN2HL -= alpha * l_IN.T.dot(bpe_HL)

b_ON -= alpha * np.sum(bpe_ON, axis=0)
b_HL -= alpha * np.sum(bpe_HL, axis=0)
