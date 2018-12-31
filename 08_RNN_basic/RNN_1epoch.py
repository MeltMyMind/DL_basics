#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 15 2018
@author: lpupp

This script implements a simple recurrent neural network with 1 RNN unit with 3
hidden nodes using numpy and python base functions. We only conduct in a single
epoch to be able to investigate the mechanics of a RNN without unnecessary
complixity. The RNN unit is tanH activated and the output node is activated
with the identity function (linear activation).

RNN structure -----------------------------------------------------------------
Input Layer (x): 1 node
RNN (h): 3 hidden states (tanh activation)
Output Layer (y): 1 node (linear activation)

Variable key ------------------------------------------------------------------
w:   weight
b:   bias
f:   function
e:   error
d:   derivative
bpe: backpropagation error
t:   time index

Sources -----------------------------------------------------------------------
https://jramapuram.github.io/ramblings/rnn-backrpop/
http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/
https://gist.github.com/karpathy/d4dee566867f8291f086
"""
# Import dependencies ---------------------------------------------------------
import numpy as np

# Constants and training hyper-parameters -------------------------------------
alpha = 1e-1        # Learning rate
n_hidden_state = 3  # Number of hidden states

# Helper functions ------------------------------------------------------------
def f_dtanh(x):
    return 1 - (x ** 2)

def f_identity(x):
    return x

def f_didentity(x):
    return 1

# Training data ---------------------------------------------------------------
data = np.array([11, 0, 5, 5])
data_length = len(data)
print('The sequence is {} time steps.'.format(data_length))

# Normalize data
data_mean = np.mean(data, axis=0)
data_sd = np.std(data, axis=0, ddof=1)
data = (data - data_mean) / data_sd
print(data)

# Define input and targets
inputs = data[:-1]
targets = data[1:]

# Length of unrolled RNN
seq_length = len(inputs)

# Network initialization ------------------------------------------------------
# Initialize weights
w_xh = np.array([[-0.2], [0.4], [-1.0]])
print(w_xh)

w_hh = np.array([[-0.4, -0.7, -0.8],
                 [ 0.1, -0.5,  0.8],
                 [ 0.9,  0.4,  0.2]])
print(w_hh)

w_hy = np.array([[-0.6, -0.6, -0.3]])
print(w_hy)

# Initialize bias
b_h = np.array([[0.1], [-0.2], [0.4]])
b_y = np.array([[-0.6]])

# Feed forward with dicts -----------------------------------------------------
# Create placeholders
hprev = np.zeros((n_hidden_state, 1))

xs, hs, h_uns, ys, loss = {}, {}, {}, {}, {}
hs[-1] = hprev
h_uns[-1] = hprev
loss_sum = 0

dw_xh, dw_hh, dw_hy = np.zeros_like(w_xh), np.zeros_like(w_hh), np.zeros_like(w_hy)
db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)

# Feed forward ----------------------------------------------------------------
for t in range(len(inputs)):
    xs[t] = inputs[t]
    h_uns[t] = np.dot(w_xh, xs[t]) + np.dot(w_hh, hs[t-1]) + b_h  # Unactivated (for slides)
    hs[t] = np.tanh(np.dot(w_xh, xs[t]) + np.dot(w_hh, hs[t-1]) + b_h)  # Hidden state
    ys[t] = f_identity(np.dot(w_hy, hs[t]) + b_y)  # RNN output node
    loss[t] = 0.5 * np.square(ys[t] - targets[t])  # Output loss
    loss_sum += loss[t] / seq_length

# Backpropagation through time ------------------------------------------------
for t in np.arange(seq_length)[::-1]:
    e_y_t = ys[t] - targets[t]            # Error (y): dL/dyhat
    bpe_y_t = e_y_t * f_didentity(ys[t])  # BPE (y):

    dw_hy += bpe_y_t * hs[t].T  # Gradient (w_hy): dL/dyhat * dyhat/dWhy
    db_y += bpe_y_t             # Gradient (b_y): dL/dyhat * dyhat/dby

    e_h_t = w_hy.T.dot(bpe_y_t)       # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
    bpe_h_t = e_h_t * f_dtanh(hs[t])  # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)

    for bptt_step in np.arange(t+1)[::-1]:

        # Gradient (w_xh): dL_t/d(h_{t-1} Whh) * d(h_{t-1} Whh)/dWhh
        dw_hh += np.outer(bpe_h_t, hs[bptt_step-1])

        dw_xh += bpe_h_t * xs[bptt_step]  # Gradient (w_xh): dL_t/d(x_t Wxh) ** d(x_t Wxh) / dWxh
        db_h += bpe_h_t           # Gradient (b_h): dL_t/d(x_t Wxh)

        # Update bpe_h for next step
        e_h_t = w_hh.T.dot(bpe_h_t)              # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
        bpe_h_t = e_h_t * f_dtanh(hs[bptt_step-1])  # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)

# Update ----------------------------------------------------------------------
# Update the weights
w_hy -= alpha * dw_hy
w_hh -= alpha * dw_hh
w_xh -= alpha * dw_xh

# Update the biases
b_h -= alpha * db_h
b_y -= alpha * db_y

# Check weights ---------------------------------------------------------------
print('\nWeights between x -> h')
print(w_xh)

print('\nWeights between h -> h')
print(w_hh)

print('\nBias of h')
print(b_h)

print('\nWeights between h -> y')
print(w_hy)

print('\nBias of y')
print(b_y)
