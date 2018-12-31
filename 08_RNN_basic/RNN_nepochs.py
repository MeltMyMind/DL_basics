#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 15 2018
@author: lpupp

This script implements a simple recurrent neural network with 1 RNN unit with 3
hidden nodes using numpy and python base functions. The RNN unit is tanH
activated and the output node is activated with the identity function
(linear activation).

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
# Learning rate and max error
alpha = 1e-1
maxError = 1
n_iter = 1000
n_hidden_state = 3

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
# Weights
w_xh = np.array([[-0.2], [0.4], [-1.0]])
#print(w_xh)

w_hh = np.array([[-0.4, -0.7, -0.8],
                 [ 0.1, -0.5,  0.8],
                 [ 0.9,  0.4,  0.2]])
#print(w_hh)

w_hy = np.array([[-0.6, -0.6, -0.3]])
#print(w_hy)

# Biases
b_h = np.array([[0.1], [-0.2],  [0.4]])
b_y = np.array([[-0.6]])

# Training --------------------------------------------------------------------
for i in range(n_iter):
    # Create placeholders
    hprev = np.zeros((n_hidden_state, 1))

    xs, hs, ys, loss = {}, {}, {}, {}
    dw_xh, dw_hh, dw_hy = np.zeros_like(w_xh), np.zeros_like(w_hh), np.zeros_like(w_hy)
    db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)

    hs[-1] = hprev
    loss_sum = 0

    # Forward pass ------------------------------------------------------------
    for t in range(len(inputs)):
        xs[t] = inputs[t]
        hs[t] = np.tanh(np.dot(w_xh, xs[t]) + np.dot(w_hh, hs[t-1]) + b_h)  # Hidden state
        ys[t] = f_identity(np.dot(w_hy, hs[t]) + b_y)  # RNN output node
        loss[t] = np.square(ys[t] - targets[t])  # Output loss
        loss_sum += loss[t] / seq_length

    # Backpropagate through time ----------------------------------------------
    for t in np.arange(seq_length - 1)[::-1]:
        e_y_t = ys[t] - targets[t]           # Error (y): dL/dyhat
        bpe_y_t = e_y_t * f_didentity(ys[t]) # BPE (y)

        dw_hy += bpe_y_t * hs[t].T   # Gradient (w_hy): dL/dyhat * dyhat/dWhy
        db_y += bpe_y_t              # Gradient (b_y): dL/dyhat * dyhat/dby

        e_h_t = w_hy.T.dot(bpe_y_t)       # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
        bpe_h_t = e_h_t * f_dtanh(hs[t])  # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)

        for bptt_step in np.arange(t+1)[::-1]:

            # Gradient (w_xh): dL_t/d(h_{t-1} Whh) * d(h_{t-1} Whh)/dWhh
            dw_hh += np.outer(bpe_h_t, hs[bptt_step-1])

            dw_xh += bpe_h_t * xs[t]  # Gradient (w_xh): dL_t/d(x_t Wxh) ** d(x_t Wxh) / dWxh
            db_h += bpe_h_t           # Gradient (b_h): dL_t/d(x_t Wxh)

            # Update bpe_h for next step
            e_h_t = w_hh.T.dot(bpe_h_t)                 # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
            bpe_h_t = e_h_t * f_dtanh(hs[bptt_step-1])  # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)

    for dparam in [dw_xh, dw_hh, dw_hy, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam) # Clip to mitigate exploding gradients

    # Update parameters -------------------------------------------------------
    w_hy -= alpha * dw_hy
    w_hh -= alpha * dw_hh
    w_xh -= alpha * dw_xh

    b_y -= alpha * db_y
    b_h -= alpha * db_h

    # Control -----------------------------------------------------------------
    # Print the error to show that we are improving
    if (i % 1000) == 0:
        print('Error after {} iterations: {}'.format(i, loss_sum))

    # Exit if the error is less than maxError
    if(loss_sum < maxError):
        print('Goal reached after {} iterations: {} is smaller than the goal of {}'.format(i, loss_sum, maxError))
        break

# Generate predictions using the trained RNN ----------------------------------
def sample(h, seed_n, n):
    """Sample a sequence from the model.

    Args:
        h: is memory state,
        seed_n: is seed number for first time step
        n: length of sequence to generate
    """
    output_vals = np.array([])
    x = seed_n
    hs = {}
    hs[-1] = h
    for i in range(n):
        hs[i] = np.tanh(np.dot(w_xh, x) + np.dot(w_hh, hs[i-1]) + b_h)
        x = np.dot(w_hy, hs[i]) + b_y  # Feed output back into RNN
        output_vals = np.append(output_vals, x)
    return output_vals, hs

preds, preds_h = sample(hs[14], data[-1], 5)
print('Normalized predictions: {}'.format(np.around(preds, 2)))
print('Real predictions: {}'.format(np.around((preds * data_sd) + data_mean, 0)))
#{k: np.around(v, 2) for k, v in preds_h.items()}
