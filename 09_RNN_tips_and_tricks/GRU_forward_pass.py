#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23rd 2018
@author: lucagaegauf

Single epoch of GRU

LSTM structure ----------------------------------------------------------------
Input Layer (x): 1 node
GRU (h, c): 3 hidden states
- Update and reset gates: sigmoid activated
- Hidden state: Tanh activated

zt = sigmoid(Uz * Xt + Wz * Ht-1 + bz)
rt = sigmoid(Ur * Xt + Wr * Ht-1 + br)
Ht = zt * Ht-1  + (1 – zt) * tanH(Uh * Xt + Wh * Ht-1 * rt + bh)

Output Layer (y): 1 node (linear activation)
yt = identity(Vy * Ht + by)

Variable key ------------------------------------------------------------------
w:   weight
b:   bias
f:   function
e:   error
d:   derivative
bpe: backpropagation error
t:   time index

Sources -----------------------------------------------------------------------
https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
"""
#%%
import numpy as np

#%% RNN training params -------------------------------------------------------
n_hidden_state = 3  # Number of hidden states

#%% Helper functions ----------------------------------------------------------
def f_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def f_dsigmoid(x):
    return x * (1 - x)

def f_dtanh(x):
    return 1 - (x ** 2)

def f_identity(x):
    return x

def f_didentity(x):
    return 1

#%% Generate and process data -------------------------------------------------
data = np.array([11, 0, 5, 5, 16])
data_length = len(data)
print('The sequence is {} time steps.'.format(data_length))

# Normalize data
data_mean = np.mean(data, axis=0)
data_sd = np.std(data, axis=0, ddof=1)
data = np.around((data - data_mean) / data_sd, 2)
data

#%% Initialize RNN parameters -------------------------------------------------
# Weights
U_z = np.array([[-0.2], [ 0.4], [-1.0]])
W_z = np.array([[-0.4, -0.7, -0.8],
                [ 0.1, -0.5,  0.8],
                [ 0.9,  0.4,  0.2]])
 
U_r = np.array([[-0.2], [-0.1], [ 0.2]])
W_r = np.array([[ 0.0,  1.4, -0.5],
                [ 0.2, -0.5,  0.3],
                [ 0.1, -0.5, -0.9]])

U_h = np.array([[ 0.3], [ 0.1], [ 0.4]])
W_h = np.array([[-0.3, -0.3, -0.8],
                [ 0.1, -0.5,  0.2],
                [-0.3, -0.7, -0.1]])

V_y = np.array([[-0.6, -0.6, -0.3]])
    
# Biases
b_z = np.array([[ 0.3], [ 1.2], [-0.4]])
b_r = np.array([[ 0.4], [-0.3], [ 0.8]])
b_h = np.array([[ 0.0], [-0.2], [-0.9]])
b_y = np.array([[ 0.5]])

#%% LSTM forward pass ---------------------------------------------------------
loss_sum = 0

inputs = data[:-1]
targets = data[1:]

seq_length = len(inputs)

ht = np.zeros((seq_length + 1, n_hidden_state))
zt = np.zeros((seq_length, n_hidden_state))
rt = np.zeros((seq_length, n_hidden_state))

# Initialize hidden states
ht[0,:] = np.array([1., 2., 3.])

yt = np.zeros((seq_length, 1))
xt = np.zeros((seq_length, 1))
loss = np.zeros((seq_length, 1))

for t in range(len(inputs)):
    xt[t] = inputs[t]
    
    zt[t,:] = f_sigmoid(U_z.T * xt[t] + np.dot(W_z.T, ht[t, :]) + b_z.T)
    rt[t,:] = f_sigmoid(U_r.T * xt[t] + np.dot(W_r.T, ht[t, :]) + b_r.T)
    ht[t+1,:] = zt[t, :] * ht[t, :] + (1 - zt[t,:]) * np.tanh(U_h.T * xt[t, :] + rt[t, :] * np.dot(W_h.T, ht[t, :]) + b_h.T)

    yt[t] = f_identity(np.dot(V_y, ht[t+1,:]) + b_y) # RNN output node
    loss[t] = 0.5 * np.square(yt[t] - targets[t]) # output loss
    loss_sum += loss[t] / seq_length
