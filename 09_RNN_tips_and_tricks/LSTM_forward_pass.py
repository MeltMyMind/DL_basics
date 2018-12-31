#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 23rd 2018
@author: lucagaegauf

Single epoch of LSTM

LSTM structure ----------------------------------------------------------------
Input Layer (x): 1 node
LSTM (h, c): 3 hidden states
- Forget, input, and output gates: sigmoid activated
- Hidden and cell states: Tanh activated

ft = sigmoid(Uf * Xt + Wf * Ht-1 + bf)
it = sigmoid(Ui * Xt + Wi * Ht-1 + bi)
ot = sigmoid(Uo * Xt + Wo * Ht-1 + bo)
Ct = ft * Ct-1 + it * tanH(Uc * Xt + Wc * Ht-1 + bc)
Ht = ot * tanH(Ct)

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
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
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
U_f = np.array([[-0.2], [ 0.4], [-1.0]])
W_f = np.array([[-0.4, -0.7, -0.8],
                [ 0.1, -0.5,  0.8],
                [ 0.9,  0.4,  0.2]])
 
U_i = np.array([[-0.2], [-0.1], [ 0.2]])
W_i = np.array([[ 0.0,  1.4, -0.5],
                [ 0.2, -0.5,  0.3],
                [ 0.1, -0.5, -0.9]])

U_o = np.array([[-0.1], [ 0.5], [-0.7]])
W_o = np.array([[ 0.4, -0.1, -0.3],
                [-0.2,  0.2,  0.3],
                [ 0.1,  0.1,  0.1]])

U_c = np.array([[ 0.3], [ 0.1], [ 0.4]])
W_c = np.array([[-0.3, -0.3, -0.8],
                [ 0.1, -0.5,  0.2],
                [-0.3, -0.7, -0.1]])

V_y = np.array([[-0.6, -0.6, -0.3]])
    
# Biases
b_f = np.array([[ 0.3], [ 1.2], [-0.4]])
b_i = np.array([[ 0.4], [-0.3], [ 0.8]])
b_o = np.array([[ 0.6], [ 0.1], [ 0.6]])
b_c = np.array([[ 0.0], [-0.2], [-0.9]])
b_y = np.array([[ 0.5]])

#%% LSTM forward pass ---------------------------------------------------------
loss_sum = 0

inputs = data[:-1]
targets = data[1:]

seq_length = len(inputs)

ht = np.zeros((seq_length + 1, n_hidden_state))
ct = np.zeros((seq_length + 1, n_hidden_state))
ft = np.zeros((seq_length, n_hidden_state))
it = np.zeros((seq_length, n_hidden_state))
ot = np.zeros((seq_length, n_hidden_state))

# Initialize hidden and cell states
ht[0,:], ct[0,:] = np.array([1., 2., 3.]), np.array([1., 2., 3.])

yt = np.zeros((seq_length, 1))
xt = np.zeros((seq_length, 1))
loss = np.zeros((seq_length, 1))

for t in range(len(inputs)):
    xt[t] = inputs[t]
    ft[t,:] = f_sigmoid(U_f.T * xt[t] + np.dot(W_f.T, ht[t,:]) + b_f.T)
    it[t,:] = f_sigmoid(U_i.T * xt[t] + np.dot(W_i.T, ht[t,:]) + b_i.T)
    ot[t,:] = f_sigmoid(U_o.T * xt[t] + np.dot(W_o.T, ht[t,:]) + b_o.T)
    ct[t+1,:] = ft[t,:] * ct[t,:] + it[t,:] * np.tanh(U_c.T * xt[t,:] + np.dot(W_c.T, ht[t,:]) + b_c.T)
    ht[t+1,:] = ot[t,:] * np.tanh(ct[t+1,:])
    
    yt[t] = f_identity(np.dot(V_y, ht[t+1,:]) + b_y) # RNN output node
    loss[t] = 0.5 * np.square(yt[t] - targets[t]) # output loss
    loss_sum += loss[t] / seq_length

#%% 
print(np.around(ft, 2))
print(np.around(it, 2))
print(np.around(ot, 2))
print(np.around(ht, 2))
print(np.around(ct, 2))


