#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 09:04:50 2018

@author: lucagaegauf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 15th 2018
@author: lucagaegauf

Automated RNN with sequence mini batches (either overlapping or disjoint)

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
#%%
import numpy as np

#%% RNN training params -------------------------------------------------------
# Learning rate and max error
alpha = 1e-1
max_error = 1e-1
max_iter = 1000
n_hidden_state = 3 

#%% Helper functions ----------------------------------------------------------
def f_dtanh(x):
    return 1 - (x ** 2)

def f_identity(x):
    return x

def f_didentity(x):
    return 1
   
#%% Generate and process data -------------------------------------------------
data = np.array([11, 0, 5, 5, 16, 5, 14, 18, 18, 1, 15, 5, 2, 19, 8, 4])
data_length = len(data)
print('The sequence is {} time steps.'.format(data_length))

# Normalize data
data_mean = np.mean(data, axis=0)
data_sd = np.std(data, axis=0, ddof=1)
data = np.around((data - data_mean) / data_sd, 2)
data

#%% Initialize RNN parameters -------------------------------------------------
# Weights
w_xh = np.array([[-0.2], [0.4], [-1.0]])
w_hh = np.array([[-0.4, -0.7, -0.8],
                 [0.1, -0.5,  0.8], 
                 [0.9, 0.4, 0.2]])
w_hy = np.array([[-0.6, -0.6, -0.3]])

# Biases
b_h = np.array([[0.1], [-0.2],  [0.4]])
b_y = np.array([[-0.6]])

#%%
def fit_RNN(inputs, targets, hprev):

    xs, hs, ys, loss = {}, {}, {}, {}
    dw_xh, dw_hh, dw_hy = np.zeros_like(w_xh), np.zeros_like(w_hh), np.zeros_like(w_hy)
    db_h, db_y = np.zeros_like(b_h), np.zeros_like(b_y)
    
    hs[-1] = hprev
    loss_sum = 0
    
    # Forward pass ------------------------------------------------------------
    for t in range(seq_length):
        xs[t] = inputs[t]
        hs[t] = np.tanh(np.dot(w_xh, xs[t]) + np.dot(w_hh, hs[t-1]) + b_h) # hidden state
        ys[t] = f_identity(np.dot(w_hy, hs[t]) + b_y) # RNN output node 
        loss[t] = np.square(ys[t] - targets[t]) # output loss
        loss_sum += loss[t] / seq_length
        
    # Backpropagate through time ----------------------------------------------
    for t in np.arange(seq_length)[::-1]:
        e_y_t = ys[t] - targets[t]   # Error (y): dL/dyhat
        bpe_y_t = e_y_t * f_didentity(ys[t])  # BPE (y) 
        
        dw_hy += bpe_y_t * hs[t].T   # Gradient (w_hy): dL/dyhat * dyhat/dWhy
        db_y += bpe_y_t              # Gradient (b_y): dL/dyhat * dyhat/dby
        
        e_h_t = w_hy.T.dot(bpe_y_t)     # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
        bpe_h_t = e_h_t * f_dtanh(hs[t])  # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)
        
        for bptt_step in np.arange(t+1)[::-1]:
            
            # Gradient (w_xh): dL_t/d(h_{t-1} Whh) * d(h_{t-1} Whh)/dWhh
            dw_hh += np.outer(bpe_h_t, hs[bptt_step-1]) 
            
            dw_xh += bpe_h_t * xs[t]  # Gradient (w_xh): dL_t/d(x_t Wxh) ** d(x_t Wxh) / dWxh
            db_h += bpe_h_t           # Gradient (b_h): dL_t/d(x_t Wxh)
            
            # Update bpe_h for next step
            e_h_t = w_hh.T.dot(bpe_h_t) # Error (h): dL_t/dyhat_t * dyhat_t/dh_t
            bpe_h_t = e_h_t * f_dtanh(hs[bptt_step-1]) # BPE (h): dL_t/dh_t * dh_t/d(h_{t-1} Whh)

    for dparam in [dw_xh, dw_hh, dw_hy, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

    return loss_sum, dw_xh, dw_hh, dw_hy, db_h, db_y, hs[p_shift-1] 

#%%
start = True
p, i = 0, 0
seq_length = 3

# CHANGE HERE TO MAKE OVERLAPPING CHUNKS => p_shift < seq_length
p_shift = seq_length

while True: 
    
    # If we have processed the whole sequence or if we are starting
    if p+seq_length+1 >= len(data) or start: 
        start = False
        hprev = np.zeros((n_hidden_state, 1)) # reset RNN memory
        p = 0  # go from start of data
        i += 1 # epoch iteration counter 
        loss_total = 0 # RNN loss at iteration beginning of epoch 

    # Make mini batches of length seq_length
    inputs = data[p:(p+seq_length)]
    targets = data[(p+1):(p+1+seq_length)]

    # forward seq_length characters through the net and fetch gradient
    loss, dw_xh, dw_hh, dw_hy, db_h, db_y, hprev = fit_RNN(inputs, targets, hprev)
    loss_total = (loss_total * p + loss * seq_length) / (seq_length + p)
  
    # Update params using vanilla SGD
    for param, dparam in zip([w_xh, w_hh, w_hy, b_h, b_y], 
                             [dw_xh, dw_hh, dw_hy, db_h, db_y]):
        param -= alpha * dparam 

    # CHANGE HERE TO MAKE OVERLAPPING CHUNKS => p > seq_length
    p += p_shift # move data pointer 
    
    if p+seq_length+1 >= len(data):
        #  Control end of epoch -----------------------------------------------
        # Print the error to show that we are improving
        if (i % 1000) == 0:
            print("Error after " + str(i) + " iterations: " + str(loss_total))
            
        # Exit if the error is less than maxError
        if(loss_total < max_error):
            print("Goal reached after " + str(i) + " iterations: " + str(loss_total) + " is smaller than the goal of " + str(max_error))
            break
        
        # Exit if max_iter is exceeded
        if (i > max_iter):
            print("Max iterations reached after " + str(i) + " iterations. Loss achieved: " + str(loss_total))
            break

#%%
def sample(h, seed_n, n):
    """ 
    Sample a sequence from the model.
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
        x = np.dot(w_hy, hs[i]) + b_y # feed output back into RNN
        output_vals = np.append(output_vals, x)
    return output_vals, hs

preds, preds_h = sample(hprev, data[-1], 5)
print('Normalized predictions: {}'.format(np.around(preds, 2)))
print('Real predictions: {}'.format(np.around((preds * data_sd) + data_mean, 0)))
#{k: np.around(v, 2) for k, v in preds_h.items()}
