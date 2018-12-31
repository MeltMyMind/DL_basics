#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sept 22 2018
@author: lucagaegauf

Plotting the filters and activation maps of a CNN. This example uses a 
pre-trained VGG19 network.

Since the VGG19 is not the same dimension as our network, we reduce the
dimensions in order to fit our purposes. As this is only a demonstration of the
concepts, this should not matter.
"""

from keras import applications
import matplotlib.pyplot as plt
import numpy as np

import cv2

#%%
# Define the activation function and how to forward pass through a CNN
def f_relu(x):
    return x * (x > 0)

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
    
    return convolved_image + bias

#%%
# Load the pre-trained VGG19 network
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (256, 256, 3))

#%%
# Get the model weights and bases for the first 3 layers
w = model.get_weights()

# 6 filters were selected such that they have enough variance to see that the
# filter train to different solutions.
w_hl1 = w[0][:,:,:,[0,1,3,10,23,40]]
b_hl1 = w[1][[0,1,3,10,23,40]]
w_hl2 = w[2][:,:,:,:10]
b_hl2 = w[3][:10]
w_hl3 = w[4][:,:,:,:16]
b_hl3 = w[5][:16]

#%% Plot the filters
plt.imshow(w_hl1[:,:,:,0])
plt.imshow(w_hl1[:,:,:,1])
plt.imshow(w_hl1[:,:,:,3])
plt.imshow(w_hl1[:,:,:,10])
plt.imshow(w_hl1[:,:,:,23])
plt.imshow(w_hl1[:,:,:,40])

#%% Plot the activation maps
# Load and resize the image
img = cv2.imread('./Desktop/puppy.png')
img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)

# We are using same convolution in the example so we will pad our inputs with
# a border of zeros (thickness = 1)
pad = ((1,1), (1,1), (0,0))

# arrays to save activation maps
a_hl1 = np.zeros((32, 32, 6))
a_hl2 = np.zeros((32, 32, 10))
a_hl3 = np.zeros((32, 32, 16))

#%% CNN forward pass with padding
# Activation map 1: six 3x3 filters
for i in range(6):
    a_hl1[:,:,i] = f_convolve(np.pad(img, pad, 'constant', constant_values=0), w_hl1[:,:,:,i], b_hl1[i])

a_hl1 = f_relu(a_hl1) # activate

# Activation map 2: ten 3x3 filters
for i in range(10):
    a_hl2[:,:,i] = f_convolve(np.pad(a_hl1, pad, 'constant', constant_values=0), w_hl2[:,:,:6,i], b_hl2[i])

a_hl2 = f_relu(a_hl2) # activate

# Activation map 3: sixteen 3x3 filters
for i in range(16):
    a_hl3[:,:,i] = f_convolve(np.pad(a_hl2, pad, 'constant', constant_values=0), w_hl3[:,:,:10,i], b_hl3[i])

a_hl3 = f_relu(a_hl3) # activate

#%% plot the activation maps
plt.imshow(a_hl1[:,:,0], cmap='gray')
plt.imshow(a_hl2[:,:,0], cmap='gray')
plt.imshow(a_hl3[:,:,1], cmap='gray')

#%% Save to file
#for i in range(6):
#    cv2.imwrite('./activation_maps/a_hl1_{}.png'.format(i), (a_hl1[:,:,i] * (255. / np.max(a_hl1[:,:,i]))).astype(int))

#for i in range(10):
#    cv2.imwrite('./activation_maps/a_hl2_{}.png'.format(i), (a_hl2[:,:,i] * (255. / np.max(a_hl2[:,:,i]))).astype(int))
    
#for i in range(16):
#    cv2.imwrite('./activation_maps/a_hl3_{}.png'.format(i), (a_hl3[:,:,i] * (255. / np.max(a_hl3[:,:,i]))).astype(int))

