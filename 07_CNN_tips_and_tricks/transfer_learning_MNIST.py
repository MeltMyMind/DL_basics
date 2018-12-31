#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2018
@author: lucagaegauf

Transfer learning to character recognition task using LeNet-5 trained on digit 
recognition task. 

LeNet-5 CNN structure based on LeCun, et al. (1998). Changes to the original:
    - Image dim 28x28x1 (original 32x32x1)
    - Output layer uses softmax (original Euclidean RBF)
    - Optimizer SDG(lr=0.01) (original ???)

Changes to LeNet-5 for transfer learning model:
    - Add 5x5 same conv layer before second pooling layer.
    - Output layer: 26-nodes

Try ---------------------------------------------------------------------------
- Reducing the training set size:
    Remember to keep it relatively balanced and see how few samples you can use
    and still get good results.
- Reducing the epoch size:
    How small can you make and still get good results?

Sources -----------------------------------------------------------------------
Download dataset from: 
  https://www.nist.gov/itl/iad/image-group/emnist-dataset
Load EMNIST data:
  https://github.com/j05t/emnist/blob/master/emnist.ipynb
Transfer learning tutorial:
  https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
"""

#%% Setup workspace
import numpy as np

from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils

from scipy import io as spio
import matplotlib.pyplot as plt

#%% Data processing
num_classes = 26
img_rows, img_cols = 28, 28

# Load data
emnist = spio.loadmat("data/emnist-letters.mat")

# load training and test dataset and labels
x_train = emnist["dataset"][0][0][0][0][0][0]
x_test  = emnist["dataset"][0][0][1][0][0][0]
y_train = emnist["dataset"][0][0][0][0][0][1]
y_test  = emnist["dataset"][0][0][1][0][0][1]

# reshape using matlab order
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train -= 1
y_test  -= 1

# labels should be onehot encoded
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

#%% Plot
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.subplot(222)
plt.imshow(x_train[1].reshape(28, 28), cmap='gray')
plt.subplot(223)
plt.imshow(x_train[2].reshape(28, 28), cmap='gray')
plt.subplot(224)
plt.imshow(x_train[3].reshape(28, 28), cmap='gray')
# show the plot
plt.show()

#%% LeNet
# Initialize the model
leNet = Sequential()

# The first set of CONV (TANH) => POOL 
leNet.add(Conv2D(6, (5, 5), input_shape=input_shape, activation="tanh"))
leNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The second set of CONV (TANH) => POOL 
leNet.add(Conv2D(16, (5, 5), activation="tanh"))
leNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Two FC (TANH) layers
leNet.add(Flatten())
leNet.add(Dense(120, activation="tanh"))
leNet.add(Dense(84, activation="tanh"))

# The softmax classifier
leNet.add(Dense(10, activation="softmax"))

# Initialize weights with pre-trained weigths
leNet.load_weights("LeNet-5.h5")

leNet.summary()

#%% Transfer learning
# Freeze the layers which you don't want to train (the first 3 layers)
for layer in leNet.layers[:3]:
    layer.trainable = False

leNet.summary() # Notice change in footnote of table

#%% Update model 
# Output from frozen layers
x = leNet.layers[2].output

# CONV => TANH => POOL
x = Conv2D(10, (5, 5), activation="tanh", padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# FC => TANH => FC => TANH
x = Flatten()(x)
x = Dense(120, activation="tanh")(x)
x = Dense(84, activation="tanh")(x)

# FC => SOFTMAX
predictions = Dense(num_classes, activation="softmax")(x)

# Create new model
leNet_update = Model(inputs=leNet.input, outputs=predictions)

leNet_update.summary()

# You will also notice a new conv layer in the first row. This is not a 
# problem as it doesn't have any parameters.

#%% Compile the model 
leNet_update.compile(loss="categorical_crossentropy", 
              optimizer=SGD(lr=0.01), 
              metrics=["accuracy"])

#%% Train the model 
leNet_update.fit(x_train, y_train, 
          batch_size=32, epochs=10, verbose=1)

#%% Evaluate the model
score = leNet_update.evaluate(x_test, y_test, verbose=0)
score

#%% Save the weights
leNet.save_weights("LeNet_update.h5")

