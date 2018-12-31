#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 2018
@author: lpupp

Training LeNet-5 for transfer learning exercise.

LeNet-5 CNN structure based on LeCun, et al. (1998). Changes to the original:
    - Image dim 28x28x1 (original 32x32x1)
    - Output layer uses softmax (original Euclidean RBF)
    - Optimizer SDG(lr=0.01) (original ???)

Sources -----------------------------------------------------------------------
LeCun, et al. (1998):
  http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf
"""
# Import dependencies ---------------------------------------------------------
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt

# Training data ---------------------------------------------------------------
num_classes = 10
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# Plot ------------------------------------------------------------------------
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

# LeNet -----------------------------------------------------------------------
# Initialize the model
leNet = Sequential()

# The first set of CONV (TANH) => POOL
leNet.add(Conv2D(6, (5, 5), input_shape=input_shape, activation='tanh'))
leNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# The second set of CONV (TANH) => POOL
leNet.add(Conv2D(16, (5, 5), activation='tanh'))
leNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Two FC (TANH) layers
leNet.add(Flatten())
leNet.add(Dense(120, activation='tanh'))
leNet.add(Dense(84, activation='tanh'))

# The softmax classifier
leNet.add(Dense(num_classes, activation='softmax'))

leNet.summary()

# Compile the model -----------------------------------------------------------
leNet.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01),
              metrics=['accuracy'])

# Train the model -------------------------------------------------------------
leNet.fit(x_train, y_train,
          batch_size=32, epochs=10, verbose=1)

# Evaluate the model ----------------------------------------------------------
score = leNet.evaluate(x_test, y_test, verbose=0)
score

# Save the weights ------------------------------------------------------------
leNet.save_weights('LeNet-5.h5')
