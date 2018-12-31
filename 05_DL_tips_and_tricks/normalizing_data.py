#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 5 2018

@author: lpupp
"""

# Import dependencies ---------------------------------------------------------
import numpy as np

from sklearn import decomposition
#from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Training data ---------------------------------------------------------------
# Set inputs (x1, x2)
X = np.array([[2.0, 1.5],
              [2.8, 1.2],
              [1.5, 1.0],
              [2.1, 1.0],
              [5.5, 4.0],
              [8.0, 4.8],
              [6.9, 4.5],
              [8.5, 5.5],
              [2.5, 2.0],
              [7.7, 3.5]])

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

# Demean ----------------------------------------------------------------------
X_demean = X - np.mean(X, axis=0)
X_demean

# Standardize -----------------------------------------------------------------
X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
print(np.around(X_standardized, 2))

# Rescale to [-1, 1] ----------------------------------------------------------
X_rescaled = 2 * (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0)) - 1
print(np.around(X_rescaled, 2))

# PCA -------------------------------------------------------------------------
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
print(np.around(X_pca, 2))

# Whiten ----------------------------------------------------------------------
def whiten(X, eps=1E-18):
    # Get the covariance matrix
    Xcov = np.dot(X.T, X)
    # Eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
    # A fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    D = np.diag(1.0 / np.sqrt(d + eps))
    # Whitening matrix
    W = np.dot(np.dot(V, D), V.T)
    # Multiply by the whitening matrix
    X_white = np.dot(X, W)

    return X_white, W

x_white, _ = whiten(X_standardized)
print(np.around(x_white, 2))
