#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy import spatial
from scipy import stats
import joblib

def compute_distance_list(X):
    return spatial.distance.pdist(X, 'euclidean')

def compute_distance_matrix(X):
    D = spatial.distance.pdist(X, 'euclidean')
    return spatial.distance.squareform(D)

def metric_neighborhood_hit(X, y, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    neighbors = knn.kneighbors(X, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))

def metric_trustworthiness(X_high, X_low, k=7):
    D_high = compute_distance_matrix(X_high)
    D_low = compute_distance_matrix(X_low)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_continuity(X_high, X_low, k=7):
    D_high = compute_distance_matrix(X_high)
    D_low = compute_distance_matrix(X_low)

    n = X_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())


def metric_pq_shepard_diagram_correlation(X_high, X_low):
    D_high = compute_distance_list(X_high)
    D_low = compute_distance_list(X_low)

    return stats.spearmanr(D_high, D_low)[0]
