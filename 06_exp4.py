#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import ae
import metrics
import nnproj
import vp

def project(X, p):
    X_new = p.fit_transform(X)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_new)

if os.path.exists('fig6_chkpt.pkl'):
    chkpt = joblib.load('fig6_chkpt.pkl')
else:
    print('Checkpoint file not found')
    exit(1)

test_sizes = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000, 500000, 1000000]

for test_size in test_sizes:
    if chkpt[test_size] == False:
        print('Test %d missing. Aborting')
        exit(1)

cmap = plt.get_cmap('tab10')

data_q3 = joblib.load('fig6_figure3.eps_data.pkl')
results_q3 = joblib.load('fig6_figure3.eps_results.pkl')

sample_test_sizes = results_q3['sample_test_sizes']

for test_size in sample_test_sizes:
    X_ptsne = np.loadtxt('../data/X_ptsne_%d.csv' % test_size, delimiter=',')
    y_ptsne = np.loadtxt('../data/y_ptsne_%d.csv' % test_size, delimiter=',')

    data_q3[test_size]['y_test_ptsne'] = y_ptsne
    data_q3[test_size]['X_ptsne'] = X_ptsne

fig, ax = plt.subplots(11, len(sample_test_sizes), figsize=(10*len(sample_test_sizes), 110))
fig.tight_layout()

figl, axl = plt.subplots(11, len(sample_test_sizes), figsize=(10*len(sample_test_sizes), 110))

for j, test_size in enumerate(sample_test_sizes):
    for i, t in enumerate(['X_tsne', 'X_nn_tsne', 'X_umap', 'X_nn_umap', 'X_mds', 'X_nn_mds', 'X_lamp', 'X_nn_lamp', 'X_lsp', 'X_nn_lsp', 'X_ptsne']):
        if t == 'X_ptsne':
            X_test = np.loadtxt('../data/X_test_ptsne_%d.csv' % test_size, delimiter=',')
            y_test = data_q3[test_size]['y_test_ptsne']
        else:
            X_test = np.load('../data/X_test_%d.npy' % test_size)
            y_test = data_q3[test_size]['y_test']

        X_2d_pred = data_q3[test_size][t]
        
        if X_2d_pred is not None:
            for color, c in enumerate(np.unique(y_test)):
                ax[i,j].axis('off')
                ax[i,j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

            for color, c in enumerate(np.unique(y_test)):
                axl[i,j].axis('off')
                axl[i,j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

            if 'X_nn' in t:
                title = '%s - learned from %d samples' % (t, test_size)
            else:
                title = '%s - projecting %d samples' % (t, test_size)

            nh = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
            tr = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
            co = metrics.metric_continuity(X_test, X_2d_pred, k=7)
            sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

            data_q3[test_size][(t, 'metrics')] = dict()
            data_q3[test_size][(t, 'metrics')]['nh'] = nh
            data_q3[test_size][(t, 'metrics')]['tr'] = tr
            data_q3[test_size][(t, 'metrics')]['co'] = co
            data_q3[test_size][(t, 'metrics')]['sh'] = sh

            axl[i,j].set_title(title)
            axl[i,j].text(0, 0, 'NH: %.2f' % nh)

fig.savefig('fig6_figure3.eps_mnist.png')
figl.savefig('fig6_figure3.eps_mnist_labeled.png')

joblib.dump(data_q3, 'fig6_figure3.eps_data.pkl')
