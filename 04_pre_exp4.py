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


X_mnist = np.load('../data/X_mnist.npy')
y_mnist = np.load('../data/y_mnist.npy')

sample_test_sizes = [2000, 10000, 30000, 60000, 100000]

#Q3
train_size = 5000

test_sizes = [1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 75000, 100000, 500000, 1000000]
tsne_times = []
umap_times = []
mds_times = []
lamp_times = []
lsp_times = []
ours_times = []
ours_tsne_times = []
ours_umap_times = []
ours_mds_times = []
ours_lamp_times = []
ours_lsp_times = []
umap_infer_times = []

X_train, X_test_full, y_train, y_test_full = train_test_split(X_mnist, y_mnist, train_size=train_size, random_state=420, stratify=y_mnist)

X_test_full = np.vstack((X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full, X_test_full))
y_test_full = np.hstack((y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full, y_test_full))

p_tsne = TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4)
p_umap = umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001)
p_umap_oos = umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001)
p_mds = MDS(n_components=2, metric=True, random_state=420)
p_lamp = vp.LAMP()
p_lsp = vp.LSP()

#seeding with t-SNE
t0 = perf_counter()
X_2d_train = project(X_train, p_tsne)
model_tsne, _ = nnproj.train_model(X_train, X_2d_train)
train_tsne_elapsed_time = perf_counter() - t0

#seeding with UMAP
t0 = perf_counter()
X_2d_train = project(X_train, p_umap_oos)
model_umap, _ = nnproj.train_model(X_train, X_2d_train)
train_umap_elapsed_time = perf_counter() - t0

#seeding with MDS
t0 = perf_counter()
X_2d_train = project(X_train, p_mds)
model_mds, _ = nnproj.train_model(X_train, X_2d_train)
train_mds_elapsed_time = perf_counter() - t0

#seeding with LAMP
t0 = perf_counter()
X_2d_train = project(X_train, p_lamp)
model_lamp, _ = nnproj.train_model(X_train, X_2d_train)
train_lamp_elapsed_time = perf_counter() - t0

#seeding with LSP
t0 = perf_counter()
X_2d_train = project(X_train, p_lsp)
model_lsp, _ = nnproj.train_model(X_train, X_2d_train)
train_lsp_elapsed_time = perf_counter() - t0

results_q3 = dict()
data_q3 = dict()
chkpt = dict()

for test_size in test_sizes:
    chkpt[test_size] = False

if os.path.exists('fig6_figure3.eps_data.pkl'):
    data_q3 = joblib.load('fig6_figure3.eps_data.pkl')

if os.path.exists('fig6_figure3.eps_results.pkl'):
    results_q3 = joblib.load('fig6_figure3.eps_results.pkl')

    tsne_times = results_q3['tsne_times']
    umap_times = results_q3['umap_times']
    mds_times = results_q3['mds_times']
    lamp_times = results_q3['lamp_times']
    lsp_times = results_q3['lsp_times']
    ours_times = results_q3['ours_times']
    ours_tsne_times = results_q3['ours_tsne_times']
    ours_umap_times = results_q3['ours_umap_times']
    ours_mds_times = results_q3['ours_mds_times']
    ours_lamp_times = results_q3['ours_lamp_times']
    ours_lsp_times = results_q3['ours_lsp_times']
    umap_infer_times = results_q3['umap_infer_times']

if os.path.exists('fig6_chkpt.pkl'):
    chkpt = joblib.load('fig6_chkpt.pkl')

for test_size in test_sizes:
    _, X_test, _, y_test = train_test_split(X_test_full, y_test_full, test_size=test_size, random_state=420, stratify=y_test_full)
    np.save('../data/X_test_%d.npy' % test_size, X_test)

    if chkpt[test_size]:
        print('%d is present processed - SKIPPING' % test_size)
        continue

    try:
        print('t-SNE, %d' % test_size)
        t0 = perf_counter()
        X_tsne = project(X_test, p_tsne)
        tsne_elapsed_time = perf_counter() - t0
    except:
        print('t-SNE, %d - FAILED' % test_size)
        X_tsne = None
        tsne_elapsed_time = -1.0

    try:
        print('UMAP, %d' % test_size)
        t0 = perf_counter()
        X_umap = project(X_test, p_umap)
        umap_elapsed_time = perf_counter() - t0
    except:
        print('UMAP, %d - FAILED' % test_size)
        X_umap = None
        umap_elapsed_time = -1.0

    try:
        print('MDS, %d' % test_size)
        t0 = perf_counter()
        X_mds = project(X_test, p_mds)
        mds_elapsed_time = perf_counter() - t0
    except:
        print('MDS, %d - FAILED' % test_size)
        X_mds = None
        mds_elapsed_time = -1.0

    try:
        print('LAMP, %d' % test_size)
        t0 = perf_counter()
        X_lamp = project(X_test, p_lamp)
        lamp_elapsed_time = perf_counter() - t0
    except:
        print('LAMP, %d - FAILED' % test_size)
        X_lamp = None
        lamp_elapsed_time = -1.0

    try:
        print('LSP, %d' % test_size)
        t0 = perf_counter()
        X_lsp = project(X_test, p_lsp)
        lsp_elapsed_time = perf_counter() - t0
    except:
        print('LSP, %d - FAILED' % test_size)
        X_lsp = None
        lsp_elapsed_time = -1.0

    try:
        print('UMAP (infer), %d' % test_size)
        t0 = perf_counter()
        X_umap_pred = p_umap_oos.transform(X_test)
        umap_infer_elapsed_time = perf_counter() - t0
    except:
        print('UMAP (infer), %d - FAILED' % test_size)
        X_umap_pred = None
        umap_infer_elapsed_time = -1.0

    print('Ours (t-SNE), %d' % test_size)
    t0 = perf_counter()
    X_nn_tsne = model_tsne.predict(X_test)
    ours_tsne_elapsed_time = perf_counter() - t0
    ours_elapsed_time = ours_tsne_elapsed_time
    ours_tsne_elapsed_time += train_tsne_elapsed_time

    print('Ours (UMAP), %d' % test_size)
    t0 = perf_counter()
    X_nn_umap = model_umap.predict(X_test)
    ours_umap_elapsed_time = perf_counter() - t0
    ours_umap_elapsed_time += train_umap_elapsed_time

    print('Ours (MDS), %d' % test_size)
    t0 = perf_counter()
    X_nn_mds = model_mds.predict(X_test)
    ours_mds_elapsed_time = perf_counter() - t0
    ours_mds_elapsed_time += train_mds_elapsed_time

    print('Ours (LAMP), %d' % test_size)
    t0 = perf_counter()
    X_nn_lamp = model_lamp.predict(X_test)
    ours_lamp_elapsed_time = perf_counter() - t0
    ours_lamp_elapsed_time += train_lamp_elapsed_time

    print('Ours (LSP), %d' % test_size)
    t0 = perf_counter()
    X_nn_lsp = model_lsp.predict(X_test)
    ours_lsp_elapsed_time = perf_counter() - t0
    ours_lsp_elapsed_time += train_lsp_elapsed_time

    umap_infer_times.append(umap_infer_elapsed_time)
    tsne_times.append(tsne_elapsed_time)
    umap_times.append(umap_elapsed_time)
    mds_times.append(mds_elapsed_time)
    lamp_times.append(lamp_elapsed_time)
    lsp_times.append(lsp_elapsed_time)
    ours_times.append(ours_elapsed_time)
    ours_tsne_times.append(ours_tsne_elapsed_time)
    ours_umap_times.append(ours_umap_elapsed_time)
    ours_mds_times.append(ours_mds_elapsed_time)
    ours_lamp_times.append(ours_lamp_elapsed_time)
    ours_lsp_times.append(ours_lsp_elapsed_time)

    if test_size <= 100000:
        data_q3[test_size] = dict()
        data_q3[test_size]['y_test'] = y_test
        data_q3[test_size]['X_tsne'] = X_tsne
        data_q3[test_size]['X_umap'] = X_umap
        data_q3[test_size]['X_mds'] = X_mds
        data_q3[test_size]['X_lamp'] = X_lamp
        data_q3[test_size]['X_lsp'] = X_lsp
        data_q3[test_size]['X_umap_pred'] = X_umap_pred
        data_q3[test_size]['X_nn_tsne'] = X_nn_tsne
        data_q3[test_size]['X_nn_umap'] = X_nn_umap
        data_q3[test_size]['X_nn_mds'] = X_nn_mds
        data_q3[test_size]['X_nn_lamp'] = X_nn_lamp
        data_q3[test_size]['X_nn_lsp'] = X_nn_lsp

    joblib.dump(data_q3, 'fig6_figure3.eps_data.pkl')

    results_q3['test_sizes'] = test_sizes
    results_q3['sample_test_sizes'] = sample_test_sizes
    results_q3['tsne_times'] = tsne_times
    results_q3['umap_times'] = umap_times
    results_q3['mds_times'] = mds_times
    results_q3['lamp_times'] = lamp_times
    results_q3['lsp_times'] = lsp_times
    results_q3['ours_times'] = ours_times
    results_q3['ours_tsne_times'] = ours_tsne_times
    results_q3['ours_umap_times'] = ours_umap_times
    results_q3['umap_infer_times'] = umap_infer_times
    results_q3['ours_mds_times'] = ours_mds_times
    results_q3['ours_lamp_times'] = ours_lamp_times
    results_q3['ours_lsp_times'] = ours_lsp_times

    joblib.dump(results_q3, 'fig6_figure3.eps_results.pkl')

    chkpt[test_size] = True
    joblib.dump(chkpt, 'fig6_chkpt.pkl')

joblib.dump(results_q3, 'fig6_figure3.eps_results.pkl')
joblib.dump(data_q3, 'fig6_figure3.eps_data.pkl')
