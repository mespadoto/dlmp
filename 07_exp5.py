#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition, manifold
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

import metrics
import nnproj
import vp

def project(X, p):
    X_new = p.fit_transform(X)
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_new)

cmap = plt.get_cmap('tab10')

X_mnist = np.load('../data/X_mnist.npy')
y_mnist = np.load('../data/y_mnist.npy')

X_fashion = np.load('../data/X_fashion.npy')
y_fashion = np.load('../data/y_fashion.npy')

X_mnist_bin = X_mnist[np.isin(y_mnist, [0, 1])]
y_mnist_bin = y_mnist[np.isin(y_mnist, [0, 1])]

X_fashion_bin = X_fashion[np.isin(y_fashion, [0, 9])]
y_fashion_bin = y_fashion[np.isin(y_fashion, [0, 9])]

all_metrics = dict()

projections = []

#fashion bin
X_train, X_test, y_train, y_test = train_test_split(X_fashion_bin, y_fashion_bin, train_size=5000, test_size=5000, random_state=420, stratify=y_fashion_bin)

p_pca = decomposition.PCA(n_components=2)
p_iso = manifold.Isomap()
p_mds = manifold.MDS(metric=True)
p_lle = manifold.LocallyLinearEmbedding(method='standard', eigen_solver='dense', n_neighbors=50)

fig, ax = plt.subplots(2, 4, figsize=(40, 20))
fig.tight_layout()

figl, axl = plt.subplots(2, 4, figsize=(40, 20))

for i, p in enumerate([p_pca, p_iso, p_mds, p_lle]):
    X_2d_train = project(X_train, p)
    m, _ = nnproj.train_model(X_train, X_2d_train)
    X_2d_pred = m.predict(X_test)
    
    nh_train = metrics.metric_neighborhood_hit(X_2d_train, y_train)
    tr_train = metrics.metric_trustworthiness(X_train, X_2d_train, k=7)
    co_train = metrics.metric_continuity(X_train, X_2d_train, k=7)
    sh_train = metrics.metric_pq_shepard_diagram_correlation(X_train, X_2d_train)

    nh_test = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    tr_test = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    co_test = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    sh_test = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

    all_metrics['fashion-bin'] = dict()
    all_metrics['fashion-bin']['nh_train'] = nh_train
    all_metrics['fashion-bin']['tr_train'] = tr_train
    all_metrics['fashion-bin']['co_train'] = co_train
    all_metrics['fashion-bin']['sh_train'] = sh_train

    all_metrics['fashion-bin']['nh_test'] = nh_test
    all_metrics['fashion-bin']['tr_test'] = tr_test
    all_metrics['fashion-bin']['co_test'] = co_test
    all_metrics['fashion-bin']['sh_test'] = sh_test

    all_metrics['fashion-bin']['X_2d_train'] = X_2d_train
    all_metrics['fashion-bin']['X_2d_pred'] = X_2d_pred

    projections.append(X_2d_train)
    projections.append(X_2d_pred)
    
    for color, c in enumerate(np.unique(y_train)):
        ax[0,i].axis('off')
        ax[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_test)):
        ax[1,i].axis('off')
        ax[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_train)):
        axl[0,i].axis('off')
        axl[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[0,i].set_title('%s training projection' % p.__class__.__name__)
        axl[0,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_train, tr_train, co_train, sh_train))

    for color, c in enumerate(np.unique(y_test)):
        axl[1,i].axis('off')
        axl[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[1,i].set_title('Projection trained on %s' % p.__class__.__name__)
        axl[1,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_test, tr_test, co_test, sh_test))

fig.savefig('fig7_figure4.eps_fashion-bin.png')
figl.savefig('fig7_figure4.eps_fashion-bin_labeled.png')


#fashion full
X_train, X_test, y_train, y_test = train_test_split(X_fashion, y_fashion, train_size=5000, test_size=5000, random_state=420, stratify=y_fashion)

p_pca = decomposition.PCA(n_components=2)
p_iso = manifold.Isomap()
p_mds = manifold.MDS(metric=True)
p_lle = manifold.LocallyLinearEmbedding(method='standard', eigen_solver='dense', n_neighbors=50)

fig, ax = plt.subplots(2, 4, figsize=(40, 20))
fig.tight_layout()

figl, axl = plt.subplots(2, 4, figsize=(40, 20))

for i, p in enumerate([p_pca, p_iso, p_mds, p_lle]):
    X_2d_train = project(X_train, p)
    m, _ = nnproj.train_model(X_train, X_2d_train)
    X_2d_pred = m.predict(X_test)
    
    nh_train = metrics.metric_neighborhood_hit(X_2d_train, y_train)
    tr_train = metrics.metric_trustworthiness(X_train, X_2d_train, k=7)
    co_train = metrics.metric_continuity(X_train, X_2d_train, k=7)
    sh_train = metrics.metric_pq_shepard_diagram_correlation(X_train, X_2d_train)

    nh_test = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    tr_test = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    co_test = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    sh_test = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)
    
    all_metrics['fashion-full'] = dict()
    all_metrics['fashion-full']['nh_train'] = nh_train
    all_metrics['fashion-full']['tr_train'] = tr_train
    all_metrics['fashion-full']['co_train'] = co_train
    all_metrics['fashion-full']['sh_train'] = sh_train

    all_metrics['fashion-full']['nh_test'] = nh_test
    all_metrics['fashion-full']['tr_test'] = tr_test
    all_metrics['fashion-full']['co_test'] = co_test
    all_metrics['fashion-full']['sh_test'] = sh_test

    all_metrics['fashion-full']['X_2d_train'] = X_2d_train
    all_metrics['fashion-full']['X_2d_pred'] = X_2d_pred

    projections.append(X_2d_train)
    projections.append(X_2d_pred)
    
    for color, c in enumerate(np.unique(y_train)):
        ax[0,i].axis('off')
        ax[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_test)):
        ax[1,i].axis('off')
        ax[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_train)):
        axl[0,i].axis('off')
        axl[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[0,i].set_title('%s training projection' % p.__class__.__name__)
        axl[0,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_train, tr_train, co_train, sh_train))

    for color, c in enumerate(np.unique(y_test)):
        axl[1,i].axis('off')
        axl[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[1,i].set_title('Projection trained on %s' % p.__class__.__name__)
        axl[1,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_test, tr_test, co_test, sh_test))

fig.savefig('fig7_figure4.eps_fashion-full.png')
figl.savefig('fig7_figure4.eps_fashion-full_labeled.png')


#mnist bin
X_train, X_test, y_train, y_test = train_test_split(X_mnist_bin, y_mnist_bin, train_size=5000, test_size=5000, random_state=420, stratify=y_mnist_bin)

p_pca = decomposition.PCA(n_components=2)
p_iso = manifold.Isomap()
p_mds = manifold.MDS(metric=True)
p_lle = manifold.LocallyLinearEmbedding(method='standard', eigen_solver='dense', n_neighbors=50)

fig, ax = plt.subplots(2, 4, figsize=(40, 20))
fig.tight_layout()

figl, axl = plt.subplots(2, 4, figsize=(40, 20))

for i, p in enumerate([p_pca, p_iso, p_mds, p_lle]):
    X_2d_train = project(X_train, p)
    m, _ = nnproj.train_model(X_train, X_2d_train)
    X_2d_pred = m.predict(X_test)

    nh_train = metrics.metric_neighborhood_hit(X_2d_train, y_train)
    tr_train = metrics.metric_trustworthiness(X_train, X_2d_train, k=7)
    co_train = metrics.metric_continuity(X_train, X_2d_train, k=7)
    sh_train = metrics.metric_pq_shepard_diagram_correlation(X_train, X_2d_train)

    nh_test = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    tr_test = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    co_test = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    sh_test = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

    all_metrics['mnist-bin'] = dict()
    all_metrics['mnist-bin']['nh_train'] = nh_train
    all_metrics['mnist-bin']['tr_train'] = tr_train
    all_metrics['mnist-bin']['co_train'] = co_train
    all_metrics['mnist-bin']['sh_train'] = sh_train

    all_metrics['mnist-bin']['nh_test'] = nh_test
    all_metrics['mnist-bin']['tr_test'] = tr_test
    all_metrics['mnist-bin']['co_test'] = co_test
    all_metrics['mnist-bin']['sh_test'] = sh_test

    all_metrics['mnist-bin']['X_2d_train'] = X_2d_train
    all_metrics['mnist-bin']['X_2d_pred'] = X_2d_pred

    projections.append(X_2d_train)
    projections.append(X_2d_pred)
    
    for color, c in enumerate(np.unique(y_train)):
        ax[0,i].axis('off')
        ax[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_test)):
        ax[1,i].axis('off')
        ax[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_train)):
        axl[0,i].axis('off')
        axl[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[0,i].set_title('%s training projection' % p.__class__.__name__)
        axl[0,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_train, tr_train, co_train, sh_train))

    for color, c in enumerate(np.unique(y_test)):
        axl[1,i].axis('off')
        axl[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[1,i].set_title('Projection trained on %s' % p.__class__.__name__)
        axl[1,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_test, tr_test, co_test, sh_test))

fig.savefig('fig7_figure4.eps_mnist-bin.png')
figl.savefig('fig7_figure4.eps_mnist-bin_labeled.png')


#mnist full
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, train_size=5000, test_size=5000, random_state=420, stratify=y_mnist)

p_pca = decomposition.PCA(n_components=2)
p_iso = manifold.Isomap()
p_mds = manifold.MDS(metric=True)
p_lle = manifold.LocallyLinearEmbedding(method='standard', eigen_solver='dense', n_neighbors=50)

fig, ax = plt.subplots(2, 4, figsize=(40, 20))
fig.tight_layout()

figl, axl = plt.subplots(2, 4, figsize=(40, 20))

for i, p in enumerate([p_pca, p_iso, p_mds, p_lle]):
    X_2d_train = project(X_train, p)
    m, _ = nnproj.train_model(X_train, X_2d_train)
    X_2d_pred = m.predict(X_test)

    nh_train = metrics.metric_neighborhood_hit(X_2d_train, y_train)
    tr_train = metrics.metric_trustworthiness(X_train, X_2d_train, k=7)
    co_train = metrics.metric_continuity(X_train, X_2d_train, k=7)
    sh_train = metrics.metric_pq_shepard_diagram_correlation(X_train, X_2d_train)

    nh_test = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    tr_test = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    co_test = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    sh_test = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

    all_metrics['mnist-full'] = dict()
    all_metrics['mnist-full']['nh_train'] = nh_train
    all_metrics['mnist-full']['tr_train'] = tr_train
    all_metrics['mnist-full']['co_train'] = co_train
    all_metrics['mnist-full']['sh_train'] = sh_train

    all_metrics['mnist-full']['nh_test'] = nh_test
    all_metrics['mnist-full']['tr_test'] = tr_test
    all_metrics['mnist-full']['co_test'] = co_test
    all_metrics['mnist-full']['sh_test'] = sh_test

    all_metrics['mnist-full']['X_2d_train'] = X_2d_train
    all_metrics['mnist-full']['X_2d_pred'] = X_2d_pred

    projections.append(X_2d_train)
    projections.append(X_2d_pred)
    
    for color, c in enumerate(np.unique(y_train)):
        ax[0,i].axis('off')
        ax[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_test)):
        ax[1,i].axis('off')
        ax[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)

    for color, c in enumerate(np.unique(y_train)):
        axl[0,i].axis('off')
        axl[0,i].scatter(X_2d_train[y_train==c,0], X_2d_train[y_train==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[0,i].set_title('%s training projection' % p.__class__.__name__)
        axl[0,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_train, tr_train, co_train, sh_train))

    for color, c in enumerate(np.unique(y_test)):
        axl[1,i].axis('off')
        axl[1,i].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(color), s=15, label=c, alpha=0.7)
        axl[1,i].set_title('Projection trained on %s' % p.__class__.__name__)
        axl[1,i].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f' % (nh_test, tr_test, co_test, sh_test))

fig.savefig('fig7_figure4.eps_mnist-full.png')
figl.savefig('fig7_figure4.eps_mnist-full_labeled.png')

joblib.dump(all_metrics, 'fig7_figure4.eps_metrics.pkl')
