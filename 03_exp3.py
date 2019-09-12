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
from sklearn import manifold, decomposition
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

cmap = plt.get_cmap('tab10')

X_mnist = np.load('../data/X_mnist.npy')
y_mnist = np.load('../data/y_mnist.npy')

X_fashion = np.load('../data/X_fashion.npy')
y_fashion = np.load('../data/y_fashion.npy')

X_dogsandcats = np.load('../data/X_dogsandcats.npy')
y_dogsandcats = np.load('../data/y_dogsandcats.npy')

X_imdb = np.load('../data/X_imdb.npy')
y_imdb = np.load('../data/y_imdb.npy')

train_size = 5000
test_size = 15000

all_metrics = dict()

for label, X, y, p_tsne, p_umap, p_isomap, p_aes, p_aem, p_pca in zip(['mnist', 'fashion', 'dogs-vs-cats', 'imdb'],
                                        [X_mnist, X_fashion, X_dogsandcats, X_imdb],
                                        [y_mnist, y_fashion, y_dogsandcats, y_imdb],
                                        [TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=10.0, n_iter=1000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=30.0, n_iter=1000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=300.0, n_iter=5000, n_iter_without_progress=300, n_jobs=4)],
                                        [umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=5, min_dist=0.3),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.3),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=60, min_dist=0.7, metric='euclidean')],
                                         [manifold.Isomap(), manifold.Isomap(), manifold.Isomap(), manifold.Isomap()],
                                         [ae.AutoencoderProjection(ae.ModelSize.SMALL), ae.AutoencoderProjection(ae.ModelSize.SMALL), ae.AutoencoderProjection(ae.ModelSize.SMALL), ae.AutoencoderProjection(ae.ModelSize.SMALL)],
                                         [ae.AutoencoderProjection(ae.ModelSize.MEDIUM), ae.AutoencoderProjection(ae.ModelSize.MEDIUM), ae.AutoencoderProjection(ae.ModelSize.MEDIUM), ae.AutoencoderProjection(ae.ModelSize.MEDIUM)],
                                         [decomposition.PCA(n_components=2), decomposition.PCA(n_components=2), decomposition.PCA(n_components=2), decomposition.PCA(n_components=2)]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=420, stratify=y)
    orig_nh = metrics.metric_neighborhood_hit(X_test, y_test)

    fig, ax = plt.subplots(2, 4, figsize=(40, 20))
    fig.tight_layout()

    figl, axl = plt.subplots(2, 4, figsize=(40, 20))

    for p, (j, k) in zip([p_tsne, p_umap, p_isomap, p_aes, p_pca, p_aem], [(0,0), (0,1), (0,2), (0,3), (1,2), (1,3)]):
        t0 = perf_counter()
        X_new = project(X_test, p)
        proj_elapsed_time = perf_counter() - t0

        proj_nh = metrics.metric_neighborhood_hit(X_new, y_test)
        proj_tr = metrics.metric_trustworthiness(X_test, X_new, k=7)
        proj_co = metrics.metric_continuity(X_test, X_new, k=7)
        proj_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_new)

        all_metrics[(p.__class__.__name__, label)] = dict()
        all_metrics[(p.__class__.__name__, label)]['orig_nh'] = orig_nh
        all_metrics[(p.__class__.__name__, label)]['proj_nh'] = proj_nh
        all_metrics[(p.__class__.__name__, label)]['proj_tr'] = proj_tr
        all_metrics[(p.__class__.__name__, label)]['proj_co'] = proj_co
        all_metrics[(p.__class__.__name__, label)]['proj_sh'] = proj_sh

        for x, c in enumerate(np.unique(y_test)):
            ax[j,k].axis('off')
            ax[j,k].scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)

        for x, c in enumerate(np.unique(y_test)):
            axl[j,k].axis('off')
            axl[j,k].scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
            axl[j,k].set_title(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % test_size)
            axl[j,k].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (proj_nh, proj_tr, proj_co, proj_sh, proj_elapsed_time))

        print(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % test_size)
        print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (proj_nh, proj_tr, proj_co, proj_sh, proj_elapsed_time))


    #seeding with t-SNE
    t0 = perf_counter()
    X_2d_train = project(X_train, p_tsne)
    nn_seed_elapsed_time = perf_counter() - t0

    t0 = perf_counter()
    model, _ = nnproj.train_model(X_train, X_2d_train)
    nn_train_elapsed_time = perf_counter() - t0

    t0 = perf_counter()
    X_2d_pred = model.predict(X_test)
    nn_pred_elapsed_time = perf_counter() - t0

    nn_nh = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    nn_tr = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    nn_co = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    nn_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

    all_metrics[('TSNE_NN', label)] = dict()
    all_metrics[('TSNE_NN', label)]['nn_nh'] = nn_nh
    all_metrics[('TSNE_NN', label)]['nn_tr'] = nn_tr
    all_metrics[('TSNE_NN', label)]['nn_co'] = nn_co
    all_metrics[('TSNE_NN', label)]['nn_sh'] = nn_sh

    for x, c in enumerate(np.unique(y_test)):
        ax[1,0].axis('off')
        ax[1,0].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)

    for x, c in enumerate(np.unique(y_test)):
        axl[1,0].axis('off')
        axl[1,0].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
        axl[1,0].set_title(label + ' - learned from t-SNE with %d samples' % train_size)
        axl[1,0].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (nn_nh, nn_tr, nn_co, nn_sh, (nn_seed_elapsed_time + nn_train_elapsed_time + nn_pred_elapsed_time)))
        axl[1,0].text(0, 1, 'seed: %.2f, train: %.2f, infer: %.2f' % (nn_seed_elapsed_time, nn_train_elapsed_time, nn_pred_elapsed_time))


    print(label + ' - learned from t-SNE with %d samples' % train_size)
    print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (nn_nh, nn_tr, nn_co, nn_sh, (nn_seed_elapsed_time + nn_train_elapsed_time + nn_pred_elapsed_time)))
    print('seed: %.2f, train: %.2f, infer: %.2f' % (nn_seed_elapsed_time, nn_train_elapsed_time, nn_pred_elapsed_time))

    #seeding with UMAP
    t0 = perf_counter()
    X_2d_train = project(X_train, p_umap)
    nn_seed_elapsed_time = perf_counter() - t0

    t0 = perf_counter()
    model, _ = nnproj.train_model(X_train, X_2d_train)
    nn_train_elapsed_time = perf_counter() - t0

    t0 = perf_counter()
    X_2d_pred = model.predict(X_test)
    nn_pred_elapsed_time = perf_counter() - t0

    nn_nh = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
    nn_tr = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
    nn_co = metrics.metric_continuity(X_test, X_2d_pred, k=7)
    nn_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

    all_metrics[('UMAP_NN', label)] = dict()
    all_metrics[('UMAP_NN', label)]['nn_nh'] = nn_nh
    all_metrics[('UMAP_NN', label)]['nn_tr'] = nn_tr
    all_metrics[('UMAP_NN', label)]['nn_co'] = nn_co
    all_metrics[('UMAP_NN', label)]['nn_sh'] = nn_sh

    for x, c in enumerate(np.unique(y_test)):
        ax[1,1].axis('off')
        ax[1,1].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)

    for x, c in enumerate(np.unique(y_test)):
        axl[1,1].axis('off')
        axl[1,1].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
        axl[1,1].set_title(label + ' - learned from UMAP with %d samples' % train_size)
        axl[1,1].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (nn_nh, nn_tr, nn_co, nn_sh, (nn_seed_elapsed_time + nn_train_elapsed_time + nn_pred_elapsed_time)))
        axl[1,1].text(0, 1, 'seed: %.2f, train: %.2f, infer: %.2f' % (nn_seed_elapsed_time, nn_train_elapsed_time, nn_pred_elapsed_time))

    print(label + ' - learned from UMAP with %d samples' % train_size)
    print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (nn_nh, nn_tr, nn_co, nn_sh, (nn_seed_elapsed_time + nn_train_elapsed_time + nn_pred_elapsed_time)))
    print('seed: %.2f, train: %.2f, infer: %.2f' % (nn_seed_elapsed_time, nn_train_elapsed_time, nn_pred_elapsed_time))

    fig.savefig('fig5_figure2.eps_%s.png' % label)
    figl.savefig('fig5_figure2.eps_%s_labeled.png' % label)

joblib.dump(all_metrics, 'fig5_figure2.eps_metrics.pkl')


