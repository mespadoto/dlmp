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

cmap = plt.get_cmap('tab10')

X_mnist = np.load('../data/X_mnist.npy')
y_mnist = np.load('../data/y_mnist.npy')

X_fashion = np.load('../data/X_fashion.npy')
y_fashion = np.load('../data/y_fashion.npy')

X_dogsandcats = np.load('../data/X_dogsandcats.npy')
y_dogsandcats = np.load('../data/y_dogsandcats.npy')

X_imdb = np.load('../data/X_imdb.npy')
y_imdb = np.load('../data/y_imdb.npy')

X_mnist_bin = X_mnist[np.isin(y_mnist, [0, 1])]
y_mnist_bin = y_mnist[np.isin(y_mnist, [0, 1])]

X_fashion_bin = X_fashion[np.isin(y_fashion, [0, 9])]
y_fashion_bin = y_fashion[np.isin(y_fashion, [0, 9])]


results_q1 = dict()

for label, X, y, p_tsne, p_umap, p_mds, p_lamp, p_lsp in zip(['mnist-bin', 'mnist-full', 'fashion-bin', 'fashion-full'],
                                        [X_mnist_bin, X_mnist, X_fashion_bin, X_fashion],
                                        [y_mnist_bin, y_mnist, y_fashion_bin, y_fashion],
                                        [TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=25.0, n_iter=3000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=10.0, n_iter=1000, n_iter_without_progress=300, n_jobs=4),
                                         TSNE(n_components=2, random_state=420, perplexity=10.0, n_iter=1000, n_iter_without_progress=300, n_jobs=4)],
                                        [umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=10, min_dist=0.001),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=5, min_dist=0.3),
                                         umap.UMAP(n_components=2, random_state=420, n_neighbors=5, min_dist=0.3)],
                                        [MDS(n_components=2, metric=True, random_state=420),
                                         MDS(n_components=2, metric=True, random_state=420),
                                         MDS(n_components=2, metric=True, random_state=420),
                                         MDS(n_components=2, metric=True, random_state=420)],
                                        [vp.LAMP(), vp.LAMP(), vp.LAMP(), vp.LAMP()],
                                        [vp.LSP(), vp.LSP(), vp.LSP(), vp.LSP()]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, test_size=3000, random_state=420, stratify=y)
    orig_nh = metrics.metric_neighborhood_hit(X_test, y_test)

    for p in [p_umap, p_tsne, p_mds, p_lamp, p_lsp]:
        t0 = perf_counter()
        X_new = project(X_test, p)
        proj_elapsed_time = perf_counter() - t0

        proj_nh = metrics.metric_neighborhood_hit(X_new, y_test)
        proj_tr = metrics.metric_trustworthiness(X_test, X_new, k=7)
        proj_co = metrics.metric_continuity(X_test, X_new, k=7)
        proj_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_new)

        fig, ax = plt.subplots(6, 1, figsize=(10, 60))
        fig.tight_layout()

        figl, axl = plt.subplots(6, 1, figsize=(10, 60))

        for x, c in enumerate(np.unique(y_test)):
            ax[0].axis('off')
            ax[0].scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)

        for x, c in enumerate(np.unique(y_test)):
            axl[0].axis('off')
            axl[0].scatter(X_new[y_test==c,0],  X_new[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
            axl[0].set_title(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % 10000)
            axl[0].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (proj_nh, proj_tr, proj_co, proj_sh, proj_elapsed_time))

        print(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % 10000)
        print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (proj_nh, proj_tr, proj_co, proj_sh, proj_elapsed_time))

        for train_size, j in zip([1000, 2000, 3000, 5000, 9000], [1,2,3,4,5]):
            X_train_p, _, y_train_p, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=420, stratify=y_train)

            t0 = perf_counter()
            X_2d = project(X_train_p, p)
            seed_elapsed_time = perf_counter() - t0

            t0 = perf_counter()
            model, hist = nnproj.train_model(X_train_p, X_2d)
            train_elapsed_time = perf_counter() - t0

            epochs = len(hist.history['loss'])
            
            t0 = perf_counter()
            X_2d_pred = model.predict(X_test)
            infer_elapsed_time = perf_counter() - t0

            nn_nh = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
            nn_tr = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
            nn_co = metrics.metric_continuity(X_test, X_2d_pred, k=7)
            nn_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)

            results_q1[(label, p.__class__.__name__, train_size)] = dict()
            results_q1[(label, p.__class__.__name__, train_size)]['proj_size'] = X_test.shape[0]
            results_q1[(label, p.__class__.__name__, train_size)]['orig_nh'] = orig_nh
            results_q1[(label, p.__class__.__name__, train_size)]['proj_nh'] = proj_nh
            results_q1[(label, p.__class__.__name__, train_size)]['proj_tr'] = proj_tr
            results_q1[(label, p.__class__.__name__, train_size)]['proj_co'] = proj_co
            results_q1[(label, p.__class__.__name__, train_size)]['proj_sh'] = proj_sh
            results_q1[(label, p.__class__.__name__, train_size)]['nn_nh'] = nn_nh
            results_q1[(label, p.__class__.__name__, train_size)]['nn_tr'] = nn_tr
            results_q1[(label, p.__class__.__name__, train_size)]['nn_co'] = nn_co
            results_q1[(label, p.__class__.__name__, train_size)]['nn_sh'] = nn_sh
            results_q1[(label, p.__class__.__name__, train_size)]['proj_elapsed_time'] = proj_elapsed_time
            results_q1[(label, p.__class__.__name__, train_size)]['seed_elapsed_time'] = seed_elapsed_time
            results_q1[(label, p.__class__.__name__, train_size)]['train_elapsed_time'] = train_elapsed_time
            results_q1[(label, p.__class__.__name__, train_size)]['infer_elapsed_time'] = infer_elapsed_time
            results_q1[(label, p.__class__.__name__, train_size)]['X_2d_pred'] = X_2d_pred
            results_q1[(label, p.__class__.__name__, train_size)]['X_2d'] = X_2d
            results_q1[(label, p.__class__.__name__, train_size)]['X_new'] = X_new
            results_q1[(label, p.__class__.__name__, train_size)]['model'] = model
            results_q1[(label, p.__class__.__name__, train_size)]['hist'] = hist
            results_q1[(label, p.__class__.__name__, train_size)]['epochs'] = epochs

            for x, c in enumerate(np.unique(y_train_p)):
                ax[j].axis('off')
                ax[j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)

            for x, c in enumerate(np.unique(y_train_p)):
                axl[j].axis('off')
                axl[j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
                axl[j].set_title(label + ' - learned from %d samples' % train_size)
                axl[j].text(0, 1, 'seed: %.2f, train: %.2f, infer: %.2f' % (seed_elapsed_time, train_elapsed_time, infer_elapsed_time))
                axl[j].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f, epochs: %d' % (nn_nh, nn_tr, nn_co, nn_sh, train_elapsed_time, epochs))

            print(label + ' - learned from %d samples' % train_size)
            print('seed: %.2f, train: %.2f, infer: %.2f' % (seed_elapsed_time, train_elapsed_time, infer_elapsed_time))
            print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, elapsed time: %.2f' % (nn_nh, nn_tr, nn_co, nn_sh, train_elapsed_time))

        fig.savefig('fig2_figure1.eps_%s_%s.png' % (label, p.__class__.__name__))
        figl.savefig('fig2_figure1.eps_%s_%s_labeled.png' % (label, p.__class__.__name__))

joblib.dump(results_q1, 'fig2_figure1.eps.pkl')
