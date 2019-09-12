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

X_mnist_bin = X_mnist[np.isin(y_mnist, [0, 1])]
y_mnist_bin = y_mnist[np.isin(y_mnist, [0, 1])]

X_fashion_bin = X_fashion[np.isin(y_fashion, [0, 9])]
y_fashion_bin = y_fashion[np.isin(y_fashion, [0, 9])]

results_q1 = dict()

train_size = 3000
test_size = 3000

epochsl = [5, 10, 25, 50, 100]

if not os.path.exists('fig3_epochs.eps.pkl'):
    for label, X, y, p_tsne, p_umap, p_mds, p_lamp, p_lsp in zip(
        ['mnist-bin', 'mnist-full', 'fashion-bin', 'fashion-full'],
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
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=420, stratify=y)
        orig_nh = metrics.metric_neighborhood_hit(X_test, y_test)
    
        for p in [p_umap, p_tsne, p_mds, p_lamp, p_lsp]:
            X_2d = project(X_train, p)
    
            proj_nh = metrics.metric_neighborhood_hit(X_2d, y_train)
            proj_tr = metrics.metric_trustworthiness(X_train, X_2d, k=7)
            proj_co = metrics.metric_continuity(X_train, X_2d, k=7)
            proj_sh = metrics.metric_pq_shepard_diagram_correlation(X_train, X_2d)
    
            X_new = X_2d
    
            fig, ax = plt.subplots(6, 1, figsize=(10, 60))
            fig.tight_layout()
    
            figl, axl = plt.subplots(6, 1, figsize=(10, 60))
    
            for x, c in enumerate(np.unique(y_train)):
                ax[0].axis('off')
                ax[0].scatter(X_new[y_train==c,0],  X_new[y_train==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    
            for x, c in enumerate(np.unique(y_train)):
                axl[0].axis('off')
                axl[0].scatter(X_new[y_train==c,0],  X_new[y_train==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
                axl[0].set_title(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % test_size)
                axl[0].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, ' % (proj_nh, proj_tr, proj_co, proj_sh))
    
            print(label + ' - projected with ' + p.__class__.__name__ + ', %d samples' % test_size)
            print('NH: %.2f, T: %.2f, C: %.2f, R: %.2f, ' % (proj_nh, proj_tr, proj_co, proj_sh))
    
    
            for epochs, j in zip(epochsl, [1,2,3,4,5]):
                model, hist = nnproj.train_model(X_train, X_2d, epochs=epochs)
                X_2d_pred = model.predict(X_test)
    
                nn_nh = metrics.metric_neighborhood_hit(X_2d_pred, y_test)
                nn_tr = metrics.metric_trustworthiness(X_test, X_2d_pred, k=7)
                nn_co = metrics.metric_continuity(X_test, X_2d_pred, k=7)
                nn_sh = metrics.metric_pq_shepard_diagram_correlation(X_test, X_2d_pred)
    
                results_q1[(label, p.__class__.__name__, epochs)] = dict()
                results_q1[(label, p.__class__.__name__, epochs)]['proj_size'] = X_test.shape[0]
                results_q1[(label, p.__class__.__name__, epochs)]['orig_nh'] = orig_nh
                results_q1[(label, p.__class__.__name__, epochs)]['proj_nh'] = proj_nh
                results_q1[(label, p.__class__.__name__, epochs)]['proj_tr'] = proj_tr
                results_q1[(label, p.__class__.__name__, epochs)]['proj_co'] = proj_co
                results_q1[(label, p.__class__.__name__, epochs)]['proj_sh'] = proj_sh
                results_q1[(label, p.__class__.__name__, epochs)]['nn_nh'] = nn_nh
                results_q1[(label, p.__class__.__name__, epochs)]['nn_tr'] = nn_tr
                results_q1[(label, p.__class__.__name__, epochs)]['nn_co'] = nn_co
                results_q1[(label, p.__class__.__name__, epochs)]['nn_sh'] = nn_sh
    
                results_q1[(label, p.__class__.__name__, epochs)]['X_2d_pred'] = X_2d_pred
                results_q1[(label, p.__class__.__name__, epochs)]['X_2d'] = X_2d
                results_q1[(label, p.__class__.__name__, epochs)]['X_new'] = X_new
                results_q1[(label, p.__class__.__name__, epochs)]['model'] = model
                results_q1[(label, p.__class__.__name__, epochs)]['hist'] = hist
    
                for x, c in enumerate(np.unique(y_test)):
                    ax[j].axis('off')
                    ax[j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
    
                for x, c in enumerate(np.unique(y_test)):
                    axl[j].axis('off')
                    axl[j].scatter(X_2d_pred[y_test==c,0], X_2d_pred[y_test==c,1],  c=cmap(x), s=15, label=c, alpha=0.7)
                    axl[j].set_title(label + ' - learned from %d samples with %d epochs' % (train_size, epochs))
                    axl[j].text(0, 0, 'NH: %.2f, T: %.2f, C: %.2f, R: %.2f, ' % (nn_nh, nn_tr, nn_co, nn_sh))
    
                print(label + ' - learned from %d samples' % train_size)
    
            fig.savefig('fig3_epochs.eps_%s_%s.png' % (label, p.__class__.__name__))
            figl.savefig('fig3_epochs.eps_%s_%s_labeled.png' % (label, p.__class__.__name__))
    
    joblib.dump(results_q1, 'fig3_epochs.eps.pkl')

results_q1 = joblib.load('fig3_epochs.eps.pkl')

for dataset in ['mnist-bin', 'mnist-full', 'fashion-bin', 'fashion-full']:
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()
    ax = fig.add_subplot(111)
 
    for p in ['MulticoreTSNE', 'UMAP', 'MDS', 'LAMP', 'LSP']:
        epochs = 100
        l = results_q1[(dataset, p, epochs)]['hist'].history['loss']
        v = results_q1[(dataset, p, epochs)]['hist'].history['val_loss']

        ax.plot(range(epochs), l[:epochs], label='%s, %d epochs, train' % (p, epochs))
        ax.plot(range(epochs), v[:epochs], label='%s, %d epochs, valid' % (p, epochs))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('# of epochs')
    ax.set_ylabel('loss')
    ax.set_ylim(0, 0.1)
    ax.set_xlim((0, max(epochsl)))
    ax.legend()

    fig.show()
    fig.savefig('fig4_convplots.eps_%s.png' % dataset)

