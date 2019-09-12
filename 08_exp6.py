#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd

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

ptsne = pd.read_csv('../data/ptsne_times.csv', header=None)
ptsne.columns = ['test_size', 'inf_time', 'train_inf_time']

results_q3 = joblib.load('fig6_figure3.eps_results.pkl')

test_sizes = results_q3['test_sizes']
sample_test_sizes = results_q3['sample_test_sizes']
tsne_times = results_q3['tsne_times']
umap_times = results_q3['umap_times']
mds_times = results_q3['mds_times']
lamp_times = results_q3['lamp_times']
lsp_times = results_q3['lsp_times']

ours_times = results_q3['ours_times']
ours_tsne_times = results_q3['ours_tsne_times']
ours_umap_times = results_q3['ours_umap_times']
umap_infer_times = results_q3['umap_infer_times']

ours_mds_times = results_q3['ours_mds_times']
ours_lamp_times = results_q3['ours_lamp_times']
ours_lsp_times = results_q3['ours_lsp_times']

ptsne_times = list(ptsne['train_inf_time'])
ptsne_infer_times = list(ptsne['inf_time'])

fig = plt.figure(figsize=(18, 15))
ax = fig.add_subplot(111)
ax.plot(test_sizes, tsne_times, label='t-SNE')
ax.plot(test_sizes[:umap_times.index(-1)], umap_times[:umap_times.index(-1)], label='UMAP')
ax.plot(test_sizes[:mds_times.index(-1)], mds_times[:mds_times.index(-1)], label='MDS')
ax.plot(test_sizes, lamp_times, label='LAMP')
ax.plot(test_sizes[:lsp_times.index(-1)], lsp_times[:lsp_times.index(-1)], label='LSP')
ax.plot(test_sizes, ours_tsne_times, label='Ours (t-SNE seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_umap_times, label='Ours (UMAP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_mds_times, label='Ours (MDS seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_lamp_times, label='Ours (LAMP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_lsp_times, label='Ours (LSP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_times, label='Ours (inference only)', ls='dashed', linewidth=3)
ax.plot(test_sizes, umap_infer_times, label='UMAP (inference only)', ls='dashed')
ax.plot(test_sizes, ptsne_infer_times, label='pt-SNE (inference only)', ls='dashed')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlabel('# of samples')
ax.set_ylabel('time (seconds), log scale')
ax.set_xlim((3000, 1000000))
ax.set_yscale('log')
ax.legend()
ax.set_title('Times for projecting MNIST data set')
fig.savefig('fig8_figure5.eps_%s_labeled.png' % 'mnist')

fig = plt.figure(figsize=(18, 15))
ax = fig.add_subplot(111)
ax.plot(test_sizes, tsne_times, label='t-SNE')
ax.plot(test_sizes[:umap_times.index(-1)], umap_times[:umap_times.index(-1)], label='UMAP')
ax.plot(test_sizes[:mds_times.index(-1)], mds_times[:mds_times.index(-1)], label='MDS')
ax.plot(test_sizes, lamp_times, label='LAMP')
ax.plot(test_sizes[:lsp_times.index(-1)], lsp_times[:lsp_times.index(-1)], label='LSP')
ax.plot(test_sizes, ours_tsne_times, label='Ours (t-SNE seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_umap_times, label='Ours (UMAP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_mds_times, label='Ours (MDS seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_lamp_times, label='Ours (LAMP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_lsp_times, label='Ours (LSP seed + train + inference)', ls=':', linewidth=2)
ax.plot(test_sizes, ours_times, label='Ours (inference only)', ls='dashed', linewidth=3)
ax.plot(test_sizes, umap_infer_times, label='UMAP (inference only)', ls='dashed')
ax.plot(test_sizes, ptsne_infer_times, label='pt-SNE (inference only)', ls='dashed')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim((3000, 1000000))
ax.set_yscale('log')
fig.savefig('fig8_figure5.eps_%s.png' % 'mnist')
