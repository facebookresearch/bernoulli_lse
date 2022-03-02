# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import pickle

import matplotlib.pyplot as plt
import numpy as np

import sys 
sys.path.append('..')

from plot_config import *
from plot_experiment_results import compile_results, run_data


def make_figure():

    res, itrs = compile_results(run_data)
    edge_samp = res['prop_edge_sampling_mean']
    # Make the plot
    fig = plt.figure(figsize=(6.75, 3))

    prob = 'hartmann6_binary'

    methods = list(method_styles.keys())[:8]

    ax = fig.add_subplot(131)
    for i, model_name in enumerate(methods):
        y = edge_samp[prob][model_name][:,-1].mean()
        n = edge_samp[prob][model_name].shape[0]
        yerr = 2 * edge_samp[prob][model_name][:,-1].std() / np.sqrt(n)
        ax.bar([i], [y], yerr=yerr, color=method_styles[model_name][0])
        print(model_name, y)
    
    ax.set_xticks(range(8))
    ax.set_xticklabels(methods, rotation=90)

    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0\%', '25\%', '50\%', '75\%', '100\%'])
    
    ax.grid(alpha=0.1)
    
    ax.set_ylabel('Proportion of samples near edge')
    ax.set_title('Binarized Hartmann6')

    prob = 'discrim_lowdim'
    ax = fig.add_subplot(132)
    for i, model_name in enumerate(methods):
        y = edge_samp[prob][model_name][:,-1].mean()
        n = edge_samp[prob][model_name].shape[0]
        yerr = 2 * edge_samp[prob][model_name][:,-1].std() / np.sqrt(n)
        ax.bar([i], [y], yerr=yerr, color=method_styles[model_name][0])
    
    ax.set_xticks(range(8))
    ax.set_xticklabels(methods, rotation=90)

    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([])
    
    ax.grid(alpha=0.1)
    
    ax.set_title('Psych. Discrimination (2-d)')
    
    prob = 'discrim_highdim'
    ax = fig.add_subplot(133)
    for i, model_name in enumerate(methods):
        y = edge_samp[prob][model_name][:,-1].mean()
        n = edge_samp[prob][model_name].shape[0]
        yerr = 2 * edge_samp[prob][model_name][:,-1].std() / np.sqrt(n)
        ax.bar([i], [y], yerr=yerr, color=method_styles[model_name][0])

    ax.set_xticks(range(8))
    ax.set_xticklabels(methods, rotation=90)

    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels([])
    
    ax.grid(alpha=0.1)
    
    ax.set_title('Psych. Discrimination (8-d)')
    
    fig.subplots_adjust(bottom=0.34, left=0.08, top=0.93, right=0.99, wspace=0.1)
    
    plt.savefig('pdfs/edge_sampling.pdf', pad_inches=0)



if __name__ == '__main__':
    make_figure()
