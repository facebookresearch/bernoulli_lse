# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import numpy as np
import pandas as pd
from copy import deepcopy

import sys

sys.path.append("..")

from plot_config import *
from plot_experiment_results import compile_results, run_data


def make_classerr_figure():
    res, itrs = compile_results(run_data)

    # Make the plot
    fig = plt.figure(figsize=(6.75, 2.3))
    metric = "class_errors"

    # Plot each problem
    prob = "hartmann6_binary"

    methods = list(method_styles.keys())[:8]

    ax = fig.add_subplot(131)
    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 760])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_ylim([0.15, 0.45])
    # ax.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Classification Error")
    ax.legend(loc="lower left", bbox_to_anchor=(0.45, -0.63), ncol=4)

    prob = "discrim_lowdim"
    ax = fig.add_subplot(132)
    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 510])
    ax.set_ylim([0.015, 0.11])
    # ax.set_yticks([0.02, 0.04, 0.06, 0.08, 0.1])
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (2-d)")
    ax.set_xlabel("Iteration")

    prob = "discrim_highdim"
    ax = fig.add_subplot(133)
    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 760])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_ylim([0.1, 0.42])
    # ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Iteration")

    fig.subplots_adjust(bottom=0.34, left=0.07, top=0.91, right=0.99, wspace=0.2)

    plt.savefig("pdfs/benchmark_classerr.pdf", pad_inches=0)



def make_bald_figure():
    res, itrs = compile_results(run_data)

    # Make the plot
    fig = plt.figure(figsize=(6.75, 2.3))

    # Plot each problem
    metric = "brier"

    prob = "hartmann6_binary"
    ax = fig.add_subplot(131)
    methods = list(method_styles.keys())

    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 760])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_ylim([0.15, 0.5])
    ax.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Brier Score")
    ax.legend(loc="lower left", bbox_to_anchor=(0.22, -0.63), ncol=5)

    prob = "discrim_lowdim"
    ax = fig.add_subplot(132)
    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 510])
    ax.set_ylim([0.01, 0.10])
    ax.set_yticks([0.02, 0.04, 0.06, 0.08, 0.1])
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (2-d)")
    ax.set_xlabel("Iteration")

    prob = "discrim_highdim"
    ax = fig.add_subplot(133)
    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 760])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_ylim([0.1, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Iteration")

    fig.subplots_adjust(bottom=0.34, left=0.065, top=0.91, right=0.99, wspace=0.2)

    plt.savefig("pdfs/benchmarks_bald.pdf", pad_inches=0)


if __name__ == "__main__":
    make_classerr_figure()
    make_bald_figure()
