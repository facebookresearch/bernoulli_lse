import numpy as np
import pandas as pd

import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('..')

from plot_config import *
import re

run_data = list(Path("../data/cameraready/").glob("*out.csv"))


def compile_results(run_data):
    dfiles = [pd.read_csv(d) for d in run_data]
    df = pd.concat(dfiles)
    df["method"] = df.OptimizeAcqfGenerator_acqf.fillna("Quasi-random").astype(
        "category"
    )
    df["method"] = df.method.cat.rename_categories(
        {
            "MCLevelSetEstimation": "Straddle",
            "MCPosteriorVariance": "BALV",
            "BernoulliMCMutualInformation": "BALD",
        }
    )

    df = df[df.final==False] 
    
    acqfs = [
        "Straddle",
        "EAVC",
        "LocalMI",
        "GlobalMI",
        "LocalSUR",
        "GlobalSUR",
        "ApproxGlobalSUR",
        "Quasi-random",
        "BALD",
        "BALV",
    ]
    metrics = [
        "brier",
        "class_errors",
        "prop_edge_sampling_mean",
        "fit_time",
        "gen_time",
    ]

    problems = df["problem"].unique()
    res = {
        metric: {prob: {acqf: [] for acqf in acqfs} for prob in problems}
        for metric in metrics
    }
    itrs = {}
    for _, g in df.groupby(["method", "rep", "problem"]):
        prob = g["problem"].values[0]

        if prob in itrs:
            if not np.array_equal(g["trial_id"].values, itrs[prob]):
                raise Exception
        else:
            itrs[prob] = g["trial_id"].values
        acqf = g["method"].values[0]
        for metric in metrics:
            res[metric][prob][acqf].append(g[metric])

    for metric in metrics:
        for prob in problems:
            for acqf in acqfs:
                res[metric][prob][acqf] = np.array(res[metric][prob][acqf])

    return res, itrs


def make_benchmark_figure():
    res, itrs = compile_results(run_data)

    # Make the plot
    fig = plt.figure(figsize=(6.75, 2.3))

    # Plot each problem
    metric = "brier"
    prob = "hartmann6_binary"

    methods = list(method_styles.keys())[:8]
    
    # prob = 'Binarized\n Hartmann6'
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
    ax.set_ylim([0.15, 0.5])
    ax.set_yticks([0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Brier Score")
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
    ax.set_ylim([0.1, 0.5])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Iteration")

    fig.subplots_adjust(bottom=0.34, left=0.065, top=0.91, right=0.99, wspace=0.2)

    plt.savefig("pdfs/benchmarks.pdf", pad_inches=0)


def make_realworld_figure():
    res, itrs = compile_results(run_data)

    # Make the plot
    fig = plt.figure(figsize=(3.25, 3))
    ax = fig.add_subplot(111)
    prob = "contrast_sensitivity_6d"

    metric = "brier"

    methods = list(method_styles.keys())[:8]

    for method in methods:
        y = res[metric][prob][method]
        ymean = np.mean(y, axis=0)
        yerr = 2 * np.std(y, axis=0) / np.sqrt(y.shape[0])
        color, ls = method_styles[method]
        ax.errorbar(itrs[prob], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(itrs[prob], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_xlim([0, 760])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_ylim([0.1, 0.6])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.grid(alpha=0.1)
    ax.set_title("Contrast Sensitivity (6-d)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Brier Score")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=[0.05, -0.7])
    fig.subplots_adjust(bottom=0.38, left=0.13, top=0.93, right=0.98)

    plt.savefig("pdfs/realworld.pdf", pad_inches=0)


if __name__ == "__main__":
    make_benchmark_figure()
    make_realworld_figure()
