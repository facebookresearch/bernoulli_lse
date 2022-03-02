# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from pathlib import Path
import pandas as pd
import numpy as np

from plot_config import *

run_data = list(Path("../data/gentime_bench/").glob("*out.csv"))

import re


def make_figure():
    alld = []
    for f in run_data:
        dlocal = pd.read_csv(f)
        dlocal["nthreads"] = re.findall(".*(\d+)threads_out.csv", str(f))[0]
        alld.append(dlocal)
    df = pd.concat(alld)
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
    
    df = df[df.trial_id.isin(([251]))]
    
    methods = [
        "EAVC",
        "LocalMI",
        "Straddle",
    ]
    
    fig = plt.figure(figsize=(6.75, 2.3))
    # Plot each problem
    prob = "hartmann6_binary"
    
    ax = fig.add_subplot(131)
    for method in methods:
        df_m = df[(df['problem'] == prob) & (df['method'] == method)]
        res = df_m.groupby('nthreads').agg({'gen_time': ['mean', 'std', 'count']})
        res = res.droplevel(axis=1, level=0).reset_index()
        ymean = res['mean']
        yerr = 2 * res['std'] / np.sqrt(res['count'])
        color, ls = method_styles[method]
        ax.errorbar(res['nthreads'], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(res['nthreads'], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_ylim([0., 3.2])
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Acquisition wall time (s)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.9, -0.63), ncol=4)

    prob = "discrim_lowdim"
    ax = fig.add_subplot(132)
    for method in methods:
        df_m = df[(df['problem'] == prob) & (df['method'] == method)]
        res = df_m.groupby('nthreads').agg({'gen_time': ['mean', 'std', 'count']})
        res = res.droplevel(axis=1, level=0).reset_index()
        ymean = res['mean']
        yerr = 2 * res['std'] / np.sqrt(res['count'])
        color, ls = method_styles[method]
        ax.errorbar(res['nthreads'], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(res['nthreads'], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_ylim([0., 3.2])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (2-d)")
    ax.set_xlabel("Number of threads")

    prob = "discrim_highdim"
    ax = fig.add_subplot(133)
    for method in methods:
        df_m = df[(df['problem'] == prob) & (df['method'] == method)]
        res = df_m.groupby('nthreads').agg({'gen_time': ['mean', 'std', 'count']})
        res = res.droplevel(axis=1, level=0).reset_index()
        ymean = res['mean']
        yerr = 2 * res['std'] / np.sqrt(res['count'])
        color, ls = method_styles[method]
        ax.errorbar(res['nthreads'], ymean, yerr=yerr, lw=1, alpha=0.3, color=color, ls=ls)
        ax.plot(res['nthreads'], ymean, lw=1, label=method, color=color, ls=ls)
    ax.set_ylim([0., 3.2])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([])
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Number of threads")

    fig.subplots_adjust(bottom=0.34, left=0.05, top=0.91, right=0.99, wspace=0.1)

    plt.savefig("pdfs/gentime_plots.pdf", pad_inches=0)


if __name__ == "__main__":
    make_figure()
