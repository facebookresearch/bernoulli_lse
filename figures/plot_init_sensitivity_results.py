import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from pathlib import Path

from plot_config import *

# need finalrun for init=10
rundata = list(Path("../data/finalrun/").glob("*out.csv")) + list(Path("../data/init_sensitivity").glob("*out.csv"))


def compile_results():
    dfiles = [pd.read_csv(d) for d in rundata]
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

    df = df[df.final == True]
    acqfs = [
        "EAVC",
        "LocalMI",
        "GlobalMI",
        "Quasi-random",
    ]

    problems = list(df["problem"].unique())
    problems.remove("discrim_lowdim")
    res = {prob: {} for prob in problems}

    for levels, g in df.groupby(["method", "rep", "problem", "init_strat_n_trials"]):
        method, _, prob, n_init = levels

        if method in acqfs and prob in problems:

            acqf = f"{method}_{n_init}"
            if acqf not in res[prob]:
                res[prob][acqf] = []
            res[prob][acqf].append(g["brier"].item())

    for prob in problems:
        for acqf in res[prob]:
            res[prob][acqf] = np.array(res[prob][acqf])

    return res


def make_init_sensitivity_figure():
    res = compile_results()

    ninits = [10, 100, 250, 500]
    # Make the plot
    fig = plt.figure(figsize=(6.75, 2.3))

    prob = "hartmann6_binary"
    ax = fig.add_subplot(131)
    methods = [
        "EAVC",
        "LocalMI",
        "GlobalMI",
    ]
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{ninit}"].mean() for ninit in ninits]
        yerr = [
            2
            * res[prob][f"{method}_{ninit}"].std()
            / np.sqrt(len(res[prob][f"{method}_{ninit}"]))
            for ninit in ninits
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            ninits,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(ninits, ymean, lw=1, label=method, color=color, ls=ls)
    # Add Sobol
    ymean = [res[prob]["Quasi-random_10"].mean() for ninit in ninits]
    yerr = [
        2
        * res[prob]["Quasi-random_10"].std()
        / np.sqrt(len(res[prob]["Quasi-random_10"]))
        for ninit in ninits
    ]
    color, ls = method_styles["Quasi-random"]
    ax.errorbar(
        ninits,
        ymean,
        yerr=yerr,
        lw=1,
        alpha=0.3,
        color=color,
        ls=ls,
    )
    ax.plot(ninits, ymean, lw=1, label="Quasi-random", color=color, ls=ls)

    ax.set_xticks(ninits)
    ax.set_ylim([0.15, 0.3])
    ax.set_yticks([0.15, 0.2, 0.25, 0.3])
    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Initial design size")
    ax.set_ylabel("Final Brier Score\nafter 750 iterations")
    ax.legend(loc="lower left", bbox_to_anchor=(0.45, -0.63), ncol=4)

    prob = "discrim_highdim"
    ax = fig.add_subplot(132)
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{ninit}"].mean() for ninit in ninits]
        yerr = [
            2
            * res[prob][f"{method}_{ninit}"].std()
            / np.sqrt(len(res[prob][f"{method}_{ninit}"]))
            for ninit in ninits
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            ninits,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(ninits, ymean, lw=1, label=method, color=color, ls=ls)
    # Add Sobol
    ymean = [res[prob]["Quasi-random_10"].mean() for ninit in ninits]
    yerr = [
        2
        * res[prob]["Quasi-random_10"].std()
        / np.sqrt(len(res[prob]["Quasi-random_10"]))
        for ninit in ninits
    ]
    color, ls = method_styles["Quasi-random"]
    ax.errorbar(
        ninits,
        ymean,
        yerr=yerr,
        lw=1,
        alpha=0.3,
        color=color,
        ls=ls,
    )
    ax.plot(ninits, ymean, lw=1, label="Quasi-random", color=color, ls=ls)
    ax.set_xticks(ninits)
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Initial design size")
    ax.set_ylim([0.1, 0.4])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4]) 

    prob = "contrast_sensitivity_6d"
    ax = fig.add_subplot(133)
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{ninit}"].mean() for ninit in ninits]
        yerr = [
            2
            * res[prob][f"{method}_{ninit}"].std()
            / np.sqrt(len(res[prob][f"{method}_{ninit}"]))
            for ninit in ninits
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            ninits,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(ninits, ymean, lw=1, label=method, color=color, ls=ls)
    # Add Sobol
    ymean = [res[prob]["Quasi-random_10"].mean() for ninit in ninits]
    yerr = [
        2
        * res[prob]["Quasi-random_10"].std()
        / np.sqrt(len(res[prob]["Quasi-random_10"]))
        for ninit in ninits
    ]
    color, ls = method_styles["Quasi-random"]
    ax.errorbar(
        ninits,
        ymean,
        yerr=yerr,
        lw=1,
        alpha=0.3,
        color=color,
        ls=ls,
    )
    ax.plot(ninits, ymean, lw=1, label="Quasi-random", color=color, ls=ls)
    ax.set_xticks(ninits)

    ax.set_ylim([0.12, 0.25])
    ax.set_yticks([0.12, 0.15, 0.18, 0.21, 0.24])
    ax.grid(alpha=0.1)
    ax.set_xlabel("Initial design size")
    ax.set_title("Contrast Sensitivity (6-d)")

    fig.subplots_adjust(bottom=0.34, left=0.093, top=0.91, right=0.99, wspace=0.3)

    plt.savefig("pdfs/init_sensitivity.pdf", pad_inches=0)


if __name__ == "__main__":
    make_init_sensitivity_figure()
