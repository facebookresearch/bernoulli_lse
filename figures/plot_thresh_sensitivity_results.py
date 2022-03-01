import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append("..")

from plot_config import *


# need cameraready for original thresh
rundata = list(Path("../data/cameraready/").glob("*out.csv"))
sensitivity_rundata = list(Path("../data/thresh_sensitivity").glob("*out.csv"))


def compile_results():
    orig_dfiles = pd.concat([pd.read_csv(d) for d in rundata])
    thresh_dfiles = pd.concat([pd.read_csv(d) for d in sensitivity_rundata])
    # the hartmann thresh=0.5 runs were duplicated so drop them
    idx = np.logical_not(np.logical_and(
        thresh_dfiles.problem == "hartmann6_binary",
        thresh_dfiles.opt_strat_target == 0.5,
    ))


    df = pd.concat([orig_dfiles, thresh_dfiles[idx]])

    df["method"] = df.OptimizeAcqfGenerator_acqf.fillna("Quasi-random").astype(
        "category"
    )
    df["method"] = df.method.cat.rename_categories(
        {
            "MCLevelSetEstimation": "Straddle",
            "MCPosteriorVariance": "BALV",  # comment me out for largerun
            "BernoulliMCMutualInformation": "BALD",  # comment me out for largerun
        }
    )

    df = df[df.final == True]
    acqfs = [
        "EAVC",
        "LocalMI",
        "GlobalMI",
    ]

    problems = list(df["problem"].unique())
    problems.remove("discrim_lowdim")
    res = {prob: {} for prob in problems}
    for levels, g in df.groupby(["method", "rep", "problem", "opt_strat_target"]):
        method, _, prob, target = levels
        if method in acqfs and prob in problems:
            acqf = f"{method}_{target}"
            if acqf not in res[prob]:
                res[prob][acqf] = []
            res[prob][acqf].append(g["brier"].item())

    for prob in problems:
        for acqf in res[prob]:
            res[prob][acqf] = np.array(res[prob][acqf])

    return res



def make_thresh_sensitivity_figure():
    res = compile_results()

    methods = [
        "EAVC",
        "LocalMI",
        "GlobalMI",
        #"Quasi-random"  # Not run for this study
    ]
    # Make the plot
    fig = plt.figure(figsize=(6.75, 2.3))

    prob = "hartmann6_binary"
    threshes = [0.5, 0.65, 0.95]
    ax = fig.add_subplot(131)
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{thresh}"].mean() for thresh in threshes]
        yerr = [
            2
            * res[prob][f"{method}_{thresh}"].std()
            / np.sqrt(len(res[prob][f"{method}_{thresh}"]))
            for thresh in threshes
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            threshes,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(threshes, ymean, color=color, ls=ls, label=method, lw=1)
    ax.set_xticks(threshes)

    ax.grid(alpha=0.1)
    ax.set_title("Binarized Hartmann6 (6-d)")
    ax.set_xlabel("Target threshold")
    ax.set_ylabel("Final Brier Score\nafter 750 iterations")
    ax.legend(loc="lower left", bbox_to_anchor=(0.9, -0.63), ncol=3)

    prob = "discrim_highdim"
    ax = fig.add_subplot(132)
    threshes = [0.65, 0.75, 0.95]
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{thresh}"].mean() for thresh in threshes]
        yerr = [
            2
            * res[prob][f"{method}_{thresh}"].std()
            / np.sqrt(len(res[prob][f"{method}_{thresh}"]))
            for thresh in threshes
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            threshes,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(threshes, ymean, color=color, ls=ls, label=method, lw=1)
    ax.set_xticks(threshes)
    ax.grid(alpha=0.1)
    ax.set_title("Psych. Discrimination (8-d)")
    ax.set_xlabel("Target threshold")

    prob = "contrast_sensitivity_6d"
    ax = fig.add_subplot(133)
    for i, method in enumerate(methods):
        ymean = [res[prob][f"{method}_{thresh}"].mean() for thresh in threshes]
        yerr = [
            2
            * res[prob][f"{method}_{thresh}"].std()
            / np.sqrt(len(res[prob][f"{method}_{thresh}"]))
            for thresh in threshes
        ]
        color, ls = method_styles[method]
        ax.errorbar(
            threshes,
            ymean,
            yerr=yerr,
            lw=1,
            alpha=0.3,
            color=color,
            ls=ls,
        )
        ax.plot(threshes, ymean, color=color, ls=ls, label=method, lw=1)
    ax.set_xticks(threshes)
    ax.grid(alpha=0.1)
    ax.set_xlabel("Target threshold")
    ax.set_title("Contrast Sensitivity (6-d)")

    fig.subplots_adjust(bottom=0.34, left=0.09, top=0.91, right=0.99, wspace=0.3)

    plt.savefig("pdfs/thresh_sensitivity.pdf", pad_inches=0)


if __name__ == "__main__":
    make_thresh_sensitivity_figure()
