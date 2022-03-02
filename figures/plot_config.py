# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc
import matplotlib
rc('font', family='serif', style='normal', variant='normal', weight='normal', stretch='normal', size=8)
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['axes.titlesize'] = 9

cmap = plt.get_cmap("tab10")

method_styles = {
    'Straddle': (cmap(0), ':'),
    'EAVC': (cmap(1), '-'),
    'LocalMI': (cmap(2), '--'),
    'GlobalMI': (cmap(3), '-'),
    'LocalSUR': (cmap(4), '--'),
    'GlobalSUR': (cmap(5), '-'),
    'ApproxGlobalSUR': (cmap(6), '-'),
    'Quasi-random': (cmap(7), ':'),
    'BALD': (cmap(8), ':'),
    'BALV': (cmap(9), ':'),
}

model_to_method_name = {
    'MCLevelSetEstimation': 'Straddle',
    'EAVC': 'EAVC',
    'LocalMI': 'LocalMI',
    'GlobalMI': 'GlobalMI',
    'LocalSUR': 'LocalSUR',
    'GlobalSUR': 'GlobalSUR',
    'ApproxGlobalSUR': 'ApproxGlobalSUR',
    'random': 'Quasi-random',
}

mean_covar_config = {
    "fixed_mean": False,
    "lengthscale_prior": "gamma",
    "outputscale_prior": "gamma",
    "kernel": "RBFKernel",
}
