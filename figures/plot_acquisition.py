# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from copy import deepcopy
import numpy as np
import torch

from botorch.utils.sampling import draw_sobol_samples

import sys 
sys.path.append('..')

from plot_config import *

from problems import DiscrimLowDim
from aepsych.models.gp_classification import GPClassificationModel
from aepsych.factory.factory import default_mean_covar_factory
from aepsych.config import Config
from aepsych.acquisition import (
    MCLevelSetEstimation,
    GlobalSUR,
    GlobalMI,
    EAVC,
    LocalMI,
    LocalSUR,
)


def make_figure():
    # Generate training data for the model
    prob = DiscrimLowDim()
        
    X = draw_sobol_samples(
        bounds=torch.tensor(prob.bounds, dtype=torch.double), n=20, q=1, seed=1403
    ).squeeze(1)
    np.random.seed(1403)
    y = torch.LongTensor([1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1])

    ###print(X)
    ###tensor([[ 0.2829,  0.2585],
        ###[-0.8400, -0.4620],
        ###[-0.3771,  0.8218],
        ###[ 0.9996, -0.8986],
        ###[ 0.5347,  0.6318],
        ###[-0.0918, -0.5853],
        ###[-0.6252,  0.1951],
        ###[ 0.2478, -0.0219],
        ###[ 0.0526,  0.9270],
        ###[-0.5706, -0.8485],
        ###[-0.1469,  0.4888],
        ###[ 0.7304, -0.2870],
        ###[ 0.8047,  0.0576],
        ###[-0.3227, -0.2292],
        ###[-0.8948,  0.6194],
        ###[ 0.4783, -0.6676],
        ###[ 0.3968,  0.5543],
        ###[-0.9803, -0.7246],
        ###[-0.3026,  0.1158],
        ###[ 0.8207, -0.1633]], dtype=torch.float64)

    # Fit a model
    lb, ub = prob.bounds
    config = deepcopy(mean_covar_config)
    config["lb"] = str(lb.tolist())
    config["ub"] = str(ub.tolist())
    mean, covar = default_mean_covar_factory(Config({"default_mean_covar_factory": config}))

    # Fit a model
    m = GPClassificationModel(lb=lb, ub=ub, mean_module=mean, covar_module=covar)
    m.fit(train_x=X, train_y=y)

    # Create a grid for plotting
    ngrid = 25
    xp = np.linspace(-1, 1, ngrid)
    yp = np.linspace(-1, 1, ngrid)
    xv, yv = np.meshgrid(xp, yp)
    x_plt = torch.tensor(np.vstack((xv.flatten(), yv.flatten())).T)

    # Make the plot
    fig = plt.figure(figsize=(6.75, 1.5))

    Xrnd = draw_sobol_samples(bounds=prob.bounds, n=512, q=1, seed=1000).squeeze(1)
    # Evaluate each acquisition fn on x_plt and Xrnd
    
    for i, acq in enumerate([
        MCLevelSetEstimation,
        LocalSUR,
        LocalMI,
        GlobalSUR,
        GlobalMI,
        EAVC,
    ]):
        if i == 0:
            acqf = acq(model=m, target=0.75, beta=3.84)
        elif i in [3, 4, 5]:
            acqf = acq(model=m, target=0.75, Xq=Xrnd)
        else:
            acqf = acq(model=m, target=0.75)
        ax = fig.add_subplot(1, 6, i + 1)
        vals_plt = acqf(x_plt.unsqueeze(1)).detach().numpy()
        vals_opt = acqf(Xrnd.unsqueeze(1)).detach().numpy()
        r = vals_plt.max() - vals_plt.min()
        levels = np.linspace(vals_plt.min() - 0.01 * r, vals_plt.max() + 0.01 * r, 30)
        ax.contourf(yv, xv, vals_plt.reshape(ngrid, ngrid), alpha=0.2, levels=levels)
        indx_max = np.argmax(vals_opt)
        ax.plot(Xrnd[indx_max, 1], Xrnd[indx_max, 0], 'r*', mew=0.5, ms=5, fillstyle='full')
        ax.set_xlim([-1.08, 1.08])
        ax.set_ylim([-1.08, 1.08])
        ax.set_title(model_to_method_name[acq.__name__])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel('$x_1$')
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('$x_2$')

    fig.subplots_adjust(wspace=0.08, left=0.058, right=0.995, top=0.87, bottom=0.23)
    plt.savefig('pdfs/acquisitions.pdf', pad_inches=0)


if __name__ == '__main__':
    make_figure()
