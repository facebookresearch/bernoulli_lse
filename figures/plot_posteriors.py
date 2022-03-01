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
from aepsych.acquisition.lookahead_utils import lookahead_at_xstar


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
    m = GPClassificationModel(lb=lb, ub=ub, mean_module=mean, covar_module=covar)
    m.fit(train_x=X, train_y=y)

    # Create a grid for plotting
    ngrid = 25
    xp = np.linspace(-1, 1, ngrid)
    yp = np.linspace(-1, 1, ngrid)
    xv, yv = np.meshgrid(xp, yp)
    x_plt = torch.tensor(np.vstack((xv.flatten(), yv.flatten())).T)
    indx_star = 165

    Xstar = x_plt[indx_star, :].unsqueeze(0)
    Xq = x_plt

    Px, P1, P0, py1 = lookahead_at_xstar(model=m, Xstar=Xstar, Xq=x_plt, gamma=prob.f_threshold)

    fig = plt.figure(figsize=(6.75, 1.75))

    # Plot the not-look-ahead level-set posterior
    axs = []
    ax = fig.add_subplot(131)
    axs.append(ax)
    ax.contourf(yv, xv, Px.detach().numpy().reshape(ngrid, ngrid), levels=np.linspace(0, 1, 20), alpha=0.2)
    ax.set_title('Level-set posterior\n$\pi(\mathbf{x} | \mathcal{D}_n)$')
    y_is_1 = y == 1
    ax.plot(X[y_is_1, 1], X[y_is_1, 0], 'kx', ls='None', mew=0.5, ms=5, alpha=0.3, label='$y=1$')
    ax.plot(X[~y_is_1, 1], X[~y_is_1, 0], 'ko', ls='None', mew=0.5, ms=5, fillstyle='none', alpha=0.3, label='$y=0$')
    ax.plot(x_plt[indx_star, 1], x_plt[indx_star, 0], 'r+', mew=1, ms=5, label='$\mathbf{x}_*$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.legend(loc='lower left', bbox_to_anchor=(3.8, 0.5))

    # Posterior under a 1 observation
    ax = fig.add_subplot(132)
    axs.append(ax)
    ax.contourf(yv, xv, P1.detach().numpy().reshape(ngrid, ngrid), levels=np.linspace(0, 1, 20), alpha=0.2)
    ax.set_title('Look-ahead posterior\n$\pi(\mathbf{x} | \mathcal{D}_{n+1}(\mathbf{x}_*, y_*=1))$')
    ax.plot(X[y_is_1, 1], X[y_is_1, 0], 'kx', ls='None', mew=0.5, ms=5, alpha=0.3)
    ax.plot(X[~y_is_1, 1], X[~y_is_1, 0], 'ko', ls='None', mew=0.5, ms=5, fillstyle='none', alpha=0.3)
    ax.plot(x_plt[indx_star, 1], x_plt[indx_star, 0], 'rx', ls='None', mew=1, ms=5)
    ax.set_xlabel('$x_1$')
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels([])

    # Posterior under a 0 observation
    ax = fig.add_subplot(133)
    axs.append(ax)
    cf = ax.contourf(yv, xv, P0.detach().numpy().reshape(ngrid, ngrid), levels=np.linspace(0, 1, 20), alpha=0.2)
    ax.set_title('Look-ahead posterior\n$\pi(\mathbf{x} | \mathcal{D}_{n+1}(\mathbf{x}_*, y_*=0))$')
    ax.plot(X[y_is_1, 1], X[y_is_1, 0], 'kx', ls='None', mew=0.5, ms=5, alpha=0.3)
    ax.plot(X[~y_is_1, 1], X[~y_is_1, 0], 'ko', ls='None', mew=0.5, ms=5, fillstyle='none', alpha=0.3)
    ax.plot(x_plt[indx_star, 1], x_plt[indx_star, 0], 'ro', ls='None', mew=1, ms=5, fillstyle='none')
    ax.set_xlabel('$x_1$')
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels([])

    fig.subplots_adjust(bottom=0.2, left=0.13, top=0.8, right=0.85)
    fig.colorbar(cf, ax=axs, ticks=[0, 0.25, 0.5, 0.75, 1.0], pad=0.01)

    plt.savefig('pdfs/posteriors.pdf', pad_inches=0)


if __name__ == '__main__':
    make_figure()
