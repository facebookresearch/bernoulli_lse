import numpy as np

import torch
from aepsych.benchmark.test_functions import (
    modified_hartmann6,
    discrim_highdim,
    novel_discrimination_testfun,
)

from aepsych.models import GPClassificationModel

from aepsych.benchmark.problem import LSEProblem


class LSEProblemWithEdgeLogging(LSEProblem):
    eps = 0.05

    def evaluate(self, strat):
        metrics = super().evaluate(strat)

        # add number of edge samples to the log

        # get the trials selected by the final strat only
        n_opt_trials = strat.strat_list[-1].n_trials

        lb, ub = strat.lb, strat.ub
        r = ub - lb
        lb2 = lb + self.eps * r
        ub2 = ub - self.eps * r

        near_edge = (
            np.logical_or(
                (strat.x[-n_opt_trials:, :] <= lb2), (strat.x[-n_opt_trials:, :] >= ub2)
            )
            .any(axis=-1)
            .double()
        )

        metrics["prop_edge_sampling_mean"] = near_edge.mean().item()
        metrics["prop_edge_sampling_err"] = (2 * near_edge.std() / np.sqrt(len(near_edge))).item()
        return metrics


class DiscrimLowDim(LSEProblemWithEdgeLogging):
    name = "discrim_lowdim"
    bounds = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.double).T
    threshold = 0.75

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(novel_discrimination_testfun(x), dtype=torch.double)


class DiscrimHighDim(LSEProblemWithEdgeLogging):
    name = "discrim_highdim"
    threshold = 0.75
    bounds = torch.tensor(
        [
            [-1, 1],
            [-1, 1],
            [0.5, 1.5],
            [0.05, 0.15],
            [0.05, 0.2],
            [0, 0.9],
            [0, 3.14 / 2],
            [0.5, 2],
        ],
        dtype=torch.double,
    ).T

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(discrim_highdim(x), dtype=torch.double)


class Hartmann6Binary(LSEProblemWithEdgeLogging):
    name = "hartmann6_binary"
    threshold = 0.5
    bounds = torch.stack(
        (
            torch.zeros(6, dtype=torch.double),
            torch.ones(6, dtype=torch.double),
        )
    )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        y = torch.tensor([modified_hartmann6(x) for x in X], dtype=torch.double)
        f = 3 * y - 2.0
        return f


class ContrastSensitivity6d(LSEProblemWithEdgeLogging):
    """
    Uses a surrogate model fit to real data from a constrast sensitivity study.
    """

    name = "contrast_sensitivity_6d"
    threshold = 0.75
    bounds = torch.tensor(
        [[-1.5, 0], [-1.5, 0], [0, 20], [0.5, 7], [1, 10], [0, 10]],
        dtype=torch.double,
    ).T

    def __init__(self):

        # Load the data
        self.data = np.loadtxt("data/csf_dataset.csv", delimiter=",", skiprows=1)
        y = torch.LongTensor(self.data[:, 0])
        x = torch.Tensor(self.data[:, 1:])

        # Fit a model, with a large number of inducing points
        self.m = GPClassificationModel(
            lb=self.bounds[0],
            ub=self.bounds[1],
            inducing_size=100,
            inducing_point_method="kmeans++",
        )

        self.m.fit(
            x,
            y,
        )

    def f(self, X: torch.Tensor) -> torch.Tensor:
        # clamp f to 0 since we expect p(x) to be lower-bounded at 0.5
        return torch.clamp(self.m.predict(torch.tensor(X))[0], min=0)
