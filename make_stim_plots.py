import pickle

import numpy as np
import torch
from botorch.utils.sampling import (
    _convert_bounds_to_inequality_constraints,
    draw_sobol_samples,
)
from contrast_discrimination.helpers import HalfGrating
from models import RegularProbitGP
from problems import ContrastSensitivity
from psychopy import visual, monitors
from scipy.stats import norm

# Load the data
with open("data/8d_contrast_sensitivity_ms_20210927.pkl", "rb") as fin:
    data = pickle.load(fin)
m = RegularProbitGP(target_value=0.75)
m.fit(
    data["X"],
    data["y"],
    data["bounds"],
)

# eval over a grid and find some spots where f is close to 1
# close to 0.5, or close to 0.75
X_test = draw_sobol_samples(
    bounds=torch.tensor(data["bounds"].T, dtype=torch.double),
    n=1000,
    q=1,
    seed=12345,
).squeeze(1)

p_test = norm.cdf(
    torch.clamp(m.predict(torch.tensor(X_test, dtype=m.dtype))[0], min=0).numpy()
)

screen = monitors.Monitor("testMonitor", gamma=1)

win = visual.Window(
    allowGUI=True,
    units="deg",
    monitor=screen,
    bpc=(8, 8, 8),
    size=[1680, 1050],
    fullscr=False,
)

# param ranges:
# pedestal: -1.5 to 0 (logspace)
# contrast: -1.5 to 0 (logspace)
# orientation: 0 to 360
# temporal frequency: 0 to 20
# spatial frequency: 0.5 to 7
# size: 1 to 10
# angle_dist: 0 to 360
# eccentricity: 0 to 10


def construct_args(x, pack=True):
    x_ = x.numpy()
    args = {
        "pedestal": x_[0],
        "contrast": x_[1],
        "orientation": x_[2] / np.pi * 180,
        "temporal_frequency": 0,
        "spatial_frequency": x_[4],
        "size": x_[5],
        "angle_dist": x_[6] / np.pi * 180,
        "eccentricity": 0,
        # 'eccentricity': x_[7],
    }
    if pack:
        args = {k: np.array([v]) for k, v in args.items()}
    return args


x50 = X_test[p_test == 0.5][0]
x75 = X_test[np.argmin(np.abs(p_test - 0.75))]
x100 = X_test[np.argmax(p_test)]

stim = HalfGrating(**construct_args(x75, pack=False), win=win)
stim.update(construct_args(x75))

bg_color = np.array([stim.pedestal_psychopy_scale] * 3)
win.setColor(bg_color)
win.color = bg_color
win.flip()
stim.draw()
