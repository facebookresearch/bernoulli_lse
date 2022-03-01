import numpy as np

from contrast_discrimination.helpers import HalfGrating

from psychopy import visual, monitors

screen = monitors.Monitor("testMonitor", gamma=1)


win = visual.Window(
    allowGUI=True,
    units="deg",
    monitor=screen,
    bpc=(8, 8, 8),
    size=[300, 300],
    fullscr=False,
)
base_args = {
    "pedestal": 0,
    "contrast": 1,
    "orientation": 0,
    "temporal_frequency": 0,  # no animation so we screenshot
    "spatial_frequency": 10,
    "size": 10,
    "eccentricity": 0,  # just plot in the middle so we can screenshot
    "angle_dist": 0,  # not used in synth test function
}

p100_args = {
    "pedestal": [-0.5],
    "orientation": [
        60
    ],  # not used in the synth test function, but we used it in the plot in the paper
    "spatial_frequency": [3],
    "size": [7],
}

p75_args = {
    "pedestal": [-1.2],
    "spatial_frequency": [2],
    "size": [2.5],
}

p50_args = {
    "pedestal": [-1.5],
    "contrast": [-1.5],
}


def make_stim_image(args):
    stim = HalfGrating(**base_args, win=win)
    stim.update(args)
    image = stim.get_texture(phase=0, noisy_half="left")
    bg_color = np.array([stim.pedestal_psychopy_scale] * 3)
    win.setColor(bg_color)
    win.color = bg_color
    win.flip()
    stim._stim.image = image
    stim._stim.draw()
    win.flip()
    frame = win.getMovieFrame()
    return frame


if __name__ == "__main__":

    f50 = make_stim_image(p50_args)
    f50.save("pdfs/p50.png")

    f75 = make_stim_image(p75_args)
    f75.save("pdfs/p75.png")

    f100 = make_stim_image(p100_args)
    f100.save("pdfs/p100.png")
