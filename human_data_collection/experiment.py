import numpy as np
import torch
from aepsych.server import AEPsychServer
from psychopy import core, data, event, gui, monitors, visual

from contrast_discrimination import config
from contrast_discrimination.helpers import HalfGrating


class ServerHelper:
    def __init__(self, config_path, db_path):
        self._server = AEPsychServer(database_path=db_path)
        with open(config_path, "r") as f:
            configstr = f.read()
        self._server.handle_setup_v01(
            {"type": "setup", "message": {"config_str": configstr}}
        )

    def ask(self):
        request = {"message": "", "type": "ask"}
        return self._server.handle_ask_v01(request)["config"]

    def tell(self, config, outcome):
        request = {
            "type": "tell",
            "message": {"config": config, "outcome": outcome},
        }
        self._server.handle_tell(request)


def run_experiment():

    seed = config.constants["seed"]
    config_path = config.constants["config_path"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    expInfo = {}
    expInfo["dateStr"] = data.getDateStr()  # add the current time
    # present a dialogue to change params
    dlg = gui.DlgFromDict(expInfo, title="multi-D JND Exp", fixed=["dateStr"])
    if not dlg.OK:
        core.quit()  # the user hit cancel so exit

    # where to save data
    fileName = "../data/csf_dataset"

    screen = monitors.Monitor("testMonitor", gamma=1)

    screen.setSizePix(config.psychopy_vars["setSizePix"])
    screen.setWidth(config.psychopy_vars["setWidth"])
    screen.setDistance(config.psychopy_vars["setDistance"])

    win = visual.Window(
        allowGUI=True,
        units="deg",
        monitor=screen,
        bpc=(8, 8, 8),
        size=config.psychopy_vars["setSizePix"],
        fullscr=False,
    )
    screen_text_g = visual.TextStim(win, text=None, alignHoriz="center", color="green")
    screen_text_r = visual.TextStim(win, text=None, alignHoriz="center", color="red")
    screen_text = visual.TextStim(win, text=None, alignHoriz="center", color="gray")

    # display instructions and wait
    message2 = visual.TextStim(
        win,
        pos=[0, +3],
        text="Hit the space bar key when ready and "
        "to advance to the next trial after you see a red cross.",
    )
    message1 = visual.TextStim(
        win,
        pos=[0, -3],
        text="You'll see a stimulus. One side will have a grating and the other will be noise."
        "  "
        "Press left or right corresponding to the side with noise. If you don't know, please guess.",
    )
    message1.draw()
    message2.draw()
    win.flip()  # to show our newly drawn 'stimuli'
    # pause until there's a keypress
    event.waitKeys()

    # start the trial: draw grating
    clock = core.Clock()

    screen_text_r.setText(
        "+"
    )  # this should update the fixation with the new background color, but it isnt for some reason
    screen_text_r.draw(win=win)
    win.flip()

    server_helper = ServerHelper(config_path=config_path, db_path=f"{fileName}.db")

    # create stimulus
    stim = HalfGrating(**config.base_params, win=win)
    i = 0
    while not server_helper._server.strat.finished:
        trial_params = server_helper.ask()
        stim.update(trial_params)

        bg_color = np.array([stim.pedestal_psychopy_scale] * 3)
        win.setColor(bg_color)
        win.color = bg_color
        win.flip()

        screen_text_r.setText("+")
        screen_text_r.draw(win=win)
        win.flip()

        fixation_keys = []
        while not fixation_keys:
            fixation_keys = event.getKeys(keyList=["space"])

        if "space" in fixation_keys:

            screen_text.setText("+")
            screen_text.draw(win=win)
            win.flip()

            noisy_half = "left" if np.random.randint(2) == 0 else "right"
            clock.reset()

            keys = stim.draw(
                noisy_half=noisy_half,
                win=win,
                pre_duration_s=config.psychopy_vars["pre_duration_s"],
                stim_duration_s=config.psychopy_vars["stim_duration_s"],
            )
            response = noisy_half in keys
            win.flip()

            if response:
                screen_text_g.setText("Correct")
                screen_text_g.draw()
                win.flip()
            else:
                screen_text_r.setText("Incorrect")
                screen_text_r.draw()
                win.flip()

            server_helper.tell(trial_params, response)
            event.clearEvents()
            i = i + 1

    win.close()

    pd_df = server_helper._server.get_dataframe_from_replay()
    pd_df.to_csv(fileName + ".csv", index=False)

    core.quit()


if __name__ == "__main__":

    run_experiment()
