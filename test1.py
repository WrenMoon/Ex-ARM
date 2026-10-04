import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
from utils.BoxPrinter import BoxPrinter
import time

leap_hand = ExArm(
        mode=Connection.mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path=Connection.model_path
    )

printer = BoxPrinter()
display_state = [np.zeros(16) for _ in range(3)]

printer.add_box("Positions", display_state[0], 4, 4, ".3f")
printer.add_box("Velocities", display_state[1], 4, 4, ".3f")
printer.add_box("Currents", display_state[2], 4, 4, ".3f")

while True:
    state = leap_hand.get_state()
    latest = state.get("real") if isinstance(state, dict) else state
    if not isinstance(latest, (tuple, list)) or len(latest) != 3:
        time.sleep(0.5)
        continue

    for display, values in zip(display_state, latest):
        display[:] = values

    printer.print()
    time.sleep(0.01)