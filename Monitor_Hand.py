"""Stream the hand's joint state as JSON for the project UI."""

import json
import math
import time

from utils.Constants import Connection
from utils.ExARM import ExArm


def main():
    hand = ExArm(
        mode=Connection.mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path=Connection.model_path,
    )
    try:
        while True:
            state = hand.get_state()
            if isinstance(state, dict):
                state = state.get("real") or state.get("sim")
            try:
                valid = (isinstance(state, (tuple, list)) and len(state) == 3
                         and all(len(values) == 16 for values in state))
            except TypeError:
                valid = False
            if valid:
                try:
                    values = [[float(value) for value in row] for row in state]
                    if all(math.isfinite(value) for row in values for value in row):
                        print(json.dumps(dict(
                            positions=values[0], velocities=values[1], currents=values[2],
                        )), flush=True)
                except (TypeError, ValueError):
                    pass
            time.sleep(0.1)
    finally:
        hand.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
