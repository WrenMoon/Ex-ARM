import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
import time


leap_hand = ExArm(
        mode="both",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

leap_hand.set_torque_enabled(True)

grip1 = dict(
    index=[-60.0, 65.0, 60.0, 70.0],
    middle=[0.0, 70.0, 15.0, 60.0],
    ring=[60.0, 65.0, 60.0, 70.0],
    thumb=[5.0, 130.0, 85.0, 75.0],
)

pose = np.array(
    grip1["index"] + grip1["middle"] + grip1["ring"] + grip1["thumb"],
    dtype=float,
)

current_limit = 40  # XL330 Present Current is reported in mA.
endLoop = False

while not endLoop:
    # leap_hand.set_goal_positions_degree([0, i, i, i, 0, i, i, i, 
    #                                       0, i, i, i, 0, 0, 0, 0])
    leap_hand.set_goal_positions_degree(pose)
    # leap_hand.set_goal_positions_degree(np.zeros(16))
    state = leap_hand.get_state()
    real_state = state.get("real") if isinstance(state, dict) else state
    if not isinstance(real_state, (tuple, list)) or len(real_state) != 3:
        print("Unable to read real hand state; skipping this step.")
        time.sleep(0.5)
        continue

    positions, velocities, currents = real_state
    print("Currents (mA):", currents)

    if abs(currents[2]) < current_limit:
        pose[2] += 1.0
    else:
        print("Current limit exceeded on joint 2, stopping increment.")

    if abs(currents[6]) < current_limit:
        pose[6] += 1.0
    else:
        print("Current limit exceeded on joint 6, stopping increment.")

    if abs(currents[10]) < current_limit:
        pose[10] += 1.0
    else:
        print("Current limit exceeded on joint 10, stopping increment.")

    if abs(currents[13]) < current_limit:
        pose[13] -= 1.0
    else:
        print("Current limit exceeded on joint 13, stopping decrement.")

    if abs(currents[2]) > current_limit and abs(currents[6]) > current_limit and abs(currents[10]) > current_limit and abs(currents[13]) > current_limit:
        print("Grasp Completed")
        endLoop = True
        
    
    time.sleep(0.5)