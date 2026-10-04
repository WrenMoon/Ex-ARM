import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
import time

leap_hand = ExArm(
        mode=Connection.mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

grip = dict(
    start_angles=[0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0],
    max_angles=[-75, 90, 90, 60,
                0, 80, 80, 65,
                75, 90, 90, 60,
                0, 100, 110, 80],
    step_sizes=[-1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1],
    max_currents= 40
)

current_positions = grip["start_angles"]
leap_hand.set_torque_enabled(True)

print("\033[2J\033[H", end="")

while True:
    state = leap_hand.get_state()
    real_state = state.get("real") if isinstance(state, dict) else state
    if not isinstance(real_state, (tuple, list)) or len(real_state) != 3:
        print("Unable to read real hand state; skipping this step.")
        time.sleep(0.5)
        continue
    
    positions, velocities, currents = real_state

    print("\033[H", end="")
    print("────────────────────────────────────────")
    for i in range(4):
        print("│", end="")
        for j in range(4):
            print(f"{current_positions[i * 4 + j]:8.3f} │", end="")
        print()
        if i < 3:
            print("────────────────────────────────────────")
    print("────────────────────────────────────────")


    for i in range(16):
        if abs(currents[i]) > grip["max_currents"]:
            print(f"Current limit exceeded on joint {i}: {currents[i]} mA")
            current_positions[i] = positions[i]  # Reset to current position
        elif current_positions[i] < grip["max_angles"][i]:
            current_positions[i] += grip["step_sizes"][i]
        elif current_positions[i] > grip["max_angles"][i]:
            current_positions[i] = grip["max_angles"][i]

    leap_hand.set_goal_positions_degree(current_positions)
    

    time.sleep(0.01)
