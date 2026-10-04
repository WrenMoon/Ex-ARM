import numpy as np

from pynput import keyboard
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
        model_path="Data/mujoco_robot.urdf"
    )

grip = dict(
    start_angles=[0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 180, 0, 0],
    max_angles=[-75, 90, 90, 60,
                0, 80, 80, 65,
                75, 90, 90, 60,
                0, 100, 110, 80],
    step_sizes=[-1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, -1, 1, 1],
    max_currents= [22, 80, 25, 20,
                   22, 80, 25, 20,
                   22, 80, 25, 20,
                   25, 70, 25, 20],
    reached_target=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
)

current_positions = grip["start_angles"]
leap_hand.set_torque_enabled(True)
leap_hand.set_goal_positions_degree(current_positions)
time.sleep(2)

printer = BoxPrinter()
display_state = [np.zeros(16) for _ in range(3)]

printer.add_box("Positions", display_state[0], 4, 4, ".3f")
printer.add_box("Velocities", display_state[1], 4, 4, ".3f")
printer.add_box("Currents", display_state[2], 4, 4, ".3f")

EndLoop = False

while not EndLoop:    
    state = leap_hand.get_state()
    real_state = state.get("real") if isinstance(state, dict) else state
    if not isinstance(real_state, (tuple, list)) or len(real_state) != 3:
        print("Unable to read real hand state; skipping this step.")
        time.sleep(0.5)
        continue
    
    positions, velocities, currents = real_state
    for display, values in zip(display_state, real_state):
            display[:] = values
    # printer.print()


    leap_hand.set_goal_positions_degree(current_positions)

    for i in range(16):
        if not grip["reached_target"][i]:
            if grip["step_sizes"][i] >= 0:
                if currents[i] > grip["max_currents"][i]:
                    grip["max_currents"][i] = currents[i]
                    print(f"Current limit exceeded on joint {i}: {currents[i]} mA")
                    print(grip["max_currents"])
                    grip["reached_target"][i] = True
                elif current_positions[i] < grip["max_angles"][i]:
                    current_positions[i] += grip["step_sizes"][i]
                if current_positions[i] > grip["max_angles"][i]:
                    current_positions[i] = grip["max_angles"][i]
                    grip["reached_target"][i] = True
            else:
                if currents[i] < -grip["max_currents"][i]:
                    grip["max_currents"][i] = -currents[i]
                    print(f"Current limit exceeded on joint {i}: {currents[i]} mA")
                    print(grip["max_currents"])
                    grip["reached_target"][i] = True
                elif current_positions[i] > grip["max_angles"][i]:
                    current_positions[i] += grip["step_sizes"][i]
                if current_positions[i] < grip["max_angles"][i]:
                    current_positions[i] = grip["max_angles"][i]
                    grip["reached_target"][i] = True
        if abs(grip["max_angles"][i] - current_positions[i]) < 1:
            grip["reached_target"][i] = True

        
    if all(grip["reached_target"]):
        EndLoop = True
        print("Grasp Completed")

    print(grip["reached_target"])

    

    time.sleep(0.2)

