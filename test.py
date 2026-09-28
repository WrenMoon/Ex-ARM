from utils.Constants import Connection
from utils.ExARM import ExArm
import time


leap_hand = ExArm(
        mode="real",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

i = 0;
while True:

    leap_hand.set_goal_positions_degree([0, i, i, i, 0, i, i, i,
                                          0, i, i, i, 0, 0, 0, 0])
    leap_hand.set_torque_enabled(True)
    positions, velocities, currents = leap_hand.get_state()
    if leap_hand.get_state() != [0,0,0]:
        print("Currents:", currents)
    i += 5
    time.sleep(0.5)