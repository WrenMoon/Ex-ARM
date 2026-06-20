from utils.ExARM import ExArm
import time
import numpy as np

arm = ExArm(
    mode="both",
    ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    port="COM5",
    baudrate=4000000,
    offsets=[0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    model_path="Data/mujoco_robot.urdf"
)

arm.set_torque_enabled(True)
arm.set_goal_positions_degree(np.zeros(16)*180)

positions, velocities, currents = arm.get_state()

arm.set_torque_enable(False)
arm.close_port()

time.sleep(100)