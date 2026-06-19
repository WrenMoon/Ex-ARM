from utils.ExARM import ExArm
from utils.planar_kinematics import PlanarFinger
from utils.leap_kinematics import URDFFinger
import time
import numpy as np


leap_hand = ExArm(
    mode="sim",
    ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    port="COM5",
    baudrate=4000000,
    offsets=[0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    model_path="Data/mujoco_robot.urdf"
)

# =========================
# FINGERS
# =========================
index = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_tip")
middle = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_2_tip")
ring = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_3_tip")
thumb = URDFFinger("Data/robot.urdf", "palm_lower_left", "thumb_fingertip_tip")

fingers = [index, middle, ring, thumb]

print(index.fk([0,0,0,0]))
print(index.fk(np.radians([0,0,0,90])))

print(index.joint_positions([0,0,0,0]))