import numpy as np
import time
from utils.ExARM import ExArm


POSES = [

    dict(
        duration=1.0,
        index=[0.0, 56.9, 0.0, 0.0],
        middle=[0.0, 0.0, 0.0, 0.0],
        ring=[0.0, 0.0, 0.0, 0.0],
        thumb=[0.0, 0.0, 0.0, 0.0],
    ),

    dict(
        duration=1.0,
        index=[0.0, 56.9, 0.0, 0.0],
        middle=[0.0, 55.0, 0.0, 0.0],
        ring=[0.0, 0.0, 0.0, 0.0],
        thumb=[0.0, 0.0, 0.0, 0.0],
    ),

    dict(
        duration=1.0,
        index=[0.0, 56.9, 0.0, 0.0],
        middle=[0.0, 55.0, 0.0, 0.0],
        ring=[0.0, 55.0, 0.0, 0.0],
        thumb=[0.0, 0.0, 0.0, 0.0],
    ),

    dict(
        duration=1.0,
        index=[0.0, 56.9, 0.0, 0.0],
        middle=[0.0, 55.0, 0.0, 0.0],
        ring=[0.0, 0.0, 0.0, 0.0],
        thumb=[0.0, 0.0, 0.0, 0.0],
    ),

    dict(
        duration=1.0,
        index=[0.0, 56.9, 0.0, 0.0],
        middle=[0.0, 55.0, 0.0, 0.0],
        ring=[0.0, 0.0, 0.0, 0.0],
        thumb=[0.0, 0.0, 0.0, 0.0],
    ),

]



def pose_to_array(pose):
    return np.array(
        pose["index"]
        + pose["middle"]
        + pose["ring"]
        + pose["thumb"],
        dtype=float,
    )


def main(**kwargs):

    leap_hand = ExArm(
        mode="both",
        ids=[0, 1, 2, 3, 4, 5, 6, 7,
             8, 9, 10, 11, 12, 13, 14, 15],
        port="COM5",
        baudrate=4000000,
        offsets=[0, -90, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0],
        model_path="Data/mujoco_robot.urdf"
    )

    leap_hand.set_torque_enabled(True)

    UPDATE_PERIOD = 0.03  # seconds

    for pose in POSES:

        target = pose_to_array(pose)

        start = time.time()

        while time.time() - start < pose["duration"]:
            leap_hand.set_goal_positions_degree(target)
            leap_hand.set_torque_enabled(True)
            time.sleep(UPDATE_PERIOD)


if __name__ == "__main__":
    main()