import numpy as np
import time
from utils.ExARM import ExArm


POSES = [

    dict(
        duration=5,
        index  =[0, 0, 0, 0],
        middle =[0, 0, 0, 0],
        ring   =[0, 0, 0, 0],
        thumb  =[0, 0, 0, 0],
    ),

    dict(
        duration=2,
        index  =[70, 22, 85, 22],
        middle =[0, 0, 0, 0],
        ring   =[-70, 22, 85, 22],
        thumb  =[-45, -100, 30, 42],
    ),

    dict(
        duration=1,
        index  =[70, 22, 85, 45],
        middle =[0, 0, 0, 0],
        ring   =[-70, 22, 85, 45],
        thumb  =[-65, -100, 30, 42],
    ),

    dict(
        duration=1,
        index  =[70, 0, 85, 0],
        middle =[0, 0, 0, 0],
        ring   =[0, 22, 85, 0],
        thumb  =[-65, -45, 37, 50],
    ),

    dict(
        duration=1,
        index  =[0, 0, 0, 0],
        middle =[-30, 0, 30, 30],
        ring   =[0, 45, 85, 0],
        thumb  =[-65, -45, 37, 50],
    ),

    dict(
        duration=2,
        index  =[0, 55, 0, 0],
        middle =[0, 55, 55, 37],
        ring   =[0, 55, 55, 37],
        thumb  =[-120, -45, 10, 70],
    ),

    dict(
        duration=0.5,
        index  =[0, 55, 55, 37],
        middle =[0, 55, 55, 37],
        ring   =[0, 55, 55, 37],
        thumb  =[-120, -45, 10, 70],
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