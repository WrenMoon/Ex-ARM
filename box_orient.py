"""
bottle_orient.py — Scripted bottle-orientation demo for the LEAP Hand.

Plays back a hard-coded sequence of hand poses designed to demonstrate
picking up and reorienting a cylindrical object (e.g. a bottle).
Each pose is held for a configurable duration before transitioning to
the next, creating a smooth choreographed motion.

Run directly:  python bottle_orient.py
"""

import numpy as np
import time
from utils.ExARM import ExArm
from utils.Constants import Connection


# ------------------------------------------------------------------ #
# Pose sequence                                                        #
# ------------------------------------------------------------------ #
# Each pose is a dict with:
#   duration  — how long (seconds) to hold the pose
#   index     — [abduct, flex, pip, dip] angles in degrees for the index finger
#   middle    — [abduct, flex, pip, dip] for the middle finger
#   ring      — [abduct, flex, pip, dip] for the ring finger
#   thumb     — [base_rot, mcp, pip, dip] for the thumb
POSES = [

    dict(
        duration=5.0,
        index=[0.0, 30.0, 0.0, 0.0],
        middle=[0.0, 30.0, 0.0, 0.0],
        ring=[0.0, 30.0, 0.0, 0.0],
        thumb=[0.0, -90.0, 0.0, 0.0],
    ),

    dict(
        duration=0.5,
        index=[0.0, 30.0, 0.0, 0.0],
        middle=[0.0, 30.0, 0.0, 0.0],
        ring=[-75.0, 95.0, 90.0, -5.0],
        thumb=[0.0, -18.0, 45.0, 100.0],
    ),

    dict(
        duration=0.2,
        index=[0.0, 30.0, 0.0, 0.0],
        middle=[0.0, 30.0, 0.0, 0.0],
        ring=[-75.0, 95.0, 90.0, 45.0],
        thumb=[0.0, -18.0, 45.0, 100.0],
    ),

    dict(
        duration=0.5,
        index=[0.0, 30.0, 0.0, 0.0],
        middle=[0.0, 30.0, 0.0, 0.0],
        ring=[-75.0, 95.0, 90.0, 45.0],
        thumb=[0.0, -73.0, 45.0, 100.0],
    ),

    dict(
        duration=0.5,
        index=[0.0, 25.0, 110.0, 90.0],
        middle=[0.0, 65.0, 75.0, 50.0],
        ring=[-75.0, 95.0, 90.0, 45.0],
        thumb=[0.0, -73.0, 45.0, 100.0],
    ),

    dict(
        duration=0.3,
        index=[30.0, 55.0, 110.0, 35.0],
        middle=[-30.0, 65.0, 90.0, 50.0],
        ring=[-75.0, 95.0, 90.0, 45.0],
        thumb=[0.0, -73.0, 45.0, 100.0],
    ),

    dict(
        duration=0.3,
        index=[30.0, 55.0, 110.0, 35.0],
        middle=[-30.0, 65.0, 90.0, 50.0],
        ring=[-75.0, 95.0, 90.0, 0.0],
        thumb=[0.0, -73.0, 45.0, 0.0],
    ),

    dict(
        duration=0.3,
        index=[30.0, 55.0, 110.0, 35.0],
        middle=[-30.0, 65.0, 90.0, 50.0],
        ring=[-75.0, 30.0, 90.0, 0.0],
        thumb=[0.0, -73.0, 45.0, 0.0],
    ),

    dict(
        duration=0.5,
        index=[80.0, 10.0, 100.0, 20.0],
        middle=[-80.0, 10.0, 100.0, 20.0],
        ring=[-75.0, 10.0, 90.0, 0.0],
        thumb=[0.0, -73.0, 45.0, 0.0],
    ),

    dict(
        duration=0.5,
        index=[80.0, 10.0, 100.0, 20.0],
        middle=[-80.0, 10.0, 100.0, 20.0],
        ring=[-75.0, 10.0, 90.0, 0.0],
        thumb=[-57.7, -112.1, 91.7, 0.0],
    ),

    dict(
        duration=0.5,
        index=[80.0, 10.0, 100.0, 20.0],
        middle=[-80.0, 10.0, 100.0, 20.0],
        ring=[-75.0, 10.0, 90.0, 0.0],
        thumb=[-57.7, 13.6, 101.9, 20.4],
    ),

    dict(
        duration=0.5,
        index=[80.0, 10.0, 100.0, 20.0],
        middle=[-80.0, 10.0, 100.0, 20.0],
        ring=[-75.0, 10.0, 90.0, 0.0],
        thumb=[-57.7, -98.5, 180.0, 20.4],
    ),

]




def pose_to_array(pose):
    """
    Flatten a finger-grouped pose dict into a 16-element NumPy array.

    The output ordering is: index(4) + middle(4) + ring(4) + thumb(4),
    matching the logical joint convention used throughout the codebase.

    Parameters
    ----------
    pose : dict with keys 'index', 'middle', 'ring', 'thumb' (each a 4-element list)

    Returns
    -------
    np.ndarray of shape (16,) and dtype float
    """
    return np.array(
        pose["index"] + pose["middle"] + pose["ring"] + pose["thumb"],
        dtype=float,
    )


def main(**kwargs):
    """
    Initialise the LEAP Hand in 'both' (real + sim) mode and execute
    the POSES sequence from start to finish.

    The inner loop re-sends the goal at ~33 Hz so the hardware servo
    controller continuously receives position commands (preventing
    the motors from going slack between updates).
    """
    leap_hand = ExArm(
        mode="both",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

    leap_hand.set_torque_enabled(True)

    UPDATE_PERIOD = 0.03  # seconds between successive goal-position writes (~33 Hz)

    for pose in POSES:
        target = pose_to_array(pose)
        start  = time.time()

        # Hold the current target for the specified duration, re-sending at UPDATE_PERIOD
        while time.time() - start < pose["duration"]:
            leap_hand.set_goal_positions_degree(target)
            leap_hand.set_torque_enabled(True)
            time.sleep(UPDATE_PERIOD)


if __name__ == "__main__":
    main()