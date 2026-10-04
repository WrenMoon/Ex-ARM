"""Play a saved pose sequence on the LEAP Hand.

Run with ``mjpython pose_sequence_runner.py box_orient`` or
``mjpython pose_sequence_runner.py bottle_orient`` on macOS; use ``python``
instead of ``mjpython`` on other platforms.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm


POSE_SERIES_DIR = Path(__file__).resolve().parent / "Data" / "Example_pose_series"
FINGER_KEYS = ("index", "middle", "ring", "thumb")
UPDATE_PERIOD = 0.03


def load_pose_sequence(sequence_name):
    """Load and validate a named pose sequence from the example data folder."""
    if Path(sequence_name).name != sequence_name:
        raise ValueError("Sequence must be an example pose-series name.")

    sequence_path = POSE_SERIES_DIR / f"{sequence_name}.json"
    with sequence_path.open(encoding="utf-8") as sequence_file:
        sequence = json.load(sequence_file)

    poses = sequence.get("poses") if isinstance(sequence, dict) else None
    if not isinstance(poses, list) or not poses:
        raise ValueError(f"{sequence_path} must contain a non-empty 'poses' list.")

    for pose_index, pose in enumerate(poses):
        if not isinstance(pose, dict):
            raise ValueError(f"Pose {pose_index} must be a JSON object.")
        duration = pose.get("duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"Pose {pose_index} must have a positive finite duration.")

        for finger in FINGER_KEYS:
            angles = pose.get(finger)
            if (
                not isinstance(angles, list)
                or len(angles) != 4
                or any(
                    isinstance(angle, bool)
                    or not isinstance(angle, (int, float))
                    or not math.isfinite(angle)
                    for angle in angles
                )
            ):
                raise ValueError(
                    f"Pose {pose_index} must have four finite numeric angles for {finger}."
                )

    return poses


def pose_to_array(pose):
    """Flatten grouped finger angles to the 16-joint order used by ExArm."""
    return np.asarray(
        [angle for finger in FINGER_KEYS for angle in pose[finger]],
        dtype=float,
    )


def main():
    sequence_names = sorted(path.stem for path in POSE_SERIES_DIR.glob("*.json"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", choices=sequence_names, help="pose sequence to play")
    args = parser.parse_args()
    poses = load_pose_sequence(args.sequence)

    leap_hand = ExArm(
        mode=Connection.mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path=Connection.model_path,
    )
    leap_hand.set_torque_enabled(True)

    try:
        for pose in poses:
            target = pose_to_array(pose)
            start = time.monotonic()
            while time.monotonic() - start < pose["duration"]:
                leap_hand.set_goal_positions_degree(target)
                leap_hand.set_torque_enabled(True)
                time.sleep(UPDATE_PERIOD)
    finally:
        leap_hand.close()


if __name__ == "__main__":
    main()
