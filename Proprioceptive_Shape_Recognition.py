"""Estimate basic object shape from LEAP Hand joint state and motor current.

Live mode passively watches a real hand. A per-finger current change marks a
contact event, and measured joint positions are converted to fingertip points
in the palm frame. Keep the palm and object fixed relative to one another
while collecting points. This is an experimental sphere-vs-box baseline, not
a general shape-reconstruction system.

Run ``python Proprioceptive_Shape_Recognition.py --demo`` for an offline check.
"""

import argparse
import time

import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
from utils.LeapKinematics import LeapKinematics


MIN_CONTACTS = 8
MAX_NORMALIZED_ERROR = 0.18
MIN_SCORE_MARGIN = 0.04


def _fit_sphere(points):
    """Return the algebraic sphere fit and normalized radial RMS error."""
    points = np.asarray(points, dtype=float)
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    target = np.sum(points ** 2, axis=1)
    solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
    center = solution[:3]
    radius_squared = solution[3] + np.dot(center, center)
    if radius_squared <= 0:
        return None

    radius = np.sqrt(radius_squared)
    errors = np.linalg.norm(points - center, axis=1) - radius
    normalized_error = np.sqrt(np.mean(errors ** 2)) / max(radius, 1e-9)
    return {"center_m": center, "radius_m": radius, "error": normalized_error}


def _fit_box(points):
    """Fit a PCA-aligned box and score how close points are to its faces."""
    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    _, axes = np.linalg.eigh(np.cov(points.T))
    local_points = (points - center) @ axes
    half_sizes = np.max(np.abs(local_points), axis=0)
    if np.min(half_sizes) <= 1e-9:
        return None

    face_distances = np.abs(np.abs(local_points) - half_sizes)
    errors = np.min(face_distances, axis=1)
    scale = np.linalg.norm(half_sizes)
    normalized_error = np.sqrt(np.mean(errors ** 2)) / max(scale, 1e-9)
    return {
        "center_m": center,
        "size_m": 2.0 * half_sizes,
        "error": normalized_error,
    }


def classify_contact_points(points):
    """Classify palm-frame contact points as sphere, box, or ambiguous."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if len(points) < MIN_CONTACTS:
        return {"shape": "ambiguous", "reason": f"need at least {MIN_CONTACTS} contacts"}
    if not np.all(np.isfinite(points)):
        raise ValueError("points must contain only finite values")

    sphere = _fit_sphere(points)
    box = _fit_box(points)
    if sphere is None or box is None:
        return {"shape": "ambiguous", "reason": "degenerate contact geometry"}

    scores = {"sphere": sphere["error"], "box": box["error"]}
    ordered = sorted(scores, key=scores.get)
    best, second = ordered
    if scores[best] > MAX_NORMALIZED_ERROR:
        shape = "ambiguous"
        reason = "neither surface fit is sufficiently close"
    elif scores[second] - scores[best] < MIN_SCORE_MARGIN:
        shape = "ambiguous"
        reason = "sphere and box fits are too similar"
    else:
        shape = best
        reason = ""

    return {
        "shape": shape,
        "reason": reason,
        "scores": scores,
        "sphere": sphere,
        "box": box,
    }


def _signed_current(raw_current):
    """Interpret Dynamixel two's-complement current readings as signed values."""
    raw_current = np.asarray(raw_current, dtype=np.int64)
    return (raw_current + 32768) % 65536 - 32768


def _read_currents(hand):
    """Return a normalized 16-joint current vector from either backend mode."""
    state = hand.get_state()
    if isinstance(state, dict):
        state = state.get("real") or state.get("sim")
    if not isinstance(state, (tuple, list)) or len(state) < 3:
        raise RuntimeError(f"Unexpected hand state format: {type(state)} {state!r}")

    currents = np.asarray(state[2], dtype=np.int64)
    if currents.size != 16:
        raise RuntimeError(f"Expected 16 motor currents, got shape {currents.shape}: {currents!r}")
    return currents


def _finger_slice(finger_idx):
    """Return the logical joint slice for a single finger."""
    return slice(finger_idx * 4, finger_idx * 4 + 4)


def collect_live_contacts(threshold, output_path, close_step_deg=1.0, max_close_deg=90.0):
    """Slowly close all fingers and stop each finger independently on contact."""
    hand = ExArm(
        mode=Connection.mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )
    kinematics = LeapKinematics()
    contacts = []

    try:
        print("Keep the hand stationary and unloaded for current calibration...")
        calibration = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                currents = _read_currents(hand)
            except RuntimeError:
                time.sleep(0.02)
                continue
            if currents.size == 16:
                calibration.append(_signed_current(currents))
            time.sleep(0.02)

        if not calibration:
            raise RuntimeError("Unable to acquire a valid current baseline from the hand.")
        baseline = np.median(np.asarray(calibration, dtype=np.int64), axis=0)

        print("Calibration complete. The hand will close slowly and stop each finger at contact.")
        print("Press Ctrl+C to stop early and classify the collected contacts.")

        state = hand.get_state()
        if isinstance(state, dict):
            state = state.get("real") or state.get("sim")
        positions_deg = np.asarray(state[0], dtype=float)
        goal_positions = positions_deg.copy()
        stopped = np.zeros(4, dtype=bool)

        while np.any(~stopped):
            state = hand.get_state()
            if isinstance(state, dict):
                state = state.get("real") or state.get("sim")
            positions_deg = np.asarray(state[0], dtype=float)
            currents = _read_currents(hand)
            current_delta = np.abs(_signed_current(currents) - baseline)

            for finger_idx in range(4):
                if stopped[finger_idx]:
                    continue

                finger_slice = _finger_slice(finger_idx)
                finger_delta = current_delta[finger_slice]

                if np.max(finger_delta) >= threshold:
                    stopped[finger_idx] = True
                    point, _ = kinematics.fk_finger(
                        finger_idx, np.radians(positions_deg[finger_slice])
                    )
                    contacts.append(point)
                    print(
                        f"Contact {len(contacts)}: {kinematics.FINGER_NAMES[finger_idx]} "
                        f"tip at [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}] m"
                    )
                    continue

                finger_goal = goal_positions[finger_slice].copy()
                if np.max(np.abs(finger_goal[finger_slice][1:])) >= max_close_deg:
                    stopped[finger_idx] = True
                    continue

                finger_goal[1:] = finger_goal[1:] + close_step_deg
                finger_goal[1:] = np.clip(finger_goal[1:], 0.0, max_close_deg)
                goal_positions[finger_slice] = finger_goal

            hand.set_goal_positions_degree(goal_positions)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        hand.close()

    if contacts:
        points = np.asarray(contacts)
        np.savetxt(output_path, points, delimiter=",", header="x_m,y_m,z_m", comments="")
        result = classify_contact_points(points)
        print(f"\nSaved {len(points)} contact points to {output_path}")
        print(f"Estimated shape: {result['shape']}")
        if result.get("scores"):
            print("Normalized fit errors:", result["scores"])
            if result["shape"] == "sphere":
                sphere = result["sphere"]
                print(f"Estimated diameter: {2 * sphere['radius_m'] * 1000:.1f} mm")
                print(f"Estimated center (palm frame): {sphere['center_m'] * 1000} mm")
            elif result["shape"] == "box":
                box = result["box"]
                print(f"Estimated box dimensions: {box['size_m'] * 1000} mm")
                print(f"Estimated center (palm frame): {box['center_m'] * 1000} mm")
        if result.get("reason"):
            print("Reason:", result["reason"])
    else:
        print("No contact events collected; check the current threshold and baseline.")


def run_demo():
    """Exercise sphere, box, and insufficient-data decisions without hardware."""
    rng = np.random.default_rng(7)
    directions = rng.normal(size=(48, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sphere_points = 0.04 * directions + np.array([0.02, -0.01, 0.03])

    box_points = []
    half_size = np.array([0.035, 0.025, 0.03])
    for axis in range(3):
        for sign in (-1.0, 1.0):
            for _ in range(8):
                point = rng.uniform(-half_size, half_size)
                point[axis] = sign * half_size[axis]
                box_points.append(point)

    results = {
        "sphere": classify_contact_points(sphere_points),
        "box": classify_contact_points(np.asarray(box_points)),
        "insufficient": classify_contact_points(sphere_points[:4]),
    }
    for expected, result in results.items():
        print(f"{expected}: {result['shape']} {result.get('scores', result.get('reason', ''))}")

    assert results["sphere"]["shape"] == "sphere"
    assert results["box"]["shape"] == "box"
    assert results["insufficient"]["shape"] == "ambiguous"
    print("Demo checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run synthetic checks without hardware")
    parser.add_argument("--current-threshold", type=float, default=25.0)
    parser.add_argument("--output", default="shape_contact_points.csv")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        collect_live_contacts(args.current_threshold, args.output)


if __name__ == "__main__":
    main()