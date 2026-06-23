"""
Vision_Kinesthetic_Teaching.py — record a hand motion, solve IK offline,
and replay it smoothly on the LEAP Hand.

Pipeline:
  1. RECORD   : webcam → MediaPipe landmarks → raw landmark video file
  2. PROCESS  : landmarks → palm frame → fingertip targets → IK → joint angles
                (saved as a .npy trajectory)
  3. REPLAY   : trajectory → interpolated smooth motion → robot

The detection, palm-frame construction, and IK logic are identical to
Vision_Retargeting.py so calibration parameters (AXIS_MAP, PALM_OFFSET,
HAND_SCALE_M) remain valid.
"""

import os
import time
import pickle
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
from utils.LeapKinematics import LeapKinematics


# ------------------------------------------------------------------ #
# Paths and filenames                                                #
# ------------------------------------------------------------------ #

MODEL_PATH = "Data/hand_landmarker.task"
MOTION_DIR = "Data/Motions"
os.makedirs(MOTION_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand tracking model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")


# ------------------------------------------------------------------ #
# Calibration parameters (same as Vision_Retargeting.py)             #
# ------------------------------------------------------------------ #

HAND_SCALE_M = 0.233

AXIS_MAP = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
], dtype=float)

PALM_OFFSET = np.array([-0.085, -0.05, 0.04], dtype=float)

ALPHA = 0.25

IK_N_STARTS = 1
IK_TOL_M    = 5e-4
IK_MAX_ITER = 30


# ------------------------------------------------------------------ #
# MediaPipe landmark constants                                       #
# ------------------------------------------------------------------ #

WRIST      = 0
MCP_INDEX  = 5
MCP_MIDDLE = 9
MCP_RING   = 13
MCP_PINKY  = 17

TIP_INDICES  = [8, 12, 16, 4]
FINGER_NAMES = ["index", "middle", "ring", "thumb"]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ------------------------------------------------------------------ #
# Palm frame and target conversion (copied unchanged)                #
# ------------------------------------------------------------------ #

def build_palm_frame(pts: np.ndarray):
    wrist  = pts[WRIST]
    i_mcp  = pts[MCP_INDEX]
    m_mcp  = pts[MCP_MIDDLE]
    r_mcp  = pts[MCP_RING]
    p_mcp  = pts[MCP_PINKY]

    palm_y = m_mcp - wrist
    norm_y = np.linalg.norm(palm_y)
    if norm_y < 1e-9:
        return None, None, None
    palm_y /= norm_y

    palm_x  = p_mcp - i_mcp
    palm_x -= np.dot(palm_x, palm_y) * palm_y
    norm_x  = np.linalg.norm(palm_x)
    if norm_x < 1e-9:
        return None, None, None
    palm_x /= norm_x

    palm_z  = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z)

    R            = np.stack([palm_x, palm_y, palm_z], axis=1)
    palm_centre  = (i_mcp + m_mcp + r_mcp) / 3.0
    scale        = HAND_SCALE_M / norm_y

    return R, palm_centre, scale


def landmarks_to_targets(pts: np.ndarray):
    R, palm_centre, scale = build_palm_frame(pts)
    if R is None:
        return None

    targets = np.zeros((4, 3))
    for i, tip_idx in enumerate(TIP_INDICES):
        tip_rel_world = pts[tip_idx] - palm_centre
        tip_mp        = R.T @ tip_rel_world
        tip_mp        *= scale
        targets[i]    = AXIS_MAP @ tip_mp + PALM_OFFSET

    return targets


def draw_skeleton(frame, hand, w, h):
    for a, b in CONNECTIONS:
        cv2.line(
            frame,
            (int(hand[a].x * w), int(hand[a].y * h)),
            (int(hand[b].x * w), int(hand[b].y * h)),
            (0, 255, 0), 2,
        )
    for lm in hand:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)


# ------------------------------------------------------------------ #
# Recording phase                                                    #
# ------------------------------------------------------------------ #

def record_motion(name: str, duration_s: float = 10.0):
    """
    Record a hand motion from the webcam.

    Saves:
      Motions/{name}.avi          — raw video for preview
      Motions/{name}_landmarks.pkl — list of (21,3) landmark arrays per frame
    """
    video_path = os.path.join(MOTION_DIR, f"{name}.avi")
    lm_path    = os.path.join(MOTION_DIR, f"{name}_landmarks.pkl")

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    landmark_list = []
    start_time    = time.time()
    frame_idx     = 0

    print(f"\nRecording '{name}' for {duration_s:.1f} seconds...")
    print("Show your hand to the camera. Press Q to stop early.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            elapsed = time.time() - start_time
            if elapsed >= duration_s:
                break

            success, frame = cap.read()
            if not success:
                break

            frame   = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(
                mp_image, int((elapsed + start_time) * 1000)
            )

            pts = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_skeleton(frame, hand, w, h)
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)

            landmark_list.append(pts)

            # overlay status
            status = f"REC  {elapsed:5.1f}s / {duration_s:.1f}s  frame {frame_idx}"
            cv2.putText(frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            writer.write(frame)
            cv2.imshow("Recording", frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(lm_path, "wb") as f:
        pickle.dump(landmark_list, f)

    print(f"Saved video:    {video_path}")
    print(f"Saved landmarks: {lm_path}")
    print(f"Recorded {frame_idx} frames at ~{fps:.1f} fps.")


# ------------------------------------------------------------------ #
# Offline IK processing phase                                        #
# ------------------------------------------------------------------ #

def process_motion(name: str):
    """
    Load a recorded landmark sequence, solve IK for every frame, and save
    the resulting joint-angle trajectory.

    Saves:
      Motions/{name}_trajectory.npy  — (N, 16) joint angles in degrees
    """
    lm_path = os.path.join(MOTION_DIR, f"{name}_landmarks.pkl")
    out_path = os.path.join(MOTION_DIR, f"{name}_trajectory.npy")

    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"No recorded landmarks found for '{name}'")

    with open(lm_path, "rb") as f:
        landmark_list = pickle.load(f)

    kin = LeapKinematics()
    trajectory_deg = []
    prev_q_rad       = None
    smoothed_q_deg   = np.zeros(16)

    print(f"\nProcessing {len(landmark_list)} frames offline...")

    for i, pts in enumerate(landmark_list):
        if pts is None:
            # no hand detected: hold previous angle if available
            if len(trajectory_deg) > 0:
                trajectory_deg.append(trajectory_deg[-1].copy())
            else:
                trajectory_deg.append(np.zeros(16))
            continue

        targets = landmarks_to_targets(pts)
        if targets is None:
            if len(trajectory_deg) > 0:
                trajectory_deg.append(trajectory_deg[-1].copy())
            else:
                trajectory_deg.append(np.zeros(16))
            continue

        q_rad, infos = kin.ik(
            targets,
            q0=prev_q_rad,
            n_starts=IK_N_STARTS,
            tol=IK_TOL_M,
            max_iter=IK_MAX_ITER,
        )
        q_rad      = kin.clip_to_limits(q_rad)
        prev_q_rad = q_rad.copy()
        q_deg      = np.degrees(q_rad)

        smoothed_q_deg = ALPHA * q_deg + (1 - ALPHA) * smoothed_q_deg
        trajectory_deg.append(smoothed_q_deg.copy())

        if (i + 1) % 30 == 0:
            print(f"  processed {i + 1}/{len(landmark_list)} frames")

    trajectory_deg = np.array(trajectory_deg, dtype=np.float64)
    np.save(out_path, trajectory_deg)
    print(f"Saved trajectory: {out_path}")
    print(f"Shape: {trajectory_deg.shape}")


# ------------------------------------------------------------------ #
# Smooth replay phase                                                #
# ------------------------------------------------------------------ #

def replay_motion(name: str, speed: float = 1.0):
    """
    Load a saved trajectory and replay it smoothly on the robot.

    Parameters
    ----------
    name  : motion file base name
    speed : playback speed multiplier (1.0 = recorded speed)
    robot : if True, send commands to ExArm; if False, preview only
    """
    traj_path = os.path.join(MOTION_DIR, f"{name}_trajectory.npy")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"No trajectory found for '{name}'. Run process first.")

    trajectory_deg = np.load(traj_path)
    n_frames = len(trajectory_deg)

    # recorded at ~30 fps; replay at 60 Hz with linear interpolation
    replay_hz   = 60.0
    record_hz   = 30.0
    dt          = 1.0 / replay_hz
    total_time  = n_frames / record_hz
    n_replay    = int(total_time * replay_hz / speed)

    leap_hand = ExArm(
        mode="sim",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

    print(f"\nReplaying '{name}' ({n_frames} frames, {total_time:.1f}s) "
          f"at {speed:.1f}x speed...")

    start = time.time()
    for i in range(n_replay):
        t = i * dt * speed * record_hz  # virtual frame index in recorded space
        t0 = int(np.floor(t))
        t1 = min(t0 + 1, n_frames - 1)
        frac = t - t0

        q = (1 - frac) * trajectory_deg[t0] + frac * trajectory_deg[t1]
        q = np.round(q, 1)

        leap_hand.set_goal_positions_degree(q)
        leap_hand.set_torque_enabled(True)

        # simple progress print
        if i % int(replay_hz) == 0:
            print(f"  replay {i / replay_hz:.1f}s / {total_time / speed:.1f}s")

        # throttle to replay_hz
        elapsed = time.time() - start
        target  = i * dt
        sleep   = target - elapsed
        if sleep > 0:
            time.sleep(sleep)

    print("Replay finished.")


# ------------------------------------------------------------------ #
# Save / load / list helpers                                         #
# ------------------------------------------------------------------ #

def list_motions():
    files = sorted(os.listdir(MOTION_DIR))
    names = set()
    for f in files:
        base, _ = os.path.splitext(f)
        if base.endswith("_landmarks"):
            names.add(base.replace("_landmarks", ""))
        elif base.endswith("_trajectory"):
            names.add(base.replace("_trajectory", ""))
        else:
            names.add(base)
    return sorted(names)


def delete_motion(name: str):
    for suffix in [".avi", "_landmarks.pkl", "_trajectory.npy"]:
        path = os.path.join(MOTION_DIR, f"{name}{suffix}")
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}")


# ------------------------------------------------------------------ #
# CLI menu                                                           #
# ------------------------------------------------------------------ #

def main():
    kin = LeapKinematics()
    fk_ref = kin.fk(np.zeros(16))
    print("\nFK fingertip positions at zero pose (LEAP frame, metres):")
    print(f"  {'FINGER':<8}  {'x':>8} {'y':>8} {'z':>8}")
    for i, name in enumerate(FINGER_NAMES):
        x, y, z = fk_ref[i]
        print(f"  {name:<8}  {x:+.4f}  {y:+.4f}  {z:+.4f}")
    print("\nThese are your calibration targets.\n")

    while True:
        print("=" * 50)
        print("Vision Kinesthetic Teaching")
        print("=" * 50)
        print("1. Record a new motion")
        print("2. Process recorded motion (offline IK)")
        print("3. Replay motion on robot")
        print("4. List saved motions")
        print("5. Delete a motion")
        print("6. Record → process → replay (full pipeline)")
        print("0. Exit")
        choice = input("Choice: ").strip()

        if choice == "1":
            name = input("Motion name: ").strip()
            dur  = float(input("Duration (seconds, default 10): ") or "10")
            record_motion(name, dur)

        elif choice == "2":
            name = input("Motion name: ").strip()
            process_motion(name)

        elif choice == "3":
            name  = input("Motion name: ").strip()
            speed = float(input("Speed multiplier (default 1.0): ") or "1.0")
            replay_motion(name, speed=speed)

        elif choice == "4":
            motions = list_motions()
            print("\nSaved motions:")
            for m in motions:
                has_video = os.path.exists(os.path.join(MOTION_DIR, f"{m}.avi"))
                has_traj  = os.path.exists(os.path.join(MOTION_DIR, f"{m}_trajectory.npy"))
                status = []
                if has_video:  status.append("video")
                if has_traj:   status.append("trajectory")
                print(f"  {m:<20} ({', '.join(status)})")

        elif choice == "5":
            name = input("Motion name to delete: ").strip()
            delete_motion(name)

        elif choice == "6":
            name = input("Motion name: ").strip()
            dur  = float(input("Duration (seconds, default 10): ") or "10")
            record_motion(name, dur)
            process_motion(name)
            speed = float(input("Replay speed multiplier (default 1.0): ") or "1.0")
            replay_motion(name, speed=speed)

        elif choice == "0":
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()