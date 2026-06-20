"""
Vision_Teleop.py  —  IK-enabled fingertip teleoperation with calibrated offsets
---------------------------------------------------------------------------
Pipeline:
  webcam → MediaPipe landmarks → palm-frame fingertip positions
         → axis remap → scale → offset → IK (LeapKinematics) → joint angles

This file uses the AXIS_MAP, PALM_OFFSET and HAND_SCALE_M you tuned during
calibration. It prints FK reference positions at startup to help verify calibration,
then runs IK and sends joint goals to ExArm (sim or real).
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from utils.ExARM import ExArm
from utils.LeapKinematics import LeapKinematics

MODEL_PATH = "Data/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand tracking model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")


# ============================================================
# CONFIGURATION  — tuned values (from your calibration)
# ============================================================

# Distance from wrist to middle-MCP on the *robot* hand (metres).
# You tuned this during calibration.
HAND_SCALE_M = 0.233

# ---- Axis remapping ----
# This AXIS_MAP is the mapping you tuned. If you need to adjust later,
# modify rows below (each row selects a combination of MP axes for LEAP axes).
AXIS_MAP = np.array([
    [0, 1, 0],   # LEAP +x  ← combination of MP axes (row picks MP x,y,z)
    [1, 0, 0],   # LEAP +y
    [0, 0, 1],   # LEAP +z
], dtype=float)

# ---- Origin offset ----
# Tuned PALM_OFFSET (in metres) to shift fingertip targets into LEAP IK origin frame.
PALM_OFFSET = np.array([
    -0.085,   # LEAP x
    -0.05,    # LEAP y
    0.04,     # LEAP z
], dtype=float)

# Output smoothing (exponential moving average) for joint degrees.
ALPHA = 0.25

# IK settings
IK_N_STARTS = 1
IK_TOL_M = 5e-4
IK_MAX_ITER = 30


# ============================================================
# CONSTANTS
# ============================================================

# MediaPipe landmark indices
WRIST = 0
MCP_INDEX = 5
MCP_MIDDLE = 9
MCP_RING = 13
MCP_PINKY = 17

# Fingertip indices — order matches LeapKinematics (0=index,1=middle,2=ring,3=thumb)
TIP_INDICES = [8, 12, 16, 4]
FINGER_NAMES = ["index", "middle", "ring", "thumb"]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ============================================================
# CORE: landmarks → robot fingertip targets
# ============================================================

def build_palm_frame(pts: np.ndarray):
    """
    Build an orthonormal right-handed frame from MediaPipe landmarks.

    Returns
    -------
    R          : (3, 3) rotation matrix whose columns are [palm_x, palm_y, palm_z]
    palm_centre: (3,)   average of index/middle/ring MCPs (used as frame origin)
    scale      : float  converts landmark units → metres
    """
    wrist = pts[WRIST]
    i_mcp = pts[MCP_INDEX]
    m_mcp = pts[MCP_MIDDLE]
    r_mcp = pts[MCP_RING]
    p_mcp = pts[MCP_PINKY]

    # palm_y: proximal direction (wrist → middle MCP)
    palm_y = m_mcp - wrist
    norm_y = np.linalg.norm(palm_y)
    if norm_y < 1e-9:
        return None, None, None
    palm_y /= norm_y

    # palm_x: lateral direction (index MCP → pinky MCP), orthogonalised
    palm_x = p_mcp - i_mcp
    palm_x -= np.dot(palm_x, palm_y) * palm_y
    norm_x = np.linalg.norm(palm_x)
    if norm_x < 1e-9:
        return None, None, None
    palm_x /= norm_x

    # palm_z: dorsal (out of palm back)
    palm_z = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z)

    R = np.stack([palm_x, palm_y, palm_z], axis=1)  # columns are frame axes

    palm_centre = (i_mcp + m_mcp + r_mcp) / 3.0

    # Scale: map the human wrist→middle-MCP distance to HAND_SCALE_M
    scale = HAND_SCALE_M / norm_y

    return R, palm_centre, scale


def landmarks_to_targets(pts: np.ndarray):
    """
    Map (21, 3) MediaPipe landmarks to (4, 3) fingertip targets
    expressed in the LEAP hand frame (metres).
    """
    R, palm_centre, scale = build_palm_frame(pts)
    if R is None:
        return None

    targets = np.zeros((4, 3))
    for i, tip_idx in enumerate(TIP_INDICES):
        # Vector from palm centre to fingertip in camera/world space
        tip_rel_world = pts[tip_idx] - palm_centre
        # Rotate into MediaPipe palm frame
        tip_mp = R.T @ tip_rel_world
        # Scale to metres
        tip_mp *= scale
        # Remap to LEAP frame axes
        tip_leap = AXIS_MAP @ tip_mp
        # Apply origin offset
        targets[i] = tip_leap + PALM_OFFSET

    return targets


# ============================================================
# DISPLAY HELPERS
# ============================================================

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


def print_debug(mp_targets, fk_targets):
    """
    Print MediaPipe fingertip targets alongside FK reference targets
    to help verify calibration (hand open).
    """
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("  FINGERTIP POSITIONS  (LEAP frame, metres)")
    print("=" * 70)
    print(f"  {'FINGER':<8}  {'MEDIAPIPE (x,y,z)':^28}  {'FK @ zero (x,y,z)':^28}")
    print("-" * 70)
    for i, name in enumerate(FINGER_NAMES):
        mx, my, mz = mp_targets[i]
        fx, fy, fz = fk_targets[i]
        print(
            f"  {name:<8} "
            f" {mx:+.4f} {my:+.4f} {mz:+.4f}    "
            f"{fx:+.4f} {fy:+.4f} {fz:+.4f}"
        )
    print("=" * 70)
    print("\n  Tune AXIS_MAP and PALM_OFFSET until columns match.")
    print("  PALM_OFFSET (current):", np.round(PALM_OFFSET, 4))
    print("  AXIS_MAP (current):")
    for row in AXIS_MAP:
        print("   ", row)


def print_status(targets, infos, q_deg):
    os.system("cls" if os.name == "nt" else "clear")
    print("Fingertip targets  (LEAP frame, metres):")
    for i, name in enumerate(FINGER_NAMES):
        x, y, z = targets[i]
        ok = "✓" if infos[i]["success"] else f"✗  residual={infos[i]['error_m']*1000:.1f} mm"
        print(f"  {name:6s}: [{x:+.3f}, {y:+.3f}, {z:+.3f}]   IK {ok}")
    print("\nJoint angles (deg) — [abduct, flex, pip, dip] per finger:")
    labels = ["Index ", "Middle", "Ring  ", "Thumb "]
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: {np.round(q_deg[i*4:(i+1)*4], 1)}")


# ============================================================
# MAIN
# ============================================================

def main():
    kin = LeapKinematics()

    # Print FK reference at zero pose
    fk_ref = kin.fk(np.zeros(16))  # (4,3) in LEAP frame, metres
    print("\nFK fingertip positions at zero pose (LEAP frame, metres):")
    print(f"  {'FINGER':<8}  {'x':>8} {'y':>8} {'z':>8}")
    for i, name in enumerate(FINGER_NAMES):
        x, y, z = fk_ref[i]
        print(f"  {name:<8}  {x:+.4f}  {y:+.4f}  {z:+.4f}")
    print("\nThese are your calibration targets. Press Q to quit.\n")

    leap_hand = ExArm(
        mode="sim",                  # change to "real" for hardware
        ids=list(range(16)),
        port="COM5",
        baudrate=4000000,
        offsets=[0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        model_path="Data/mujoco_robot.urdf",
    )

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # IK warm-start and smoothing initialization
    smoothed_q_deg = np.zeros(16)
    prev_q_rad = None

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        print("Press Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_skeleton(frame, hand, w, h)

                # (21, 3) landmark array
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)

                # Compute MediaPipe fingertip targets in LEAP frame
                mp_targets = landmarks_to_targets(pts)
                if mp_targets is None:
                    cv2.imshow("MediaPipe Hand Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # OPTIONAL: quick debug compare (hand open)
                # print_debug(mp_targets, fk_ref)
                leap_hand.set_custom_markers(mp_targets)

                # Solve IK — warm-started from previous frame's solution
                q_rad, infos = kin.ik(
                    mp_targets,
                    q0=prev_q_rad,
                    n_starts=IK_N_STARTS,
                    tol=IK_TOL_M,
                    max_iter=IK_MAX_ITER,
                )

                # Clip to joint limits and convert to degrees
                q_rad = kin.clip_to_limits(q_rad)
                prev_q_rad = q_rad.copy()
                q_deg = np.degrees(q_rad)

                # Exponential smoothing on joint angles
                smoothed_q_deg = ALPHA * q_deg + (1 - ALPHA) * smoothed_q_deg

                # Print status and send to robot
                # print_status(mp_targets, infos, smoothed_q_deg)
                leap_hand.set_goal_positions_degree(np.round(smoothed_q_deg, 1))
                leap_hand.set_torque_enabled(True)

            cv2.imshow("MediaPipe Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()