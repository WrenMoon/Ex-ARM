"""
Vision_Retargeting.py — IK-enabled fingertip teleoperation with calibrated offsets.

Pipeline:
  webcam → MediaPipe landmarks → palm-frame fingertip positions
         → axis remap → scale → offset → IK (LeapKinematics) → joint angles

Unlike Vision_Teleop.py (which maps landmark angles directly), this pipeline
reconstructs 3-D fingertip positions in the robot's palm frame, then uses
inverse kinematics to compute the joint angles that reproduce those positions.

Key calibration parameters (AXIS_MAP, PALM_OFFSET, HAND_SCALE_M) must be
tuned once per camera setup. FK reference positions are printed at startup
to assist with calibration.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from utils.Constants import Connection
from utils.ExARM import ExArm
from utils.LeapKinematics import LeapKinematics

# Path to the MediaPipe hand landmarker model file.
# Downloaded automatically on first run if not present.
MODEL_PATH = "Data/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand tracking model...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")


# ------------------------------------------------------------------ #
# Calibration parameters                                               #
# ------------------------------------------------------------------ #

# Distance from wrist to middle-MCP on the robot hand (metres).
# Used to scale MediaPipe's normalised landmark coordinates to real-world metres.
HAND_SCALE_M = 0.233

# Axis remapping matrix (3×3).
# Maps a fingertip vector expressed in the MediaPipe palm frame to the
# LEAP Hand palm frame. Tune by comparing print_debug() output with FK reference.
AXIS_MAP = np.array([
    [0, 1, 0],   # LEAP +x  ← MediaPipe axes [x, y, z]
    [1, 0, 0],   # LEAP +y
    [0, 0, 1],   # LEAP +z
], dtype=float)

# Origin offset (metres) applied after axis remapping.
# Shifts all fingertip targets to align with the LEAP IK coordinate origin.
PALM_OFFSET = np.array([
    -0.085,   # LEAP x
    -0.05,    # LEAP y
     0.04,    # LEAP z
], dtype=float)

# Exponential moving average coefficient for joint angle smoothing.
# Higher α → more responsive but jitterier; lower α → smoother but more lag.
ALPHA = 0.25

# IK solver settings (per-finger, per-frame)
IK_N_STARTS = 1    # number of random restarts (1 = warm-start only, fastest)
IK_TOL_M    = 5e-4 # convergence tolerance in metres
IK_MAX_ITER = 30   # max L-BFGS-B iterations per finger


# ------------------------------------------------------------------ #
# MediaPipe landmark index constants                                   #
# ------------------------------------------------------------------ #

WRIST      = 0
MCP_INDEX  = 5
MCP_MIDDLE = 9
MCP_RING   = 13
MCP_PINKY  = 17

# Fingertip landmark indices in MediaPipe ordering.
# Output order matches LeapKinematics: [index, middle, ring, thumb]
TIP_INDICES  = [8, 12, 16, 4]
FINGER_NAMES = ["index", "middle", "ring", "thumb"]

# Skeleton edges for camera overlay rendering
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


# ------------------------------------------------------------------ #
# Palm frame construction and coordinate transform                     #
# ------------------------------------------------------------------ #

def build_palm_frame(pts: np.ndarray):
    """
    Build an orthonormal right-handed coordinate frame from MediaPipe landmarks.

    The frame is centred at the average MCP position and oriented so that:
      - palm_y points from wrist toward the middle finger MCP (proximal)
      - palm_x points laterally from index to pinky MCP
      - palm_z is the dorsal normal (cross product of x and y)

    Parameters
    ----------
    pts : (21, 3) MediaPipe landmark array in normalised camera coordinates

    Returns
    -------
    R            : (3, 3) rotation matrix [palm_x | palm_y | palm_z] as columns
    palm_centre  : (3,)   average of index/middle/ring MCPs (frame origin)
    scale        : float  converts landmark units to metres using HAND_SCALE_M
    Returns (None, None, None) if the frame cannot be built (degenerate input).
    """
    wrist  = pts[WRIST]
    i_mcp  = pts[MCP_INDEX]
    m_mcp  = pts[MCP_MIDDLE]
    r_mcp  = pts[MCP_RING]
    p_mcp  = pts[MCP_PINKY]

    # palm_y: wrist → middle MCP (proximal direction)
    palm_y = m_mcp - wrist
    norm_y = np.linalg.norm(palm_y)
    if norm_y < 1e-9:
        return None, None, None
    palm_y /= norm_y

    # palm_x: index MCP → pinky MCP, orthogonalised w.r.t. palm_y
    palm_x  = p_mcp - i_mcp
    palm_x -= np.dot(palm_x, palm_y) * palm_y
    norm_x  = np.linalg.norm(palm_x)
    if norm_x < 1e-9:
        return None, None, None
    palm_x /= norm_x

    # palm_z: dorsal normal via right-hand rule
    palm_z  = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z)

    R            = np.stack([palm_x, palm_y, palm_z], axis=1)
    palm_centre  = (i_mcp + m_mcp + r_mcp) / 3.0
    scale        = HAND_SCALE_M / norm_y  # landmark units → metres

    return R, palm_centre, scale


def landmarks_to_targets(pts: np.ndarray):
    """
    Transform (21, 3) MediaPipe landmarks to (4, 3) fingertip target positions
    in the LEAP Hand palm frame (metres).

    Steps per fingertip:
      1. Compute vector from palm centre to fingertip in camera space.
      2. Rotate into the MediaPipe palm frame (removes hand orientation).
      3. Scale from landmark units to metres.
      4. Apply AXIS_MAP to re-express in the LEAP palm frame convention.
      5. Add PALM_OFFSET to align with the IK coordinate origin.

    Returns None if the palm frame cannot be constructed.
    """
    R, palm_centre, scale = build_palm_frame(pts)
    if R is None:
        return None

    targets = np.zeros((4, 3))
    for i, tip_idx in enumerate(TIP_INDICES):
        tip_rel_world = pts[tip_idx] - palm_centre   # camera-space vector
        tip_mp        = R.T @ tip_rel_world           # rotate into palm frame
        tip_mp        *= scale                        # scale to metres
        targets[i]    = AXIS_MAP @ tip_mp + PALM_OFFSET

    return targets


# ------------------------------------------------------------------ #
# Display helpers                                                      #
# ------------------------------------------------------------------ #

def draw_skeleton(frame, hand, w, h):
    """
    Render the MediaPipe hand skeleton as green lines and red joint dots
    on the given BGR frame.
    """
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
    Print a side-by-side comparison of MediaPipe fingertip targets and FK
    reference positions to aid AXIS_MAP / PALM_OFFSET calibration.

    With a flat open hand, both columns should match after correct tuning.
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
            f"  {name:<8}  {mx:+.4f} {my:+.4f} {mz:+.4f}    "
            f"{fx:+.4f} {fy:+.4f} {fz:+.4f}"
        )
    print("=" * 70)
    print("\n  Tune AXIS_MAP and PALM_OFFSET until columns match.")
    print("  PALM_OFFSET (current):", np.round(PALM_OFFSET, 4))
    print("  AXIS_MAP (current):")
    for row in AXIS_MAP:
        print("   ", row)


def print_status(targets, infos, q_deg):
    """
    Print per-finger IK status and current joint angles to stdout.

    Shows target positions, IK success/failure with residual error,
    and the resulting joint angles grouped by finger.
    """
    os.system("cls" if os.name == "nt" else "clear")
    print("Fingertip targets  (LEAP frame, metres):")
    for i, name in enumerate(FINGER_NAMES):
        x, y, z = targets[i]
        ok = "✓" if infos[i]["success"] else f"✗  residual={infos[i]['error_m']*1000:.1f} mm"
        print(f"  {name:6s}: [{x:+.3f}, {y:+.3f}, {z:+.3f}]   IK {ok}")
    print("\nJoint angles (deg) — [abduct, flex, pip, dip] per finger:")
    for i, lbl in enumerate(["Index ", "Middle", "Ring  ", "Thumb "]):
        print(f"  {lbl}: {np.round(q_deg[i*4:(i+1)*4], 1)}")


# ------------------------------------------------------------------ #
# Main loop                                                            #
# ------------------------------------------------------------------ #

def main():
    """
    Initialise kinematics and the robot, then stream IK-based joint goals
    to the LEAP Hand at the webcam frame rate (~30 fps).

    Warm-starting the IK solver from the previous frame's solution is
    critical for real-time performance: with IK_N_STARTS=1 and a good
    warm-start, each finger typically converges in <5 ms.
    """
    kin = LeapKinematics()

    # Print FK reference at zero pose so the user can verify calibration
    fk_ref = kin.fk(np.zeros(16))
    print("\nFK fingertip positions at zero pose (LEAP frame, metres):")
    print(f"  {'FINGER':<8}  {'x':>8} {'y':>8} {'z':>8}")
    for i, name in enumerate(FINGER_NAMES):
        x, y, z = fk_ref[i]
        print(f"  {name:<8}  {x:+.4f}  {y:+.4f}  {z:+.4f}")
    print("\nThese are your calibration targets. Press Q to quit.\n")

    leap_hand = ExArm(
        mode="both",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

    # MediaPipe setup
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

    # IK warm-start: initialised to zero; updated each frame for continuity
    smoothed_q_deg = np.zeros(16)
    prev_q_rad     = None

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        print("Press Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame    = cv2.flip(frame, 1)   # mirror for natural interaction
            h, w, _  = frame.shape

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_skeleton(frame, hand, w, h)

                # Build (21, 3) landmark array
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)

                # Map landmarks to (4, 3) fingertip targets in LEAP frame (metres)
                mp_targets = landmarks_to_targets(pts)
                if mp_targets is None:
                    cv2.imshow("MediaPipe Hand Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Visualise IK targets as markers in the MuJoCo sim viewer
                leap_hand.set_custom_markers(mp_targets)

                # Solve full-hand IK, warm-started from the previous frame
                q_rad, infos = kin.ik(
                    mp_targets,
                    q0=prev_q_rad,
                    n_starts=IK_N_STARTS,
                    tol=IK_TOL_M,
                    max_iter=IK_MAX_ITER,
                )

                # Enforce joint limits and cache solution for next-frame warm-start
                q_rad      = kin.clip_to_limits(q_rad)
                prev_q_rad = q_rad.copy()
                q_deg      = np.degrees(q_rad)

                # Smooth joint angles to reduce IK-solution discontinuities
                smoothed_q_deg = ALPHA * q_deg + (1 - ALPHA) * smoothed_q_deg

                # Optional: uncomment to print per-finger debug info each frame
                # print_status(mp_targets, infos, smoothed_q_deg)

                # Send smoothed angles to the robot
                leap_hand.set_goal_positions_degree(np.round(smoothed_q_deg, 1))
                leap_hand.set_torque_enabled(True)

            cv2.imshow("MediaPipe Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()