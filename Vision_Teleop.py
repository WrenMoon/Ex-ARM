"""
Vision_Teleop.py — Direct angle-mapping hand teleoperation via webcam.

Pipeline:
  webcam → MediaPipe hand landmarker → joint flexion angles
         → clip & smooth → LEAP Hand goal positions

This is the simpler of the two vision pipelines. Joint angles are computed
directly from landmark geometry using 3-point flexion calculations, without
any IK. Best used for quick demonstrations where exact fingertip tracking
is not required.

Run directly:  python Vision_Teleop.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
from utils.Constants import Connection
from utils.ExARM import ExArm

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
# Angle computation utilities                                          #
# ------------------------------------------------------------------ #

def get_angle(p1, p2, p3):
    """
    Compute the interior angle at vertex p2 formed by the rays p2→p1 and p2→p3.

    Parameters
    ----------
    p1, p2, p3 : (3,) arrays — 3-D landmark coordinates

    Returns
    -------
    Angle in degrees in [0, 180]. Returns 180° if either ray is degenerate.
    """
    v1 = p1 - p2
    v2 = p3 - p2

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    v1 /= n1
    v2 /= n2

    return np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))


def flexion(a, b, c):
    """
    Compute joint flexion angle from three landmarks.

    Convention:
      - Straight finger → 0°
      - Bent finger     → positive value

    Computed as  max(0, 180° − interior_angle(a, b, c)).
    """
    return max(0.0, 180.0 - get_angle(a, b, c))


def signed_angle(v1, v2, normal):
    """
    Compute the signed angle from v1 to v2, measured around the given normal.

    Uses atan2 for a full ±180° range.

    Parameters
    ----------
    v1, v2 : (3,) direction vectors (need not be normalised)
    normal  : (3,) axis vector defining the sign convention

    Returns
    -------
    Signed angle in degrees.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    return np.degrees(
        np.arctan2(np.dot(np.cross(v1, v2), normal), np.dot(v1, v2))
    )


# ------------------------------------------------------------------ #
# MediaPipe skeleton connections for overlay rendering                 #
# ------------------------------------------------------------------ #

# Each tuple is a (start_landmark, end_landmark) pair drawn as a line.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)             # Palm transverse
]


# ------------------------------------------------------------------ #
# Main loop                                                            #
# ------------------------------------------------------------------ #

def main():
    """
    Open the webcam, detect hand landmarks with MediaPipe, map them to
    LEAP Hand joint angles, and stream goals to the robot at ~30 fps.

    Exponential moving average (α = 0.2) smooths the joint angles to
    reduce jitter from landmark noise.
    """
    leap_hand = ExArm(
        mode="both",
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path="Data/mujoco_robot.urdf"
    )

    # Configure MediaPipe HandLandmarker in VIDEO mode for frame-by-frame inference
    BaseOptions          = mp.tasks.BaseOptions
    HandLandmarker       = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode    = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        cap           = cv2.VideoCapture(0)
        smoothed_pose = np.zeros(16)

        print("Press Q to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Mirror the frame so movements feel natural (like a mirror)
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Convert BGR → RGB for MediaPipe
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Run landmark detection on this video frame
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]

                # Draw skeleton overlay for visual feedback
                for a, b in CONNECTIONS:
                    cv2.line(
                        frame,
                        (int(hand[a].x * w), int(hand[a].y * h)),
                        (int(hand[b].x * w), int(hand[b].y * h)),
                        (0, 255, 0), 2
                    )
                for lm in hand:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

                # Convert normalised landmark list to a (21, 3) NumPy array
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float64)

                # ---- Per-finger flexion ----
                # Each finger: [wrist, MCP, PIP] or [MCP, PIP, DIP] triplets
                idx_mcp  = flexion(pts[0], pts[5],  pts[6])
                idx_pip  = flexion(pts[5], pts[6],  pts[7])
                idx_dip  = flexion(pts[6], pts[7],  pts[8])

                mid_mcp  = flexion(pts[0], pts[9],  pts[10])
                mid_pip  = flexion(pts[9], pts[10], pts[11])
                mid_dip  = flexion(pts[10], pts[11], pts[12])

                ring_mcp = flexion(pts[0], pts[13], pts[14])
                ring_pip = flexion(pts[13], pts[14], pts[15])
                ring_dip = flexion(pts[14], pts[15], pts[16])

                # ---- Thumb angle computation ----
                # Estimate thumb rotation from the angle of the proximal bone
                # relative to the camera plane, then offset to match LEAP convention
                thumb_base   = pts[2] - pts[1]
                thumb_rotate = (
                    np.degrees(np.arctan2(thumb_base[0], -thumb_base[1])) - 60
                )
                thumb_mcp = flexion(pts[1], pts[2], pts[3])
                thumb_ip  = flexion(pts[2], pts[3], pts[4])

                # ---- Assemble 16-joint pose vector ----
                # Logical ordering: index(4) + middle(4) + ring(4) + thumb(4)
                allegro_pose = np.array([
                    0,       idx_mcp, idx_mcp, idx_mcp,   # Index  (abduct, flex, pip, dip)
                    0,       mid_mcp, mid_pip, mid_dip,   # Middle
                    0,       ring_mcp, ring_pip, ring_dip, # Ring
                    -90,     0,       thumb_mcp, thumb_ip  # Thumb
                ], dtype=np.float64)

                # Hard clip to safe operating range before sending to hardware
                allegro_pose = np.clip(allegro_pose, -90, 90)

                os.system("cls" if os.name == "nt" else "clear")

                # Exponential moving average to smooth out landmark jitter
                alpha         = 0.2
                smoothed_pose = alpha * allegro_pose + (1 - alpha) * smoothed_pose

                print("\nLEAP / ALLEGRO FORMAT\n")
                print(np.round(smoothed_pose, 1))

                leap_hand.set_goal_positions_degree(np.round(smoothed_pose, 1))
                leap_hand.set_torque_enabled(True)

            cv2.imshow("MediaPipe Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()