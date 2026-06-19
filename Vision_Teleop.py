import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
from utils.ExARM import ExArm

MODEL_PATH = "Data/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe hand tracking model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0

    v1 /= n1
    v2 /= n2

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)

    return np.degrees(np.arccos(dot))


def flexion(a, b, c):
    """
    Returns:
        straight finger -> 0
        bent finger -> positive
    """
    return max(0.0, 180.0 - get_angle(a, b, c))


def signed_angle(v1, v2, normal):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    return np.degrees(
        np.arctan2(
            np.dot(np.cross(v1, v2), normal),
            np.dot(v1, v2)
        )
    )


# ==========================================
# HAND SKELETON CONNECTIONS
# ==========================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ==========================================
# MAIN
# ==========================================

def main():
    leap_hand = ExArm(
        mode="sim",
        ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        port="COM5",
        baudrate=4000000,
        offsets=[0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        model_path="Data/mujoco_robot.urdf"
    )

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
        min_tracking_confidence=0.5
    )

    with HandLandmarker.create_from_options(options) as landmarker:

        cap = cv2.VideoCapture(0)

        print("Press Q to quit")
        smoothed_pose = np.zeros(16)


        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                # ==========================
                # DRAW SKELETON
                # ==========================

                for a, b in CONNECTIONS:

                    x1 = int(hand[a].x * w)
                    y1 = int(hand[a].y * h)

                    x2 = int(hand[b].x * w)
                    y2 = int(hand[b].y * h)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                for lm in hand:

                    cx = int(lm.x * w)
                    cy = int(lm.y * h)

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                # ==========================
                # LANDMARK ARRAY
                # ==========================

                pts = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in hand
                ], dtype=np.float64)

                # ==========================
                # PALM FRAME
                # ==========================

                wrist = pts[0]

                index_mcp = pts[5]
                middle_mcp = pts[9]
                ring_mcp = pts[13]
                pinky_mcp = pts[17]

                palm_x = pinky_mcp - index_mcp
                palm_x /= np.linalg.norm(palm_x)

                palm_y = middle_mcp - wrist
                palm_y /= np.linalg.norm(palm_y)

                palm_z = np.cross(palm_x, palm_y)

                if np.linalg.norm(palm_z) > 1e-6:
                    palm_z /= np.linalg.norm(palm_z)

                # Index
                idx_mcp = flexion(pts[0], pts[5], pts[6])
                idx_pip = flexion(pts[5], pts[6], pts[7])
                idx_dip = flexion(pts[6], pts[7], pts[8])

                # Middle
                mid_mcp = flexion(pts[0], pts[9], pts[10])
                mid_pip = flexion(pts[9], pts[10], pts[11])
                mid_dip = flexion(pts[10], pts[11], pts[12])

                # Ring
                ring_mcp = flexion(pts[0], pts[13], pts[14])
                ring_pip = flexion(pts[13], pts[14], pts[15])
                ring_dip = flexion(pts[14], pts[15], pts[16])

                # Thumb

                thumb_base = pts[2] - pts[1]

                thumb_rotate = np.degrees(
                    np.arctan2(
                        thumb_base[0],
                        -thumb_base[1]
                    ) 
                ) - 60

                thumb_mcp = flexion(pts[1], pts[2], pts[3])
                thumb_ip  = flexion(pts[2], pts[3], pts[4])

                allegro_pose = np.array([

                    0,
                    idx_mcp,
                    idx_mcp,
                    idx_mcp,

                    0,
                    mid_mcp,
                    mid_pip,
                    mid_dip,

                    0,
                    ring_mcp,
                    ring_pip,
                    ring_dip,

                    -90,
                    0,
                    thumb_mcp,
                    thumb_ip

                ], dtype=np.float64)

                allegro_pose = np.clip(
                    allegro_pose,
                    -90,
                    90
                )

                os.system(
                    "cls" if os.name == "nt" else "clear"
                )

                alpha = 0.2

                smoothed_pose = (
                    alpha * allegro_pose +
                    (1 - alpha) * smoothed_pose
                )
                
                print("\nLEAP / ALLEGRO FORMAT\n")
                print(np.round(smoothed_pose, 1))
                leap_hand.set_goal_positions_degree(np.round(smoothed_pose, 1))
                leap_hand.set_torque_enabled(True)


            cv2.imshow(
                "MediaPipe Hand Tracking",
                frame
            )

            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()