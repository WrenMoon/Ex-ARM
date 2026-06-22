# Ex - ARM: LEAP Hand Python Control Suite

A full-featured Python SDK for the [LEAP Hand](https://leaphand.com/) open-source dexterous robotic hand.  
This suite provides hardware abstraction, MuJoCo simulation, analytical kinematics, vision-based teleoperation, and a GUI pose sequencer — all from a clean, unified interface.

---

## ✨ Features

| Module | Description |
|---|---|
| `LeapHand` | Low-level Dynamixel SDK driver — sync read/write, torque control, logical↔physical joint remapping |
| `SimHand` | MuJoCo viewer backend with custom marker rendering for visualising fingertip targets in real-time |
| `ExArm` | Unified abstraction layer — transparently controls real hardware, simulation, or **both simultaneously** |
| `LeapKinematics` | Full analytical FK + numerical IK (L-BFGS-B, multi-start) derived directly from the URDF joint origins |
| `Vision_Teleop` | Webcam → MediaPipe → angle-based teleoperation (direct flexion mapping) |
| `Vision_Retargeting` | Webcam → MediaPipe → **IK-based** fingertip retargeting with calibrated axis remapping |
| `Ex-GUI` | Tkinter GUI pose sequencer — build, edit, play, and export pose sequences interactively |
| `bottle_orient` | Example scripted manipulation sequence (bottle orientation task) |

---

## 📁 Project Structure

```
├── utils/
│   ├── LeapHand.py          # Dynamixel hardware driver
│   ├── SimHand.py           # MuJoCo simulation backend
│   ├── ExARM.py             # Unified real/sim abstraction (ExArm)
│   └── LeapKinematics.py    # Forward & Inverse Kinematics engine
│
├── Data/
│   ├── mujoco_robot.urdf    # LEAP Hand URDF (MuJoCo-compatible)
│   └── hand_landmarker.task # MediaPipe hand tracking model (auto-downloaded)
│
├── Vision_Teleop.py         # Direct angle-mapped vision teleoperation
├── Vision_Retargeting.py    # IK-based fingertip retargeting via webcam
├── Ex-GUI.py                # Interactive pose sequencer GUI
├── bottle_orient.py         # Example: scripted bottle orientation task
├── test.py                  # Basic hardware connection test
├── test2.py                 # Keyboard-driven IK simulation test
├── test4.py                 # URDF path fixer utility
├── pose_session.json        # Saved pose session (GUI output)
└── pose_editor_autosave.json
```

---

## 🔧 Requirements

python 3.11 or 3.10

```bash
pip install dynamixel-sdk numpy scipy mujoco opencv-python mediapipe pygame
```


---

## 🚀 Quick Start

### 1. Hardware only

```python
from utils.LeapHand import LeapHand
import numpy as np

hand = LeapHand(
    ids=list(range(16)),
    port="COM4",          # or "/dev/ttyUSB0" on Linux
    baudrate=4000000,
    offsets=[0]*16
)

hand.set_torque_enabled(True)
hand.set_goal_positions_degree(np.zeros(16))   # all joints to 0°

positions, velocities, currents = hand.get_state()
hand.close_port()
```

### 2. Simulation only

```python
from utils.SimHand import SimHand
import numpy as np

sim = SimHand(model_path="Data/mujoco_robot.urdf")
sim.set_goal_positions_degree(np.zeros(16))
sim.close()
```

### 3. Real + Sim simultaneously (recommended)

```python
from utils.ExARM import ExArm
import numpy as np

hand = ExArm(
    mode="both",                              # "real" | "sim" | "both"
    ids=list(range(16)),
    port="COM4",
    baudrate=4000000,
    offsets=[0]*16,
    model_path="Data/mujoco_robot.urdf"
)

hand.set_torque_enabled(True)
hand.set_goal_positions_degree(np.zeros(16))

state = hand.get_state()   # returns {"real": ..., "sim": ...} in "both" mode
hand.close()
```

---

## 🦾 Kinematics

`LeapKinematics` implements closed-form forward kinematics and numerical inverse kinematics (L-BFGS-B with multi-start) derived from the LEAP Hand URDF joint origins.

**Joint ordering (logical, 16 elements):**
```
[ 0.. 3]  Index  : abduct, flex, PIP, DIP
[ 4.. 7]  Middle : abduct, flex, PIP, DIP
[ 8..11]  Ring   : abduct, flex, PIP, DIP
[12..15]  Thumb  : base_rot, MCP, PIP, DIP
```

### Forward Kinematics

```python
from utils.LeapKinematics import LeapKinematics
import numpy as np

kin = LeapKinematics()

q = np.zeros(16)                    # all joints at zero (radians)
tips = kin.fk(q)                    # → (4, 3) fingertip positions in palm frame (metres)
kin.print_fk(q)                     # pretty-print all fingertip positions

# Single finger
tip, transforms = kin.fk_finger(0, q[:4])   # finger 0 = Index

# Degree variants
tips = kin.fk_degree(np.zeros(16))
```

### Inverse Kinematics

```python
# Single finger IK with multi-start for robustness
q_sol, info = kin.ik_finger_multistart(
    finger_idx=0,
    target_pos=np.array([-0.05, 0.12, 0.02]),  # metres, palm frame
    n_starts=8,
    tol=1e-4
)
print(f"IK success: {info['success']}, error: {info['error_m']*1000:.2f} mm")

# Full-hand IK (all 4 fingers)
targets = kin.fk(np.zeros(16))      # use FK positions as targets
q_sol, infos = kin.ik(targets, n_starts=4)

# Degree variants
q_deg, infos = kin.ik_degree(targets)
```

### Utilities

```python
kin.clip_to_limits(q)          # clamp 16-joint vector to URDF limits
kin.is_within_limits(q)        # True/False bounds check
J = kin.jacobian_finger(0, q[:4])  # numerical 3×4 Jacobian
```

---

## 👁️ Vision Teleoperation

Two webcam-based control pipelines are provided:

### Direct Angle Mapping (`Vision_Teleop.py`)
MediaPipe hand landmarks → joint flexion angles → LEAP joint goals.  
Best for low-latency, low-compute use.

```bash
python Vision_Teleop.py
```

### IK-Based Retargeting (`Vision_Retargeting.py`)
MediaPipe landmarks → calibrated palm frame → fingertip positions → IK → joint goals.  
Produces more anatomically accurate retargeting across different hand sizes.

```bash
python Vision_Retargeting.py
```

**Calibration parameters** (edit at the top of `Vision_Retargeting.py`):

| Parameter | Description |
|---|---|
| `HAND_SCALE_M` | Wrist-to-middle-MCP distance on the robot hand (metres) |
| `AXIS_MAP` | 3×3 matrix remapping MediaPipe axes to LEAP frame axes |
| `PALM_OFFSET` | Origin offset (metres) to align fingertip targets to IK frame |
| `ALPHA` | Exponential smoothing factor for joint angles (0–1) |

> The MediaPipe hand tracking model (`hand_landmarker.task`) is **automatically downloaded** on first run if not present.

---

## 🖥️ Pose Sequencer GUI (`Ex-GUI.py`)

An interactive Tkinter application for building and playing back hand pose sequences.

```bash
python Ex-GUI.py
```

**Features:**
- **Per-joint sliders** with ±1° / ±5° increment buttons and direct text entry
- **Finger utilities** — copy joint values between fingers (e.g. Index → Ring)
- **Zero utilities** — zero individual fingers or all joints at once
- **Pose sequencer** — save, reorder (move up/down), duplicate, delete, and insert poses
- **Playback** — play, pause, and stop named pose sequences with per-pose durations
- **Session persistence** — save/load sessions as JSON; **autosave** on every change
- **Clipboard export** — copy individual poses or full sequences as Python `dict` literals ready to paste into scripts
- **Import** — paste exported sequences back into the GUI

Session files are stored as `pose_session.json` (manual save) and `pose_editor_autosave.json` (automatic).

---

## 🎬 Scripted Sequences (`bottle_orient.py`)

Defines poses as Python dicts and executes them in sequence — useful for repeatable manipulation tasks.

```python
from utils.ExARM import ExArm
# See bottle_orient.py for the full example
python bottle_orient.py
```

Poses are defined per-finger for readability:

```python
dict(
    duration=2,
    index  =[70, 22, 85, 22],
    middle =[0,  0,  0,  0 ],
    ring   =[-70, 22, 85, 22],
    thumb  =[-45, -100, 30, 42],
),
```

---

## 🔄 Joint Mapping

The LEAP Hand's physical Dynamixel IDs differ from the logical joint ordering used throughout this SDK. `LeapHand.py` handles this remapping transparently.

**Logical ordering** (used in all public APIs):
```
Index  : joints  0–3   (abduct, flex, PIP, DIP)
Middle : joints  4–7
Ring   : joints  8–11
Thumb  : joints 12–15
```

**Physical → Logical remapping:**
```python
LOGICAL_TO_PHYSICAL = [8, 9, 10, 11,   # Index
                        4, 5,  6,  7,   # Middle
                        0, 1,  2,  3,   # Ring
                       12,13, 14, 15]   # Thumb
```

Similarly, `SimHand.py` maps MuJoCo `qpos` indices to the same logical ordering via `SIM_TO_REAL` / `REAL_TO_SIM`.

---

## 📐 Architecture Overview

```
ExArm  ──────────────────────────────────────────────
  │                                                  │
  ├── LeapHand  (real hardware, Dynamixel SDK)        │
  │     └── GroupSyncWrite / GroupSyncRead            │
  │                                                   │
  └── SimHand  (MuJoCo passive viewer)               │
        └── custom marker rendering                  │

Vision_Retargeting                                    
  │                                                  
  ├── MediaPipe HandLandmarker (webcam)               
  ├── Palm frame construction + axis remap           
  ├── LeapKinematics.ik()  (L-BFGS-B multi-start)    
  └── ExArm.set_goal_positions_degree()              

Ex-GUI                                                
  ├── PoseManager   (pose CRUD + JSON serialisation)  
  ├── RobotController  (threaded 30 Hz update loop)   
  ├── PlaybackController  (threaded pose sequencing)  
  └── PoseEditorGUI  (Tkinter widgets + joint sliders)
```

---

## ⚙️ Configuration Reference

| Parameter | Location | Description |
|---|---|---|
| `port` | `ExArm` / `LeapHand` | Serial port (e.g. `"COM4"`, `"/dev/ttyUSB0"`) |
| `baudrate` | `ExArm` / `LeapHand` | Dynamixel baud rate (e.g. `4000000`) |
| `ids` | `ExArm` / `LeapHand` | List of 16 Dynamixel motor IDs |
| `offsets` | `ExArm` / `LeapHand` | Per-joint angle offsets in degrees (logical order) |
| `model_path` | `ExArm` / `SimHand` | Path to MuJoCo URDF/XML model |
| `mode` | `ExArm` | `"real"`, `"sim"`, or `"both"` |
| `HAND_SCALE_M` | `Vision_Retargeting` | Robot wrist→MCP scale factor (metres) |
| `AXIS_MAP` | `Vision_Retargeting` | MediaPipe → LEAP axis remapping matrix |
| `PALM_OFFSET` | `Vision_Retargeting` | IK origin offset (metres) |
| `ALPHA` | `Vision_Retargeting` | Smoothing factor for vision output |

---

## 🤝 Acknowledgements

- [LEAP Hand](https://leaphand.com/) — Kenneth Shaw, Ananye Agarwal, Deepak Pathak (CMU)
- [Dynamixel SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK) — ROBOTIS
- [MuJoCo](https://mujoco.org/) — DeepMind
- [MediaPipe](https://mediapipe.dev/) — Google

---
