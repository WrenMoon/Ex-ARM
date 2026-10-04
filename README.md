# Ex-ARM: LEAP Hand control, simulation, and perception

Ex-ARM is a Python project for a 16-joint LEAP Hand. It brings together Dynamixel motor control, a MuJoCo hand viewer, forward and inverse kinematics, webcam teleoperation, a pose-sequencing GUI, recorded motion replay, and proprioceptive object recognition. Each tool can be used independently to control, observe, teach, or study the hand.

The project is under active development. The included proprioception model has been evaluated on simulated grasps; recognition accuracy on the physical hand has not yet been measured. The [research manuscript](Manuscript/Manuscript.tex) describes that distinction and the current experiment.

## Contents

- [What is included](#what-is-included)
- [Setup](#setup)
- [Architecture and joint conventions](#architecture-and-joint-conventions)
- [Controlling the hand](#controlling-the-hand)
- [Forward and inverse kinematics](#forward-and-inverse-kinematics)
- [Vision-based control and motion teaching](#vision-based-control-and-motion-teaching)
- [Pose editor and scripted sequences](#pose-editor-and-scripted-sequences)
- [Proprioceptive object recognition](#proprioceptive-object-recognition)
- [Data, configuration, and troubleshooting](#data-configuration-and-troubleshooting)
- [Current limitations](#current-limitations)

## What is included

| Area | Main files | Purpose |
| --- | --- | --- |
| Hardware control | [`utils/LeapHand.py`](utils/LeapHand.py), [`utils/ExARM.py`](utils/ExARM.py) | Command motors, read joint state, and select real, simulated, or combined operation. |
| Basic hand simulation | [`utils/SimHand.py`](utils/SimHand.py), [`Data/Leap_Model/`](Data/Leap_Model/) | Display and position the LEAP Hand in MuJoCo. |
| Kinematics | [`utils/LeapKinematics.py`](utils/LeapKinematics.py) | Compute fingertip positions and solve joint angles for fingertip targets. |
| Live vision control | [`Vision/Vision_Teleop.py`](Vision/Vision_Teleop.py), [`Vision/Vision_Retargeting.py`](Vision/Vision_Retargeting.py) | Follow a human hand seen by a webcam using direct angle mapping or inverse kinematics. |
| Motion teaching | [`Vision/Vision_Kinesthetic_Teaching.py`](Vision/Vision_Kinesthetic_Teaching.py) | Record hand landmarks, process an angle trajectory, and replay it. |
| Pose editing and playback | [`Ex-GUI.py`](Ex-GUI.py), [`Pose_Sequence_Runner.py`](Pose_Sequence_Runner.py) | Build, save, and play 16-joint pose sequences. |
| Proprioception research | [`Grasp.py`](Grasp.py), [`Train_Proprioception.py`](Train_Proprioception.py), [`View_Proprioception.py`](View_Proprioception.py), [`Run_Proprioception.py`](Run_Proprioception.py) | Generate simulated grasps, train a class-and-size model, inspect trials, and run recognition on the physical hand. |
| Diagnostics and utilities | [`test1.py`](test1.py), [`utils/BoxPrinter.py`](utils/BoxPrinter.py), [`utils/URDF_Convertor.py`](utils/URDF_Convertor.py) | Display live motor state and assist with URDF mesh paths. |

The [`Data/`](Data/) folder contains the hand model, three STEP objects, example pose series, the MediaPipe hand landmark model, saved GUI sessions, and generated recognition data. The [`Manuscript/`](Manuscript/) folder contains the LaTeX research manuscript and its PDF.

## Setup

Use Python 3.10 or newer; Python 3.11 is a practical choice for the pinned packages. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-proprioception.txt
```

On Windows, activate with `.venv\Scripts\activate`. The requirements file installs NumPy, MuJoCo, CadQuery, `pynput`, and the Dynamixel SDK. These cover the core hand interface and the proprioception workflow. Install the additional libraries used by kinematics and vision:

```bash
python -m pip install scipy opencv-python mediapipe
```

The pose GUI also needs Tkinter in the Python installation. The vision programs need a webcam and [`Data/hand_landmarker.task`](Data/hand_landmarker.task). A LEAP Hand, working motor power, and a serial connection are needed only for real-hardware operation.

**Run commands from the repository root.** Several model and data paths are relative to it. On macOS, use `mjpython` in place of `python` for programs that open a MuJoCo viewer, including `ExArm` in `sim` or `both` mode. Headless training with `Train_Proprioception.py` uses `python`.

## Architecture and joint conventions

```text
Pose editor / sequences ─┐
Vision tools ─────────────┼──> ExArm ──> LeapHand ──> Dynamixel motors
Manual control ───────────┘       └────> SimHand  ──> MuJoCo hand viewer

STEP objects + hand URDF ──> ProprioceptionSimulation ──> final angles
                                              final angles ──> ProprioceptionModel
Physical hand ──> Grasp ──> measured final angles ──────────> class and size
```

[`utils/ExARM.py`](utils/ExARM.py) defines `ExArm`, the common interface used by most applications. Set `mode="real"` for motors, `mode="sim"` for the basic MuJoCo hand viewer, or `mode="both"` to send commands to both. In single-backend mode, `get_state()` returns `(positions, velocities, currents)`. In `both` mode it returns a dictionary with `"real"` and `"sim"` entries. Commands are forwarded to active backends; the basic simulation sets joint positions directly rather than modeling motor motion. Its current values are zeros.

The public joint vector always contains **16 angles in degrees**:

| Logical indices | Finger | Joint meaning |
| --- | --- | --- |
| 0–3 | Index | Abduction, flexion, PIP, DIP |
| 4–7 | Middle | Abduction, flexion, PIP, DIP |
| 8–11 | Ring | Abduction, flexion, PIP, DIP |
| 12–15 | Thumb | Base rotation, MCP, PIP, DIP |

This is the order accepted by `ExArm.set_goal_positions_degree()` and returned by hardware state reads. [`utils/LeapHand.py`](utils/LeapHand.py) contains `LOGICAL_TO_PHYSICAL` for motor wiring; its current mapping is the identity mapping. [`utils/SimHand.py`](utils/SimHand.py) separately swaps the first two MuJoCo coordinates of each non-thumb finger to present the same logical order. Kinematics uses radians internally; its methods ending in `_degree` accept or return degrees.

## Controlling the hand

### Connection settings

[`utils/Constants.py`](utils/Constants.py) defines `Connection.mode`, `Port`, `baudrate`, `ids`, `offsets`, and `model_path` for the general control tools. The checked-in values are `mode="sim"`, `Port="COM3"`, 4,000,000 baud, IDs 0–15, zero angle offsets, and `Data/Leap_Model/mujoco_robot.urdf`. Set the port and calibrations for your own hardware before changing the mode to `real` or `both`. The proprioception runner has its own `Proprioception.hand_mode`, described later.

A minimal `ExArm` session uses the configured backend:

```python
from utils.Constants import Connection
from utils.ExARM import ExArm

hand = ExArm(
    mode=Connection.mode,
    ids=Connection.ids,
    port=Connection.Port,
    baudrate=Connection.baudrate,
    offsets=Connection.offsets,
    model_path=Connection.model_path,
)
try:
    hand.set_torque_enabled(True)
    hand.set_goal_positions_degree([0.0] * 16)
    print(hand.get_state())
finally:
    hand.close()
```

With the default `sim` mode this opens the hand viewer. For direct low-level control, `LeapHand` exposes `set_torque_enabled()`, `set_goal_positions_degree()`, `get_state()`, operating-mode and goal-current helpers, and `close_port()`. It uses synchronized position writes and a bulk read of position, velocity, and current from all motors. It converts degrees to motor ticks using the configured offsets and a 180° position bias. Measured velocities and currents are the signed values read from Dynamixel registers.

While `ExArm` is active, its keyboard listener uses **Space** to disable torque and interrupt the main program. Keep cleanup in a `finally` block when writing a script. [`test1.py`](test1.py) is a manual state monitor: it uses [`utils/BoxPrinter.py`](utils/BoxPrinter.py) to display all positions, velocities, and currents in a live terminal layout. Run it with `python test1.py`, or `mjpython test1.py` on macOS if `Connection.mode` opens the simulation viewer.

### Basic MuJoCo backend

`SimHand` loads `Connection.model_path`, opens a passive MuJoCo viewer, and exposes the same position/state/reset/close methods used by `ExArm`. It also supports custom visual markers for fingertip targets. This backend is useful for inspecting a pose or following GUI and vision commands. It **does not add a grasped object or stop at collision**; the separate proprioception simulator does that.

The original [`robot.urdf`](Data/Leap_Model/robot.urdf), MuJoCo-compatible [`mujoco_robot.urdf`](Data/Leap_Model/mujoco_robot.urdf), and STL link meshes are under `Data/Leap_Model/`. Joint limits at 0, 8, 13, and 14 were widened to fit the proprioception grasp. The [`utils/URDF_Convertor.py`](utils/URDF_Convertor.py) utility strips ROS `package:///` mesh prefixes, but its current input/output constants still point at `Data/` rather than `Data/Leap_Model/`; update those paths before using it.

## Forward and inverse kinematics

[`utils/LeapKinematics.py`](utils/LeapKinematics.py) computes each fingertip position from a 16-joint pose using the hand's fixed transforms. It also solves fingertip targets with SciPy's bounded L-BFGS-B optimizer, supports multiple starting guesses, and provides numerical finger Jacobians. Fingertip positions and IK targets are in the hand's palm frame, in **metres**. Kinematic angle inputs and outputs are in **radians** unless a method name ends in `_degree`.

```python
import numpy as np
from utils.LeapKinematics import LeapKinematics

kin = LeapKinematics()
q = np.zeros(16)                         # radians, logical joint order
tips = kin.fk(q)                         # four fingertip XYZ positions, metres
index_q, info = kin.ik_finger(0, tips[0])
whole_hand_q, infos = kin.ik(tips, q0=q)
tips_from_degrees = kin.fk_degree(np.zeros(16))
```

Other methods include `fk_finger()`, `fk_all_links()`, `ik_finger_multistart()`, `ik_degree()`, `jacobian_finger()`, `clip_to_limits()`, and `is_within_limits()`. Check the returned `info["success"]` and `info["error_m"]` before using an IK result. The module has its own hard-coded `JOINT_LIMITS`; these have not yet been synchronized with the wider URDF limits at joints 0, 8, 13, and 14.

## Vision-based control and motion teaching

The programs in [`Vision/`](Vision/) use MediaPipe's hand landmarks from a webcam. They control whichever backend is selected by `Connection.mode`. They are separate from the proprioception model; the object-recognition model never receives camera images.

### Direct angle teleoperation

[`Vision/Vision_Teleop.py`](Vision/Vision_Teleop.py) calculates finger flexion angles from landmark triplets, assembles a 16-joint pose, clips it, smooths it, and streams goals to `ExArm`. This route does not solve inverse kinematics. From the repository root:

```bash
python -m Vision.Vision_Teleop
```

Use `mjpython -m Vision.Vision_Teleop` on macOS when the selected backend opens MuJoCo. This experimental script currently has its camera preview and Q-key handling commented out; stop it from the terminal when finished.

### Fingertip retargeting

[`Vision/Vision_Retargeting.py`](Vision/Vision_Retargeting.py) builds a palm frame from landmarks, rescales and remaps fingertip positions to the LEAP frame, solves IK for each finger, smooths the joint angles, and streams them to `ExArm`. It displays the tracked hand in an OpenCV window; press **Q** there to quit.

```bash
python -m Vision.Vision_Retargeting
```

`HAND_SCALE_M`, `AXIS_MAP`, `PALM_OFFSET`, and the smoothing/IK settings near the top of that file define the camera-to-hand calibration. Match them to your camera and hand setup. The script can also show fingertip target markers in the MuJoCo viewer through `SimHand`.

### Record, process, and replay a motion

[`Vision/Vision_Kinesthetic_Teaching.py`](Vision/Vision_Kinesthetic_Teaching.py) provides a terminal menu with three stages:

1. **Record:** save a webcam video and per-frame MediaPipe landmarks in `Data/Motions/<name>.avi` and `<name>_landmarks.pkl`.
2. **Process:** solve IK offline and save a 16-joint degree trajectory as `<name>_trajectory.npy`.
3. **Replay:** interpolate that trajectory and command the selected hand backend. The menu can also list or delete saved motions.

```bash
python -m Vision.Vision_Kinesthetic_Teaching
```

Its camera-to-hand calibration mirrors the retargeting script. The combined record/process/replay menu option currently calls replay with an unsupported argument; use menu options **1, 2, then 3** separately. The `Data/Motions/` directory is created when the program runs.

## Pose editor and scripted sequences

### Ex-GUI

[`Ex-GUI.py`](Ex-GUI.py) is a Tkinter editor for arranging and playing poses. Run `python Ex-GUI.py` from the repository root; use `mjpython Ex-GUI.py` on macOS when `Connection.mode` is `sim` or `both`.

- Each of the 16 joints has a slider, direct angle entry, and ±1°/±5° controls.
- Finger tools copy a four-joint pose between fingers; zero tools reset one finger or all joints.
- Poses have names and durations. You can create, update, duplicate, insert, delete, reorder, play, pause, and stop them.
- The GUI can copy a pose or sequence to the clipboard as Python-style dictionaries and import a pasted sequence.
- Manual sessions use [`Data/Ex-GUI/pose_session.json`](Data/Ex-GUI/pose_session.json); autosave uses [`pose_editor_autosave.json`](Data/Ex-GUI/pose_editor_autosave.json).

`RobotController` sends the current GUI angles at roughly 30 Hz when torque is enabled. The application uses `Connection.mode`, so the default connects to the basic MuJoCo viewer. If connection fails, it displays a warning and continues with a mock robot; visible GUI movement then does not mean motors are moving.

### JSON pose sequences

[`Pose_Sequence_Runner.py`](Pose_Sequence_Runner.py) validates and replays a named JSON sequence from [`Data/Example_pose_series/`](Data/Example_pose_series/). The repository includes `box_orient.json` and `bottle_orient.json`. For example:

```bash
python Pose_Sequence_Runner.py box_orient
```

Each pose supplies a positive duration in seconds and four arrays of four **degree** angles:

```json
{
  "poses": [
    {
      "duration": 1.0,
      "index": [0, 30, 0, 0],
      "middle": [0, 30, 0, 0],
      "ring": [0, 30, 0, 0],
      "thumb": [0, 0, 0, 0]
    }
  ]
}
```

The runner flattens these finger arrays into logical joint order and sends each pose for its duration using `Connection.mode`. Use `mjpython` on macOS if that mode opens MuJoCo. GUI session JSON uses a different format: each saved pose has a name, duration, and one flat 16-angle array.

## Proprioceptive object recognition

The recognition research asks whether the **final 16 joint angles after one controlled grasp** reveal an object's class and size. It currently considers a cube, sphere, and cylinder fixed at one common pose. Training uses collision in MuJoCo; the final physical program uses signed motor current as a contact proxy.

```text
STEP objects + fixed placement + size sweep
    -> MuJoCo grasp -> final angles -> train and validate
    -> saved class-and-size model

Physical object -> current-limited grasp -> measured final angles
                -> known class and size, or unknown
```

### Simulated grasps

[`utils/ProprioceptionSimulation.py`](utils/ProprioceptionSimulation.py) imports STEP geometry with CadQuery, converts millimetres to metres, adds a fixed object body and explicit contact pairs to the hand model, and replays the grasp configured in `Proprioception.grip`. The default scale sweep runs from **150% down to 50% in 5% steps**, giving 21 sizes for each of the three objects and 63 scenes. The cube's target size is its **side length**; sphere and cylinder targets are **radius**. All dimensions scale uniformly.

The grasp advances each active finger by its configured angle step. Contact on any link stops **all four joints of that finger**; other fingers keep moving. A finger can also finish at its angle target. MuJoCo refines contact to the last clear pose and records the final angles and every intermediate step. It sets positions directly rather than simulating motor dynamics or current. The older `Data/Objects/config.json` is not read by this pipeline; object definitions, mount pose, sweep, offsets, and grasp values are in [`utils/Constants.py`](utils/Constants.py).

```bash
python Train_Proprioception.py
```

This command regenerates the simulated grasp files, trains the model, and prints validation metrics. To inspect any recorded object/size trial, run `mjpython View_Proprioception.py` on macOS or `python View_Proprioception.py` elsewhere. A terminal menu selects the object and scale; the MuJoCo viewer replays the recorded grasp. **Space** plays/pauses, **Left/Right** step frames, **R** restarts, **N/P** change scenarios, **M/Esc** returns to the menu, and **Q** quits.

### Model and validation

[`utils/ProprioceptionModel.py`](utils/ProprioceptionModel.py) normalizes the 16 final angles by the configured grasp spans. A fully connected **32-unit tanh layer** feeds three class logits and one continuous scale output: **676 trainable parameters** for the three current classes. Predicted physical size is the scale factor multiplied by that class's reference size.

Training uses NumPy/Adam, class cross-entropy plus scale squared error, and 50 noisy copies per simulated grasp with 0.5° Gaussian angle noise by default. Every fifth scale is excluded from the validation model's training data. A final model is then fitted using all 63 simulated sizes. An `unknown` result is a rule-based rejection if a grasp is distant from references, has low class probability, disagrees with the nearest reference class, or is too similar to another class.

The checked-in report records **13/15 (86.7%) raw class predictions correct** and **1.36 mm** mean absolute size error on held-out simulated sizes. The rejection rule accepted **5/15**, all with the correct class. Its distance threshold was derived using those same held-out sizes, so accepted/rejected results are descriptive rather than an independent test of novel-object detection. No labeled real-hand recognition results are in the repository.

### Physical recognition

[`Grasp.py`](Grasp.py) slowly closes all four fingers with the configured target angles. When a joint's signed measured current exceeds its threshold, that **entire finger stops at its measured angles**. `python Grasp.py` performs and logs only this physical grasp. [`Run_Proprioception.py`](Run_Proprioception.py) loads the trained model first, runs the grasp on the real hand, prints class and size or `unknown`, and saves a JSON result:

```bash
python Run_Proprioception.py
```

Set `Connection.Port` and verify the hand's motor IDs, angle offsets, calibrated current thresholds, and object fixture before running it. `Proprioception.hand_mode` must remain `"real"` for the final recognition program. The physical object must be mounted in the same relative position used for training.

### Outputs and adding an object

All experiment outputs are under [`Data/Proprioception/`](Data/Proprioception/):

| File | Contents |
| --- | --- |
| `simulation.csv` | One final 16-angle grasp per object and scale, including size and finger stop reasons. |
| `training_steps.csv` | Recorded angle trajectory for every simulated scene; input to the viewer. |
| `training_samples.csv` | Original and noise-augmented angles used for the validation and final training stages. |
| `validation.csv`, `training_report.json` | Held-out predictions, rejection reasons, metrics, and training settings. |
| `model.npz` | Weights, object metadata, reference grasps, and stored settings. |
| `log.csv`, `grasp_results.csv` | Last physical grasp's state trace and final measured angles. |
| `recognition_result.json` | Last physical run's timestamp, final angles, and prediction. |

Training overwrites its generated CSVs and model; a physical run overwrites its last-run logs. To add a class, put a STEP file measured in millimetres in [`Data/Objects/`](Data/Objects/), add its metadata and reference size to `Proprioception.objects`, and retrain. Check that the object does not touch the hand at the initial pose and that the grasp fits the URDF limits. Retrain after changing STEP or URDF geometry: model loading checks many constants but does not hash those files.

## Data, configuration, and troubleshooting

| Path | Role |
| --- | --- |
| [`utils/Constants.py`](utils/Constants.py) | `Connection` controls the general hand backend. `Proprioception` controls the recognition grasp, objects, scale sweep, noise, network, rejection rules, and output paths. |
| [`Data/Leap_Model/`](Data/Leap_Model/) | Hand URDFs, STL meshes, and part files. |
| [`Data/Example_pose_series/`](Data/Example_pose_series/) | JSON sequences for the pose runner. |
| [`Data/Ex-GUI/`](Data/Ex-GUI/) | Saved and autosaved editor sessions. |
| `Data/Motions/` | Generated videos, landmark recordings, and IK trajectories from motion teaching. |
| [`Data/Objects/`](Data/Objects/) | Cube, sphere, and cylinder STEP models for recognition. |
| [`Data/Proprioception/`](Data/Proprioception/) | Generated simulations, model, validation, and physical recognition outputs. |
| [`Manuscript/`](Manuscript/) | Source and PDF for the research paper based on the implemented recognition study. |

| Symptom | What to check |
| --- | --- |
| Cannot open the hardware serial port | Set `Connection.Port` to the device on your system and check power, USB connection, baud rate, and motor IDs. |
| MuJoCo viewer fails on macOS | Run with `mjpython` from the environment in which MuJoCo is installed and use a desktop session. |
| GUI moves but the hand does not | Check `Connection.mode` and any connection warning; the GUI can fall back to a mock robot. |
| Kinematics or vision import fails | Install SciPy, OpenCV, and MediaPipe as applicable; run from the repository root. |
| Training viewer has no scenarios | Run `python Train_Proprioception.py` to generate `training_steps.csv`. |
| Model reports changed settings | Retrain after changing the grasp, object list, or training settings. |
| Recognition returns `unknown` | Read `recognition_result.json` for the rejection reason and compare measured angles with `simulation.csv`. |

## Current limitations

- `SimHand` is a kinematic MuJoCo viewer, not a dynamic motor model. The separate recognition simulator checks mesh collision, while the real grasp uses motor current; that sim-to-real difference still needs physical measurement.
- The recognition data use one fixed object pose, one CAD model per class, and uniform scale variation. The model receives final joint angles only. Unknown-object detection has not been tested against a separate set of unseen classes.
- Vision calibration constants are local to the vision scripts, and `LeapKinematics.JOINT_LIMITS` have not yet been updated to match the widened URDF limits at joints 0, 8, 13, and 14.
- The direct-angle vision script's preview/Q-key exit is currently commented out. The motion-teaching script's combined menu option is currently broken; its separate stages work as the intended route.
- `test1.py` is a manual state display, not an automated test suite. The GUI's mock fallback is useful for editing poses but cannot verify physical motion.

The included manuscript documents the current simulated recognition results and identifies real-hand validation as the next research step.
