# Ex-ARM: LEAP Hand control and perception

Ex-ARM is a Python repository for controlling a 16-joint LEAP Hand and studying what the hand can learn from its own joint angles. It includes a Dynamixel hardware interface, a MuJoCo hand viewer, hand kinematics, webcam control, pose and motion tools, and two object-recognition experiments:

- **Physical-only recognition:** collect labeled real-hand trials and identify an object's class from the final angles of every configured grip. It offers a nearest-trial baseline and a small neural classifier.
- **Simulation-trained recognition:** grasp fixed CAD objects across a size sweep in MuJoCo, train a class-and-size model, then apply the same grasp to the real hand. This is a separate experimental pipeline; its real-hand accuracy has not been established.

The [manuscript](Manuscript/Manuscript.tex) focuses on the planned 20-object physical-only study and reports the available pilot data. The repository does not yet contain a completed 20-class dataset.

## Contents

- [Choose a program](#choose-a-program)
- [Install and configure](#install-and-configure)
- [Repository map](#repository-map)
- [Shared hand control and conventions](#shared-hand-control-and-conventions)
- [The controlled physical grasp](#the-controlled-physical-grasp)
- [Physical-only object recognition](#physical-only-object-recognition)
- [Simulation-trained class and size recognition](#simulation-trained-class-and-size-recognition)
- [Pose editing and sequence playback](#pose-editing-and-sequence-playback)
- [Vision control and motion teaching](#vision-control-and-motion-teaching)
- [Kinematics, diagnostics, and utilities](#kinematics-diagnostics-and-utilities)
- [Data and configuration reference](#data-and-configuration-reference)
- [Known limits and troubleshooting](#known-limits-and-troubleshooting)

## Choose a program

Start with `python Project.py`. The [project launcher](Project.py) groups the existing scripts into **Physical-only**, **Simulation-trained**, **Pose**, **Vision**, and **Diagnostics** menus. It starts each program in the repository root, where its relative data paths work. The scripts also remain directly runnable:

| Goal | Direct command | Requires |
| --- | --- | --- |
| Collect, resume, extend, train, or identify using physical trials | `python Physical_Only_Proprioception.py` | Real hand for collection and recognition; saved trials for training |
| Open the physical-only presentation window | `python Physical_Only_UI.py` | Tkinter, MuJoCo for the hand preview, real hand for grasps |
| Run and log one sequence of configured physical grips | `python Grasp.py` | Real hand |
| Generate CAD grasps and train the simulation model | `python Train_Proprioception.py` | MuJoCo, CadQuery; no real hand |
| Browse recorded simulated grasps | `python View_Proprioception.py` | MuJoCo viewer and previous training output |
| Identify class and size with the simulation-trained model | `python Run_Proprioception.py` | Trained simulation model and real hand |
| Edit and play hand poses | `python Ex-GUI.py` | Tkinter and the selected hand backend |
| Play a named JSON pose sequence | `python Pose_Sequence_Runner.py box_orient` | Selected hand backend |
| Direct webcam angle control | `python -m Vision.Vision_Teleop` | Webcam, MediaPipe, OpenCV, selected backend |
| Webcam fingertip retargeting | `python -m Vision.Vision_Retargeting` | Webcam, MediaPipe, OpenCV, SciPy, selected backend |
| Record, process, and replay vision motions | `python -m Vision.Vision_Kinesthetic_Teaching` | Webcam for recording; selected backend for replay |
| Display live joint state | `python test1.py` | Selected backend |

Run direct commands from the repository root. On macOS, replace `python` with `mjpython` when a command opens a **MuJoCo viewer**, including `View_Proprioception.py` and tools using `Connection.mode = "sim"` or `"both"`. The project launcher chooses `mjpython` for those commands on macOS. Headless simulation training uses ordinary Python. The physical-only UI embeds an offscreen MuJoCo rendering in Tkinter rather than opening the separate viewer.

## Install and configure

Python 3.11 is a practical starting point. Create and activate a virtual environment, then install the core requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-proprioception.txt
```

On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`. The [requirements file](requirements-proprioception.txt) pins NumPy, MuJoCo, and CadQuery and also installs `pynput` and `dynamixel-sdk`. Even a real-only `ExArm` currently imports both the hardware and MuJoCo backend modules, so those core packages are needed for the physical scripts too.

For kinematics and vision, install their additional packages:

```bash
python -m pip install scipy opencv-python mediapipe
```

The GUI programs also require a Python installation with Tkinter. Vision needs a webcam and the included [MediaPipe hand-landmarker model](Data/hand_landmarker.task). Real-hand commands need hand power and a working serial connection. Check package compatibility in the environment you actually use; the vision packages are not pinned in the core requirements file.

Before a motor-control run, edit [utils/Constants.py](utils/Constants.py):

1. Set `Connection.Port`, `baudrate`, `ids`, and `offsets` for your hand. The checked-in port is `COM3`, which is not a macOS serial-device name.
2. Set `Connection.mode` to `"real"`, `"sim"`, or `"both"` for the general hand, pose, vision, and diagnostic tools. The checked-in value is `"both"`, so those tools attempt **both hardware and a MuJoCo viewer**.
3. Check every entry of the ordered `Proprioception.grip` list and its current limits before a trial. `Grasp.py` and both recognition paths use this sequence. `Run_Proprioception.py` requires `Proprioception.hand_mode = "real"`, as checked in.
4. Place objects at the same position and orientation for training and recognition. The simulation pipeline uses `Proprioception.mount_translation_m` and `mount_rotation_rpy_deg`; the physical-only pipeline relies on consistent real placement.

## Repository map

```text
Project.py                         One menu for the existing programs
Grasp.py                           Shared controlled real-hand grasp
Physical_Only_Proprioception.py    Physical trial collection, training, inference
Physical_Only_UI.py                Presentation UI for that same physical workflow
Train_Proprioception.py           Simulated grasp generation and model training
View_Proprioception.py            Recorded simulated-grasp viewer
Run_Proprioception.py             Real-hand inference with the simulation model
Ex-GUI.py                          Pose editor
Pose_Sequence_Runner.py           JSON pose-sequence player
Vision/                           Webcam teleoperation, retargeting, motion teaching
utils/                            Hand interfaces, kinematics, models, simulation, helpers
Data/                             Hand model, CAD objects, examples, trials, models, runs
Manuscript/                       Research manuscript and its LaTeX files
requirements-proprioception.txt   Core Python dependencies
```

The root scripts are entry points. [utils/ExARM.py](utils/ExARM.py) joins the hardware and basic viewer backends; [utils/LeapHand.py](utils/LeapHand.py) talks to Dynamixel motors; [utils/SimHand.py](utils/SimHand.py) displays the hand. The two recognition implementations live in [utils/PhysicalOnlyNetwork.py](utils/PhysicalOnlyNetwork.py), [utils/ProprioceptionSimulation.py](utils/ProprioceptionSimulation.py), and [utils/ProprioceptionModel.py](utils/ProprioceptionModel.py). There is no `src/` package layer or installation step for this repository.

## Shared hand control and conventions

All public hand commands and measured positions use **16 angles in degrees**, in logical finger order:

| Indices | Finger | Joint order |
| --- | --- | --- |
| 0–3 | Index | Abduction, MCP flexion, PIP, DIP |
| 4–7 | Middle | Abduction, MCP flexion, PIP, DIP |
| 8–11 | Ring | Abduction, MCP flexion, PIP, DIP |
| 12–15 | Thumb | Base rotation, MCP, PIP, DIP |

`ExArm` accepts `mode="real"`, `"sim"`, or `"both"`. `set_goal_positions_degree()` sends the same logical joint vector to the active backend or backends. `get_state()` returns `(positions, velocities, currents)` for one backend and `{"real": ..., "sim": ...}` in `both` mode. **Space** while an `ExArm` program is active requests torque off and interrupts the program; scripts should close the hand in a `finally` block. Some individual programs also have their own Q or window controls.

```python
from utils.Constants import Connection
from utils.ExARM import ExArm

hand = ExArm(mode=Connection.mode, ids=Connection.ids,
             port=Connection.Port, baudrate=Connection.baudrate,
             offsets=Connection.offsets, model_path=Connection.model_path)
try:
    hand.set_torque_enabled(True)
    hand.set_goal_positions_degree([0.0] * 16)
    print(hand.get_state())
finally:
    hand.close()
```

The hardware driver uses synchronized position writes and bulk reads of position, velocity, and current. It applies per-joint offsets and a 180° position bias when converting angles to Dynamixel ticks. Its checked-in logical-to-physical mapping is the identity, but it is defined explicitly in `LeapHand.py`. Read velocities are raw motor register counts. The grasp compares signed current **register readings** to the configured limits; calibrate those values for the installed motors and operating mode.

`SimHand` loads [Data/Leap_Model/mujoco_robot.urdf](Data/Leap_Model/mujoco_robot.urdf), converts between logical and MuJoCo joint order, and sets joint positions directly in a passive viewer. Its currents are zeros; it does not simulate motor current or an object grasp. The **separate** `GraspSimulation` adds a CAD object and collision checks for recognition training. Kinematics uses radians internally and palm-frame positions in metres; methods ending in `_degree` accept or return degrees.

## The controlled physical grasp

[Grasp.py](Grasp.py) implements the grip sequence shared by both recognition paths. For each entry in `Proprioception.grip`, it moves the hand to that grip's `start_angles`, then advances all active fingers by its configured joint steps. For a moving joint, a signed current reading beyond `max_currents` stops **all four joints of that finger** at their measured positions. The other fingers continue until they contact or reach their configured target angles. It moves to the next grip's starting pose before running that grip.

After each grip, the program retries the final hand-state read for up to `final_read_timeout_s` if the hand momentarily stops reporting positions. One trial returns the final angles concatenated in grip-list order: **16 × number of grips** values, currently 48. `Data/Proprioception/grasp_results.csv` has columns `joint_0` through `joint_47`; `log.csv` has a `grip` column (numbered from 1) and one 16-angle physical-hand state per row. These two top-level files are replaced by the next complete trial. Physical-only collection copies and checks them inside the named trial folder immediately after success. Hand commands, live UI poses, and per-frame simulation poses remain 16 angles because the hardware still has 16 joints.

Use `python Grasp.py` to check this motion and log a grasp without running recognition. The physical-only **collection** workflow saves the result, commands the hand back to its starting angles, and waits before asking for the next object placement. The standalone `Grasp.py` and simulation-model runner do not perform that collection step.

## Physical-only object recognition

This is the main workflow for a real-object class experiment. It uses **no CAD object models or simulated training grasps**. The input to either classifier is the concatenated final-angle vector produced by the shared grip sequence. It predicts a **class**, not object size.

### Use: collect, resume, and extend a dataset

Run `python Physical_Only_Proprioception.py`, or choose **Physical-only object recognition → terminal menu** in `Project.py`:

| Menu option | Action |
| --- | --- |
| **1** | Create a collection plan or resume its remaining grasps; train the nearest-trial model when complete |
| **2** | Rebuild the nearest-trial model from saved, complete trials |
| **3** | Identify one object with the nearest-trial model |
| **4** | Quit this menu |
| **5** | Add trials to all or selected classes, or add new classes; then collect them |
| **6** | Train or retrain the physical-only neural classifier |
| **7** | Identify with the neural classifier, with repeat grasps when needed |
| **8** | Open the physical-only presentation UI |
| **9** | Retake a saved trial; keep the original in `retaken/` |

For a new study, choose a **new dataset name**, enter at least two class names and the desired number of trials per class, and collect the prompted trials. One completed trial contains all configured grips and is stored at `Data/Proprioception/Physical_Only/<dataset>/trials/<class>/<number>/` as `grasp_results.csv` plus the full `log.csv`. The program saves `collection.json` **before** collecting, so after an interruption you can choose option **1** and enter the same dataset name to resume. It skips complete trials and moves incomplete trial folders to `incomplete/` for inspection. After each saved collection trial, the hand is commanded open before the next placement prompt. Press **R** at the follow-up prompt to retake that trial immediately, or use option **9** later to select a class and trial number.

When all planned trials are present, option **1** trains `model.json` automatically. Choose **6 separately** to train `neural_model.npz`. To add data later, first finish any pending collection, then choose **5**. New trials or classes change the dataset: retrain the neural model before option **7**. Use a **new dataset** if you change the grasp, motor IDs, or offsets; the saved plan and models check these settings and reject incompatible measurements.

The checked-in `initial_9` pilot has three real grasps each of a 7.5 cm side-length cube, a 7.5 cm diameter sphere, and a 5 cm diameter cylinder. `Household_1` contains four household classes (`ball`, `bottle`, `bowl`, `mouse`) and a nearest-trial model. These are pilots, not the proposed 20-object experiment.

For the 20-object study, a practical sequence is: create a new named dataset, collect at least three independent real trials per class, train the neural model with option **6**, inspect its held-out report, then use option **7** or the UI for new recognition runs. Add further trials or classes with option **5** and retrain. Type the dataset name explicitly when training or identifying: pressing Enter at those prompts selects `PhysicalOnly.default_dataset_name`, which is `multi_grip_study` in the checked-in constants. Keep a separate record of object identity, placement, and session for any new test set; the program's built-in folds do not by themselves test a new day or a new physical instance.

### How it works: two class-only models

**Nearest-trial baseline.** [Physical_Only_Proprioception.py](Physical_Only_Proprioception.py) saves all measured angle vectors in `model.json`. Prediction chooses the class of the saved vector with the smallest Euclidean distance in **degrees**. Its leave-one-trial-out report excludes each trial in turn when scoring that trial. This baseline has no probability or unknown-object decision: it always chooses one of its saved classes.

**Neural classifier.** [PhysicalOnlyNetwork.py](utils/PhysicalOnlyNetwork.py) normalizes each angle by its grip and joint's configured start-to-target span; joints with zero span contribute no normalized change. A dense **(16 × grips) → 24 tanh → number-of-classes** network produces softmax class probabilities. Training uses NumPy and Adam, class balancing, weight decay, and, by default, 20 noisy copies plus the original of each real trial. The synthetic copies add both per-joint and shared-per-finger Gaussian angle noise to every grip. They help tolerance to small measurement changes but are not additional independent grasps.

The neural evaluation holds out **real trials before generating noise** in up to three folds. Validation loss selects training duration; a final model trains on all physical trials. Held-out logits set a probability temperature and held-out nearest-reference distances set the saved distance threshold. Training requires at least **three real trials per class**. The model saves a fingerprint of its physical samples and settings, and loading it asks for retraining if either changed.

A prediction can be a known class, `uncertain` (low top probability or small gap between the two leading classes), or `unknown` (too far from saved physical grasps). In terminal option **7**, an uncertain or unknown result can trigger another placement and grasp, up to `PhysicalOnly.network_max_grasps` (default three). The program averages per-class probabilities and uses the median reference distance across those grasps for the combined decision. It stops early when it accepts a known class. Unknown-object rejection has **not** been validated on a separate unseen-class set.

The checked-in `initial_9` neural report records **9/9** correct held-out pilot trials; its nearest-trial leave-one-out result is also **9/9**. The `Household_1` nearest-trial model records **11/12** leave-one-out trials correct. These results concern repeated grasps of a few available objects and do not establish accuracy for 20 classes, new sizes, new object instances, or new sessions.

### Use: presentation window

Open `python Physical_Only_UI.py` or choose option **8** in the physical-only terminal menu. The [UI](Physical_Only_UI.py) operates on the **same datasets and models**:

- **Collect:** create a plan, collect the next pending grasp, or expand a completed plan. The hand opens after each saved collection grasp.
- **Train:** build the nearest-trial model or neural classifier.
- **Recognize:** run a nearest-trial grasp or start and repeat a neural reading. The neural view shows each class probability and the final decision; the nearest-trial view shows per-class angle distances.

The MuJoCo hand panel mirrors measured physical joint angles for display. It does not generate training data or determine contact. Drag to rotate and scroll to zoom. Hardware and training jobs run in background threads so the Tk window can remain responsive. If offscreen MuJoCo rendering cannot start, the UI displays a preview-unavailable message while its other controls remain available.

## Simulation-trained class and size recognition

This pipeline trains from [STEP objects](Data/Objects/) and the MuJoCo hand model, then applies the shared physical grasp at inference. Its saved `model.npz` is unrelated to the physical-only `neural_model.npz`.

### Use: train, inspect, and run

1. Check the object entries, grasp, mount pose, size sweep, and joint offsets in [utils/Constants.py](utils/Constants.py).
2. Run `python Train_Proprioception.py`. It regenerates the simulation CSVs and trains the model without commanding the real hand.
3. Run `python View_Proprioception.py` to inspect recorded scenes (`mjpython` on macOS). Select an object and scale in the terminal. In the viewer: **Space** plays or pauses; **Left/Right** steps; **R** restarts; **N/P** changes scene; **M/Esc** returns to the terminal menu; **Q** quits.
4. With the correct real-hand port and the same object placement, run `python Run_Proprioception.py`. It checks that the saved model matches the current settings before opening the hand, then prints a class and size or `unknown` and saves `recognition_result.json`.

### How simulation and training work

[ProprioceptionSimulation.py](utils/ProprioceptionSimulation.py) imports each STEP file through CadQuery, tessellates it, converts its geometry from **millimetres to metres**, scales it, and adds a fixed mesh object to the MuJoCo hand model. It checks hand-object contacts as the configured grasp advances. Contact on any link stops that whole finger. It refines the last clear angle by bisection and records that pose; an angle target can also stop a finger. An unexpected palm or stationary-finger contact is an error. The simulator sets joint positions directly; it does not model motor force, compliance, current, or communication delay.

The checked-in sweep is **150% down to 50% in 5% increments**: 21 sizes for each of the cube, sphere, and cylinder, or 63 simulated grasps. The label `size_m` means **cube side length** for `cube` and **diameter** for `sphere` and `cylinder`. Each class's `reference_size_m` in `Proprioception.objects` is its size at 100%; changing it changes the size label, while replacing or rescaling the STEP file changes the actual simulated geometry. Check both together.

[ProprioceptionModel.py](utils/ProprioceptionModel.py) normalizes final angles by each grip's span, ignoring joints with no planned movement. Its dense **(16 × grips) → 32 tanh → (classes + one scale output)** network predicts class probabilities and a continuous size factor. The reported size is that factor multiplied by the predicted class's reference size. NumPy/Adam minimizes class cross-entropy plus weighted squared scale error. Training adds 50 Gaussian-noise copies per simulated trial by default.

Every fifth size in the sweep is held out to measure interpolation to unseen simulated sizes; the final saved model then retrains on **all** simulated sizes. Recognition rejects a grasp as `unknown` if it is too far from reference angles, has low class probability, disagrees with the nearest reference class, or has similar distances to two classes. The distance threshold is derived from held-out **known** sizes, so this is not a validated novel-object detector.

The checked-in [training report](Data/Proprioception/training_report.json) records **13/15 (86.7%)** raw class predictions correct on held-out simulated sizes and **1.02 mm** mean absolute size error. Its rejection rules accepted **3/15** held-out grasps, all correctly classified. Those numbers are for simulation. The real current-threshold grasp can stop at different angles from instantaneous simulated contact; the simulation-trained model's physical performance is still an open research question.

### Add or change a simulated object

Put a STEP file in [Data/Objects/](Data/Objects/) and add a matching entry to `Proprioception.objects` in [utils/Constants.py](utils/Constants.py): `file`, `class_name`, `size_name`, and `reference_size_m`. The physical-only workflow does **not** use this list. Check the imported object's units, mount placement, clearance at the starting pose, contact behavior, and URDF joint limits, then rerun training and inspect the viewer. Replacing a STEP or URDF file also calls for retraining: model loading compares configuration values but does **not** hash geometry files. [Data/Objects/config.json](Data/Objects/config.json) is an older, unused configuration and must not be treated as the source of these settings.

## Pose editing and sequence playback

[Ex-GUI.py](Ex-GUI.py) is a Tkinter pose editor. Choose **Pose tools → Open pose editor** in `Project.py`, or run it directly. Its 16 sliders and angle entries let you edit fingers, copy a finger pose, zero joints, and add, update, duplicate, insert, reorder, or delete named poses. A background controller sends the current angles at about 30 Hz **when torque is enabled**. Playback holds each saved pose for its configured duration; play, pause, and stop are available. The GUI can copy a pose or sequence as a Python-style dictionary and import a pasted sequence.

Manual session save/load uses `Data/Ex-GUI/pose_session.json`; autosave uses `pose_editor_autosave.json` in that folder. These session files are a **list of named poses with flat 16-angle arrays**. If backend connection fails, the GUI reports a warning and uses its mock controller; moving a slider in mock mode does not move the hand.

[Pose_Sequence_Runner.py](Pose_Sequence_Runner.py) plays a **different JSON format** from [Data/Example_pose_series/](Data/Example_pose_series/). Choose a sequence in the project menu or run, for example, `python Pose_Sequence_Runner.py bottle_orient`. Each pose has a positive `duration` and four arrays of four degree angles:

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

The runner validates that structure, flattens the four fingers into logical joint order, and repeatedly commands each target for its duration. It uses `Connection.mode`, so check the selected backend and hand clearance before playback. The editor's session JSON is **not** directly accepted by this runner.

## Vision control and motion teaching

The three [Vision](Vision/) programs use the included MediaPipe hand-landmarker model with webcam frames. They send commands to `ExArm` using `Connection.mode`. Their camera-to-hand calibration is currently kept **inside the individual scripts**, separate from `utils/Constants.py`.

| Program | Use and internal method | Output or control |
| --- | --- | --- |
| [Vision_Teleop.py](Vision/Vision_Teleop.py) | Estimates joint flexion from three-point landmark angles, clips and smooths a 16-angle pose; no IK | Streams joint goals from the live webcam. Its OpenCV preview and Q handling are currently commented out, so stop it from the terminal or via the `ExArm` Space interruption. |
| [Vision_Retargeting.py](Vision/Vision_Retargeting.py) | Builds a palm frame, maps four human fingertips into the LEAP frame, solves bounded per-finger IK, and smooths the answer | Streams goals and shows an OpenCV preview; press **Q** to exit that window. In simulation mode, it also asks the viewer to draw target markers. |
| [Vision_Kinesthetic_Teaching.py](Vision/Vision_Kinesthetic_Teaching.py) | Records landmarks, processes them through the same fingertip mapping and IK offline, then interpolates the angle trajectory for replay | Menu options **1 record**, **2 process**, and **3 replay**; generated files go to `Data/Motions/`. |

In the retargeting and teaching scripts, `HAND_SCALE_M`, `AXIS_MAP`, `PALM_OFFSET`, smoothing, and IK settings define the webcam-to-hand calibration. Fingertip positions and IK targets are in the hand's palm frame in **metres**. Retargeting warm-starts the IK solver from the previous video frame. Teaching records `<name>.avi` and `<name>_landmarks.pkl`, processes `<name>_trajectory.npy` of shape `(frames, 16)` in **degrees**, and replays interpolated commands. Its menu can also list and delete saved motions. The combined **record → process → replay** menu option currently passes an unsupported argument to `replay_motion`; use options **1, 2, then 3** separately.

## Kinematics, diagnostics, and utilities

[LeapKinematics.py](utils/LeapKinematics.py) implements forward kinematics for the four fingertips and their links, numerical finger Jacobians, and SciPy L-BFGS-B inverse kinematics within its hard-coded joint limits. Public angle inputs and outputs are **radians**, except methods ending in `_degree`. Positions use the palm frame in **metres**. The IK result includes `success` and `error_m`; inspect them before using a target.

```python
import numpy as np
from utils.LeapKinematics import LeapKinematics

kin = LeapKinematics()
tips_m = kin.fk(np.zeros(16))           # four fingertip XYZ positions
q_rad, info = kin.ik_finger(0, tips_m[0])
print(info["success"], info["error_m"])
```

`test1.py` uses [BoxPrinter.py](utils/BoxPrinter.py) to show live positions, velocities, and currents in an updating terminal layout; it is a **manual diagnostic**, not an automated test. `test2.py` is an older standalone current-threshold experiment with its **own** grasp dictionary. It is not the grasp used for proprioception training or recognition and is not in the project launcher's normal workflow. `utils/BoxPrinter.py` manages ANSI terminal boxes and terminal-size/layout checks.

The hand CAD files are in [Data/Leap_Model/](Data/Leap_Model/): `robot.urdf`, the MuJoCo-compatible `mujoco_robot.urdf`, STL meshes, and part files. [URDF_Convertor.py](utils/URDF_Convertor.py) removes `package:///` mesh prefixes, but its current `INPUT` and `OUTPUT` constants point to `Data/robot.urdf` and `Data/mujoco_robot.urdf`, not the files under `Data/Leap_Model/`. Correct those paths before invoking the converter. The hard-coded limits in `LeapKinematics.py` are independent of the URDF and have not been reconciled with all of the wider grasp limits.

## Data and configuration reference

| Location | What it contains | Writer or reader |
| --- | --- | --- |
| [Data/Leap_Model/](Data/Leap_Model/) | URDFs, STL collision/visual meshes, part files | `SimHand`, `GraspSimulation`, UI preview, kinematics reference |
| [Data/Objects/](Data/Objects/) | STEP cube, sphere, cylinder; older unused `config.json` | Simulation training and viewer read the STEP files |
| [Data/Example_pose_series/](Data/Example_pose_series/) | Named JSON pose sequences | Pose sequence runner |
| [Data/Ex-GUI/](Data/Ex-GUI/) | Editor session and autosave JSON | Pose editor |
| `Data/Motions/` | Recorded video, landmarks, processed trajectories | Vision kinesthetic teaching |
| [Data/hand_landmarker.task](Data/hand_landmarker.task) | MediaPipe hand detection model | All vision scripts |
| [Data/Proprioception/](Data/Proprioception/) | Simulated-grasp CSVs, sim model and report, latest raw physical grasp and sim-model recognition result | `Train_Proprioception.py`, `Grasp.py`, `Run_Proprioception.py` |
| [Data/Proprioception/Physical_Only/](Data/Proprioception/Physical_Only/) | Named real-trial datasets, baseline/neural models, reports and saved inference runs | Physical-only terminal program and UI |
| [Manuscript/](Manuscript/) | LaTeX research manuscript and build files | Research writing, separate from runtime code |

Inside a physical dataset, `collection.json` is the planned classes, counts, ordered grip list, motor IDs, and offsets. `trials/<class>/<number>/` contains completed trial records. Terminal option **9** retakes a saved trial: it verifies the replacement before moving the original into `retaken/<class>/`, then retrains the nearest-trial model. During terminal collection, **R** after saving a trial retakes it immediately. Retrain the neural model with option **6** after any retake. `model.json` is the nearest-trial baseline. `neural_model.npz` and `neural_report.json` are the physical classifier and its held-out report. `runs/` holds nearest-trial recognition, while `neural_runs/` holds individual and combined neural recognition. The latest top-level `Data/Proprioception/log.csv` and `grasp_results.csv` are overwritten by the next trial; the copied trial and run folders preserve earlier measurements.

Inside `Data/Proprioception/`, simulation training writes `simulation.csv` (one final pose per object/scale), `training_steps.csv` (viewer frames), `training_samples.csv` (original and noisy model inputs), `validation.csv`, `training_report.json`, and `model.npz`. Physical inference with that model writes `recognition_result.json`. Re-running simulation training replaces its generated model and CSVs, so copy an experiment elsewhere first if you need to preserve a previous run.

[utils/Constants.py](utils/Constants.py) is the main place to change experiment settings:

| Class | Main settings |
| --- | --- |
| `Connection` | General backend mode, port, baud rate, motor IDs and offsets, MuJoCo hand-model path |
| `Proprioception` | Shared grasp, current limits, timing and read retries; CAD objects and reference sizes; fixed mount and scale sweep; simulation noise, class-and-size network, rejection rules, and output paths |
| `PhysicalOnly` | Dataset root and defaults; UI dimensions; physical-network training, noise, validation, rejection thresholds, and repeat-grasp limit |

There are a few legacy local constants in the vision scripts, the pose runner, kinematics module, and `URDF_Convertor.py`; changing `Constants.py` does not update those automatically. If a saved model says the settings or physical trial fingerprint changed, retrain the corresponding model. Changing the number, order, or angles of the grips requires a **new physical dataset** and rerunning simulation training. The included pilot datasets and reports were recorded with a previous single-grip configuration and cannot train or run the current three-grip models; their reported results are historical.

## Known limits and troubleshooting

| Symptom or question | Check |
| --- | --- |
| The general hand tools try to open a serial port unexpectedly | The checked-in `Connection.mode` is `"both"`; select `"sim"` for a viewer-only run or configure the real port. |
| A physical tool cannot communicate with the hand | Check power, serial-device name, baud rate, motor IDs, offsets, and whether another program owns the port. `Grasp.py` retries a missing final read for a limited time. |
| A physical model refuses to load | Check whether the grasp, calibration, class counts, trials, or neural settings changed; complete collection and retrain. |
| The sim-to-real result is `unknown` | Inspect `Data/Proprioception/recognition_result.json` for the reason and compare the measured angles with `simulation.csv`; different real contact and simulated collision stop angles are a known gap. |
| The MuJoCo viewer fails on macOS | Launch the viewer script with `mjpython` in an environment with MuJoCo and a desktop session; the project menu chooses it when available. |
| The physical UI has no MuJoCo hand image | The preview needs MuJoCo offscreen rendering and the configured URDF. The UI displays the preview error in its canvas. |
| The pose GUI moves but hardware does not | Check `Connection.mode`, torque state, and any mock-mode connection warning. |
| Vision does not track or cannot start | Check the webcam, `Data/hand_landmarker.task`, vision packages, and the calibration constants in the selected script. |
| The simulation viewer has no scenarios | Run `Train_Proprioception.py` first to create `training_steps.csv`. |

The fixed-pose simulation uses one CAD model per class and direct position stepping; it does not reproduce real compliance or current-triggered stopping. The physical-only pilots use few object instances and repeats, and the 20-object study remains planned. Neither rejection rule has been evaluated against a dedicated unseen-object test set. The repository has no automated hardware or camera test suite; validate each workflow on the actual setup before reporting research performance.
