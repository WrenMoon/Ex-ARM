"""Choose and replay recorded training grasps in the MuJoCo viewer."""

import csv
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco.viewer
import numpy as np

from utils.Constants import Proprioception
from utils.ProprioceptionSimulation import GraspSimulation


def load_training_steps():
    path = Path(Proprioception.training_steps_path)
    if not path.exists():
        raise FileNotFoundError("Run Train_Proprioception.py to record the training grasps first")
    scenarios = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if "grip" not in (reader.fieldnames or []):
            raise ValueError("Training steps use an older grip format; rerun training")
        for row in reader:
            key = (row["class"], int(float(row["scale_percent"])))
            angles = np.asarray([float(row[f"joint_{joint}"]) for joint in range(16)])
            reasons = [row[f"finger_{finger}_stop"] for finger in range(4)]
            scenarios.setdefault(key, []).append((int(row["grip"]), int(row["step"]),
                                                  angles, reasons))
    if not scenarios:
        raise ValueError("No recorded training steps found")
    for key, frames in scenarios.items():
        for grip_number in range(1, len(Proprioception.grip) + 1):
            steps = [frame[1] for frame in frames if frame[0] == grip_number]
            if steps != list(range(len(steps))) or not steps:
                raise ValueError(f"Incomplete training steps for {key}, grip {grip_number}")
        if [frame[0] for frame in frames] != sorted(frame[0] for frame in frames):
            raise ValueError(f"Training grips are out of order for {key}")
    return scenarios


def choose_scenario(scenarios):
    while True:
        objects = [entry["class_name"] for entry in Proprioception.objects
                   if any(key[0] == entry["class_name"] for key in scenarios)]
        print("\nObjects:")
        for index, name in enumerate(objects, 1):
            print(f"  {index}. {name}")
        choice = input("Select an object number, or Q to quit: ").strip().lower()
        if choice == "q":
            return None
        if not choice.isdigit() or not 1 <= int(choice) <= len(objects):
            print("Choose a listed number.")
            continue
        name = objects[int(choice) - 1]

        while True:
            scales = sorted((scale for object_name, scale in scenarios
                             if object_name == name), reverse=True)
            print(f"\n{name} sizes:")
            for index, scale in enumerate(scales, 1):
                print(f"  {index}. {scale}%")
            choice = input("Select a size number, B for objects, or Q to quit: ").strip().lower()
            if choice == "q":
                return None
            if choice == "b":
                break
            if choice.isdigit() and 1 <= int(choice) <= len(scales):
                return name, scales[int(choice) - 1]
            print("Choose a listed number.")


def show_scenario(key, frames):
    name, scale = key
    object_info = next(entry for entry in Proprioception.objects
                       if entry["class_name"] == name)
    simulation = GraspSimulation(object_info, scale)
    simulation.set_angles(frames[0][2])
    keys = SimpleQueue()
    frame_index = 0
    playing = True
    next_frame_time = time.monotonic() + Proprioception.step_time_s

    print(f"\nViewing {name} at {scale}% ({len(frames)} recorded steps)")
    print("Space: play/pause | Left/Right: step | R: restart | N/P: next/previous scenario")
    print("M or Esc: menu | Q: quit")
    with mujoco.viewer.launch_passive(simulation.model, simulation.data,
                                      key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = Proprioception.mount_translation_m
            viewer.cam.distance = 0.35
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25

        while viewer.is_running():
            changed = False
            now = time.monotonic()
            while not keys.empty():
                key_code = keys.get()
                if key_code == 32:  # Space
                    playing = not playing
                    next_frame_time = now + Proprioception.step_time_s
                elif key_code == 263:  # Left
                    playing = False
                    frame_index = max(0, frame_index - 1)
                    changed = True
                elif key_code == 262:  # Right
                    playing = False
                    frame_index = min(len(frames) - 1, frame_index + 1)
                    changed = True
                elif key_code == 82:  # R
                    frame_index = 0
                    playing = True
                    next_frame_time = now + Proprioception.step_time_s
                    changed = True
                elif key_code == 78:  # N
                    return "next"
                elif key_code == 80:  # P
                    return "previous"
                elif key_code in (77, 256):  # M or Esc
                    return "menu"
                elif key_code == 81:  # Q
                    return "quit"

            if playing and now >= next_frame_time:
                frame_index = min(frame_index + 1, len(frames) - 1)
                changed = True
                next_frame_time = now + Proprioception.step_time_s
                if frame_index == len(frames) - 1:
                    playing = False
            if changed:
                with viewer.lock():
                    simulation.set_angles(frames[frame_index][2])
                grip_number, step, _, reasons = frames[frame_index]
                if frame_index == len(frames) - 1 or not playing:
                    print(f"Grip {grip_number}, step {step}: {reasons}")
            viewer.sync()
            time.sleep(Proprioception.viewer_refresh_s)
    return "menu"


def main():
    scenarios = load_training_steps()
    order = [(entry["class_name"], scale)
             for entry in Proprioception.objects
             for scale in sorted((value for name, value in scenarios
                                  if name == entry["class_name"]), reverse=True)]
    selection = choose_scenario(scenarios)
    while selection is not None:
        action = show_scenario(selection, scenarios[selection])
        if action == "quit":
            return
        if action in ("next", "previous"):
            step = 1 if action == "next" else -1
            selection = order[(order.index(selection) + step) % len(order)]
        else:
            selection = choose_scenario(scenarios)


if __name__ == "__main__":
    main()
