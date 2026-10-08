"""Train and run a class-only model using grasps from the physical hand."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil

from Grasp import grasp
from utils.Constants import Connection, PhysicalOnly, Proprioception


def dataset_folder(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", name):
        raise ValueError("Use letters, numbers, spaces, underscores, or hyphens for names")
    return Path(PhysicalOnly.data_folder) / name


def read_angles(path):
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        expected = [f"joint_{joint}" for joint in range(16)]
        if reader.fieldnames != expected:
            raise ValueError(f"Expected 16 joint columns in {path}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one grasp in {path}")
    angles = [float(rows[0][name]) for name in expected]
    if not all(math.isfinite(angle) for angle in angles):
        raise ValueError(f"Invalid angle in {path}")
    return angles


def distance(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def predict(angles, samples):
    if len(angles) != 16 or not all(math.isfinite(angle) for angle in angles):
        raise ValueError("Expected 16 finite joint angles")
    if not samples:
        raise ValueError("The model has no training trials")
    nearest = min(samples, key=lambda sample: distance(angles, sample["angles_deg"]))
    return dict(class_name=nearest["class_name"],
                nearest_trial=nearest["trial"],
                distance_deg=distance(angles, nearest["angles_deg"]))


def train_dataset(name):
    folder = dataset_folder(name)
    trials_folder = folder / "trials"
    if not trials_folder.is_dir():
        raise FileNotFoundError(f"No trials in {trials_folder}")

    samples = []
    classes = []
    for class_folder in sorted(path for path in trials_folder.iterdir() if path.is_dir()):
        trial_files = sorted(class_folder.glob("*/grasp_results.csv"))
        if len(trial_files) != PhysicalOnly.trials_per_class:
            raise ValueError(f"{class_folder.name} needs {PhysicalOnly.trials_per_class} trials")
        classes.append(class_folder.name)
        for path in trial_files:
            samples.append(dict(class_name=class_folder.name,
                                trial=path.parent.name,
                                angles_deg=read_angles(path)))
    if len(classes) < 2:
        raise ValueError("At least two classes are needed")

    correct = 0
    for index, sample in enumerate(samples):
        others = samples[:index] + samples[index + 1:]
        result = predict(sample["angles_deg"], others)
        correct += result["class_name"] == sample["class_name"]

    model = dict(format_version=1,
                 method="nearest physical trial, Euclidean distance in degrees",
                 grip=Proprioception.grip,
                 motor_ids=Connection.ids,
                 motor_offsets_deg=Connection.offsets,
                 classes=classes,
                 samples=samples,
                 leave_one_out_correct=correct,
                 leave_one_out_total=len(samples))
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(f"Saved {len(samples)} physical trials to {folder / 'model.json'}")
    print(f"Leave-one-trial-out: {correct}/{len(samples)} correct")
    return model


def load_model(name):
    path = dataset_folder(name) / "model.json"
    model = json.loads(path.read_text())
    if model.get("format_version") != 1:
        raise ValueError("Unsupported physical-only model format")
    if model["grip"] != Proprioception.grip:
        raise ValueError("The grasp changed. Collect new physical trials before recognizing")
    if model["motor_ids"] != Connection.ids or model["motor_offsets_deg"] != Connection.offsets:
        raise ValueError("Motor IDs or angle offsets changed. Collect new physical trials")
    return model


def open_hand():
    from utils.ExARM import ExArm

    return ExArm(mode="real", ids=Connection.ids, port=Connection.Port,
                 baudrate=Connection.baudrate, offsets=Connection.offsets,
                 model_path=Connection.model_path)


def save_grasp(folder):
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Proprioception.grasp_result_path, folder / "grasp_results.csv")
    shutil.copy2(Proprioception.grasp_log_path, folder / "log.csv")


def collect_dataset(name):
    folder = dataset_folder(name)
    if folder.exists():
        raise FileExistsError(f"{folder} already exists. Choose a new dataset name")
    count = int(input("How many object classes? "))
    if count < 2:
        raise ValueError("Enter at least two classes")
    classes = []
    for index in range(count):
        class_name = input(f"Name for class {index + 1}: ").strip().lower()
        dataset_folder(class_name)
        if class_name in classes:
            raise ValueError("Class names must be different")
        classes.append(class_name)

    print(f"The hand will make {PhysicalOnly.trials_per_class} grasps per class.")
    print("Place each object at the same point and press Enter when ready.")
    hand = open_hand()
    try:
        for class_name in classes:
            for trial in range(1, PhysicalOnly.trials_per_class + 1):
                input(f"Place {class_name}, trial {trial}, then press Enter: ")
                grasp(hand)
                save_grasp(folder / "trials" / class_name / str(trial))
                print("Trial saved. Remove the object before the next trial.")
    finally:
        hand.close()
    return train_dataset(name)


def recognize(name):
    model = load_model(name)
    input("Place the object at the training position, then press Enter: ")
    hand = open_hand()
    try:
        angles = grasp(hand)
    finally:
        hand.close()

    result = predict(angles, model["samples"])
    timestamp = datetime.now(timezone.utc)
    run_folder = dataset_folder(name) / "runs" / timestamp.strftime("%Y%m%d_%H%M%S_%f")
    save_grasp(run_folder)
    output = dict(time_utc=timestamp.isoformat(), dataset=name,
                  final_angles_deg=angles, prediction=result)
    (run_folder / "recognition_result.json").write_text(json.dumps(output, indent=2) + "\n")
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print("\n" + "=" * 60)
    print(BOLD + f"FINAL RESULT: {result['class_name']}".center(60) + RESET)
    print("=" * 60 + "\n")
    # print(f"Nearest physical trial: {result['class_name']} {result['nearest_trial']}")
    # print(f"Angle distance: {result['distance_deg']:.2f} degrees")
    print(f"Result saved to {run_folder}")
    return output

3
def ask_dataset_name(new=False):
    if new:
        name = input("New dataset name: ").strip()
        if not name:
            raise ValueError("Enter a name for the new dataset")
        return name
    name = input(f"Dataset name [{PhysicalOnly.default_dataset_name}]: ").strip()
    return name or PhysicalOnly.default_dataset_name


def main():
    while True:
        print("\nPhysical-only proprioception")
        print("1. Collect three real grasps per class and train")
        print("2. Train from saved physical trials")
        print("3. Identify an object with the real hand")
        print("4. Quit")
        choice = input("Choose: ").strip()
        try:
            if choice == "1":
                collect_dataset(ask_dataset_name(new=True))
            elif choice == "2":
                train_dataset(ask_dataset_name())
            elif choice == "3":
                recognize(ask_dataset_name())
            elif choice == "4":
                break
            else:
                print("Choose 1, 2, 3, or 4")
        except (FileNotFoundError, FileExistsError, ValueError) as error:
            print(error)


if __name__ == "__main__":
    main()
