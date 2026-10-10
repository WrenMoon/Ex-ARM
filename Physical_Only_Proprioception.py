"""Train and run a class-only model using grasps from the physical hand."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile
import time

from Grasp import grasp
from utils.Constants import Connection, PhysicalOnly, Proprioception


def dataset_folder(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", name):
        raise ValueError("Use letters, numbers, spaces, underscores, or hyphens for names")
    return Path(PhysicalOnly.data_folder) / name


def read_angles(path):
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        expected = [f"joint_{joint}" for joint in range(16 * len(Proprioception.grip))]
        if reader.fieldnames != expected:
            raise ValueError(f"Expected {len(expected)} joint columns in {path}")
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
    count = 16 * len(Proprioception.grip)
    if len(angles) != count or not all(math.isfinite(angle) for angle in angles):
        raise ValueError(f"Expected {count} finite joint angles")
    if not samples:
        raise ValueError("The model has no training trials")
    nearest = min(samples, key=lambda sample: distance(angles, sample["angles_deg"]))
    return dict(class_name=nearest["class_name"],
                nearest_trial=nearest["trial"],
                distance_deg=distance(angles, nearest["angles_deg"]))


def read_collection(name):
    path = dataset_folder(name) / "collection.json"
    plan = json.loads(path.read_text())
    if plan.get("format_version") not in (1, 2):
        raise ValueError("Unsupported collection format")
    classes = plan.get("classes")
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Invalid classes in collection.json")
    for class_name in classes:
        dataset_folder(class_name)
    if plan["format_version"] == 1:
        count = plan.get("trials_per_class")
        plan["trial_counts"] = {class_name: count for class_name in classes}
    counts = plan.get("trial_counts")
    if not isinstance(counts, dict) or set(counts) != set(classes):
        raise ValueError("Invalid class trial counts in collection.json")
    if any(type(count) is not int or count < 1 for count in counts.values()):
        raise ValueError("Every class needs at least one trial")
    if plan["grip"] != Proprioception.grip:
        raise ValueError("The grasp changed. Start a new physical dataset")
    if plan["motor_ids"] != Connection.ids or plan["motor_offsets_deg"] != Connection.offsets:
        raise ValueError("Motor IDs or angle offsets changed. Start a new physical dataset")
    return plan


def write_collection(name, plan):
    folder = dataset_folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    saved = dict(format_version=2, classes=plan["classes"],
                 trial_counts=plan["trial_counts"], grip=plan["grip"],
                 motor_ids=plan["motor_ids"],
                 motor_offsets_deg=plan["motor_offsets_deg"])
    temporary = folder / "collection.json.tmp"
    temporary.write_text(json.dumps(saved, indent=2) + "\n")
    temporary.replace(folder / "collection.json")


def trial_complete(folder):
    try:
        angles = read_angles(folder / "grasp_results.csv")
        with open(folder / "log.csv", newline="") as file:
            rows = list(csv.DictReader(file))
        last_by_grip = {}
        for row in rows:
            grip_number = int(row["grip"])
            if grip_number not in range(1, len(Proprioception.grip) + 1):
                return False
            last_by_grip[grip_number] = json.loads(row["positions"])
        if set(last_by_grip) != set(range(1, len(Proprioception.grip) + 1)):
            return False
        logged_angles = [angle for grip_number in sorted(last_by_grip)
                         for angle in last_by_grip[grip_number]]
        return len(logged_angles) == len(angles) and all(
            abs(first - float(second)) < 0.001 for first, second in zip(angles, logged_angles))
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return False


def load_samples(name):
    folder = dataset_folder(name)
    trials_folder = folder / "trials"
    if not trials_folder.is_dir():
        raise FileNotFoundError(f"No trials in {trials_folder}")

    if (folder / "collection.json").exists():
        plan = read_collection(name)
        classes = plan["classes"]
        trial_counts = plan["trial_counts"]
    else:
        # Datasets made before collection.json used three trials per class.
        classes = sorted(path.name for path in trials_folder.iterdir() if path.is_dir())
        trial_counts = {class_name: PhysicalOnly.default_trials_per_class
                        for class_name in classes}
    if len(classes) < 2:
        raise ValueError("At least two classes are needed")

    samples = []
    recorded_classes = {path.name for path in trials_folder.iterdir() if path.is_dir()}
    if recorded_classes != set(classes):
        raise ValueError("Saved class folders do not match the collection plan")
    for class_name in classes:
        class_folder = trials_folder / class_name
        count = trial_counts[class_name]
        expected_trials = {str(trial) for trial in range(1, count + 1)}
        recorded_trials = {path.name for path in class_folder.iterdir()
                           if path.is_dir() and not path.name.startswith(".")}
        if recorded_trials != expected_trials:
            raise ValueError(f"{class_name} needs trials 1 through {count}")
        for trial in range(1, count + 1):
            trial_folder = class_folder / str(trial)
            if not trial_complete(trial_folder):
                raise ValueError(f"Incomplete grasp files in {trial_folder}")
            samples.append(dict(class_name=class_name, trial=str(trial),
                                angles_deg=read_angles(trial_folder / "grasp_results.csv")))
    return samples, classes, trial_counts


def train_dataset(name):
    samples, classes, trial_counts = load_samples(name)

    correct = None
    if min(trial_counts.values()) > 1:
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
                 trial_counts=trial_counts,
                 classes=classes,
                 samples=samples,
                 leave_one_out_correct=correct,
                 leave_one_out_total=len(samples) if correct is not None else 0)
    folder = dataset_folder(name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    print(f"Saved {len(samples)} physical trials to {folder / 'model.json'}")
    if correct is None:
        print("Leave-one-trial-out requires at least two trials per class")
    else:
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
    if (dataset_folder(name) / "collection.json").exists():
        plan = read_collection(name)
        saved_counts = model.get("trial_counts")
        if saved_counts is None and type(model.get("trials_per_class")) is int:
            saved_counts = {class_name: model["trials_per_class"]
                            for class_name in model["classes"]}
        if model.get("classes") != plan["classes"] or saved_counts != plan["trial_counts"]:
            raise ValueError("The physical trials changed. Finish collection and retrain")
    return model


def open_hand():
    from utils.ExARM import ExArm

    return ExArm(mode="real", ids=Connection.ids, port=Connection.Port,
                 baudrate=Connection.baudrate, offsets=Connection.offsets,
                 model_path=Connection.model_path)


def save_grasp(folder):
    folder.parent.mkdir(parents=True, exist_ok=True)
    if folder.exists():
        raise FileExistsError(f"Trial folder already exists: {folder}")
    with tempfile.TemporaryDirectory(prefix=f".{folder.name}_", dir=folder.parent) as temporary:
        temporary_folder = Path(temporary)
        shutil.copy2(Proprioception.grasp_result_path, temporary_folder / "grasp_results.csv")
        shutil.copy2(Proprioception.grasp_log_path, temporary_folder / "log.csv")
        if not trial_complete(temporary_folder):
            raise ValueError("The grasp files are incomplete; trial was not saved")
        temporary_folder.rename(folder)


def release_hand(hand):
    hand.set_goal_positions_degree(Proprioception.grip[0]["start_angles"])
    time.sleep(Proprioception.settle_time_s)


def create_collection(name, classes, trial_counts):
    folder = dataset_folder(name)
    if (folder / "collection.json").exists() or (folder / "model.json").exists():
        raise FileExistsError(f"Dataset already exists: {name}")
    classes = [class_name.strip().lower() for class_name in classes]
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Enter at least two different classes")
    for class_name in classes:
        dataset_folder(class_name)
    if isinstance(trial_counts, int):
        trial_counts = {class_name: trial_counts for class_name in classes}
    if set(trial_counts) != set(classes) or any(type(count) is not int or count < 1
                                                for count in trial_counts.values()):
        raise ValueError("Every class needs at least one trial")
    if (folder / "trials").exists():
        raise FileExistsError("This dataset already contains trials")
    plan = dict(classes=classes, trial_counts=trial_counts,
                grip=Proprioception.grip, motor_ids=Connection.ids,
                motor_offsets_deg=Connection.offsets)
    write_collection(name, plan)
    return plan


def ensure_collection_plan(name):
    if (dataset_folder(name) / "collection.json").exists():
        return read_collection(name)
    _, classes, counts = load_samples(name)
    if (dataset_folder(name) / "model.json").exists():
        load_model(name)
    plan = dict(classes=classes, trial_counts=counts,
                grip=Proprioception.grip, motor_ids=Connection.ids,
                motor_offsets_deg=Connection.offsets)
    write_collection(name, plan)
    return plan


def pending_trials(name):
    plan = read_collection(name)
    folder = dataset_folder(name)
    pending = []
    for class_name in plan["classes"]:
        for trial in range(1, plan["trial_counts"][class_name] + 1):
            trial_folder = folder / "trials" / class_name / str(trial)
            if not trial_complete(trial_folder):
                pending.append((class_name, trial))
    return pending


def prepare_trial(name, class_name, trial):
    folder = dataset_folder(name) / "trials" / class_name / str(trial)
    if folder.exists() and not trial_complete(folder):
        incomplete = dataset_folder(name) / "incomplete" / class_name
        incomplete.mkdir(parents=True, exist_ok=True)
        saved = incomplete / f"{trial}_{time.time_ns()}"
        folder.rename(saved)
        print(f"Moved an incomplete trial to {saved}")
    return folder


def collect_one_trial(name, on_angles=None, on_grip=None):
    pending = pending_trials(name)
    if not pending:
        return None
    class_name, trial = pending[0]
    folder = prepare_trial(name, class_name, trial)
    hand = open_hand()
    try:
        try:
            angles = grasp(hand, on_angles=on_angles, on_grip=on_grip)
            save_grasp(folder)
        finally:
            release_hand(hand)
            if on_angles is not None:
                on_angles(Proprioception.grip[0]["start_angles"])
            if on_grip is not None:
                on_grip(1)
    finally:
        hand.close()
    return dict(class_name=class_name, trial=trial, angles_deg=angles,
                remaining=len(pending_trials(name)))


def expand_collection(name, extra_for_all=0, extra_by_class=None, new_classes=None):
    plan = ensure_collection_plan(name)
    if pending_trials(name):
        raise ValueError("Finish the current collection before adding trials")
    extra_by_class = extra_by_class or {}
    new_classes = new_classes or {}
    if type(extra_for_all) is not int or extra_for_all < 0:
        raise ValueError("Additional grasp count must be zero or greater")
    classes = plan["classes"].copy()
    counts = plan["trial_counts"].copy()
    for class_name, extra in extra_by_class.items():
        if class_name not in counts or type(extra) is not int or extra < 0:
            raise ValueError(f"Invalid additional grasp count for {class_name}")
    for class_name, count in new_classes.items():
        dataset_folder(class_name)
        if class_name in counts or type(count) is not int or count < 1:
            raise ValueError(f"Invalid new class or grasp count: {class_name}")
    for class_name in classes:
        counts[class_name] += extra_for_all + extra_by_class.get(class_name, 0)
    for class_name, count in new_classes.items():
        classes.append(class_name)
        counts[class_name] = count
    if classes == plan["classes"] and counts == plan["trial_counts"]:
        return None
    plan["classes"] = classes
    plan["trial_counts"] = counts
    write_collection(name, plan)
    return plan


def collect_dataset(name):
    folder = dataset_folder(name)
    plan_path = folder / "collection.json"
    if plan_path.exists():
        plan = read_collection(name)
        print(f"Resuming {name}: {plan['trial_counts']}")
    else:
        if (folder / "model.json").exists():
            raise ValueError("This older dataset is already trained. Choose a new dataset name")
        count = int(input("How many object classes? "))
        if count < 2:
            raise ValueError("Enter at least two classes")
        requested = input(f"Grasps per class [{PhysicalOnly.default_trials_per_class}]: ").strip()
        trials_per_class = int(requested) if requested else PhysicalOnly.default_trials_per_class
        if trials_per_class < 1:
            raise ValueError("Enter at least one grasp per class")
        classes = []
        for index in range(count):
            class_name = input(f"Name for class {index + 1}: ").strip().lower()
            dataset_folder(class_name)
            if class_name in classes:
                raise ValueError("Class names must be different")
            classes.append(class_name)
        existing = folder / "trials"
        if existing.is_dir():
            existing_classes = {path.name for path in existing.iterdir() if path.is_dir()}
            if not existing_classes.issubset(set(classes)):
                raise ValueError("Existing trial classes do not match the entered names")
            for class_folder in existing.iterdir():
                if not class_folder.is_dir():
                    continue
                for trial_folder in class_folder.iterdir():
                    if not trial_folder.is_dir() or trial_folder.name.startswith("."):
                        continue
                    if not trial_folder.name.isdigit() or int(trial_folder.name) > trials_per_class:
                        raise ValueError("Existing trials exceed the entered number per class")
        plan = dict(classes=classes,
                    trial_counts={class_name: trials_per_class for class_name in classes},
                    grip=Proprioception.grip,
                    motor_ids=Connection.ids,
                    motor_offsets_deg=Connection.offsets)
        write_collection(name, plan)
        print(f"Collection plan saved to {plan_path}")

    classes = plan["classes"]
    trial_counts = plan["trial_counts"]
    pending = pending_trials(name)

    total = sum(trial_counts.values())
    completed = total - len(pending)
    print(f"Completed {completed}/{total} grasps")
    if not pending:
        return train_dataset(name)

    print("Place each object at the same point and press Enter when ready.")
    hand = open_hand()
    try:
        for class_name, trial in pending:
            input(f"Place {class_name}, trial {trial}, then press Enter: ")
            trial_folder = prepare_trial(name, class_name, trial)
            try:
                grasp(hand)
                save_grasp(trial_folder)
            finally:
                release_hand(hand)
            print("Trial saved and hand opened. Remove the object before the next trial.")
            while input("Press Enter to continue, or R to retake this trial: ").strip().lower() == "r":
                print(f"Retaking {class_name}, trial {trial}")
                retake_trial(name, class_name, trial, hand=hand)
    finally:
        hand.close()
    return train_dataset(name)


def retake_trial(name, class_name, trial, hand=None):
    plan = read_collection(name)
    class_name = class_name.strip().lower()
    if class_name not in plan["trial_counts"]:
        raise ValueError(f"Choose one of: {', '.join(plan['classes'])}")
    if trial < 1 or trial > plan["trial_counts"][class_name]:
        raise ValueError(f"Choose a trial from 1 to {plan['trial_counts'][class_name]}")
    folder = dataset_folder(name) / "trials" / class_name / str(trial)
    if not trial_complete(folder):
        raise ValueError(f"Trial {trial} for {class_name} is incomplete; use collection to resume it")

    # Record and check the replacement before moving the original trial.
    archive = dataset_folder(name) / "retaken" / class_name
    archive.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retake_", dir=archive) as temporary:
        replacement = Path(temporary) / "replacement"
        own_hand = hand is None
        if own_hand:
            hand = open_hand()
        try:
            try:
                grasp(hand)
                save_grasp(replacement)
            finally:
                release_hand(hand)
        finally:
            if own_hand:
                hand.close()
        saved = archive / f"{trial}_{time.time_ns()}"
        folder.rename(saved)
        try:
            replacement.rename(folder)
        except OSError:
            saved.rename(folder)
            raise
    print(f"Retaken trial {trial} for {class_name}; previous trial saved at {saved}")
    if own_hand and not pending_trials(name):
        train_dataset(name)
    print("Retrain the neural model with option 6 before using it again")
    return folder


def extend_dataset(name):
    if not (dataset_folder(name) / "collection.json").exists():
        ensure_collection_plan(name)
        print("Saved a collection plan for this older dataset")
    plan = read_collection(name)
    load_samples(name)  # Finish any interrupted collection before extending it.
    classes = plan["classes"]
    counts = plan["trial_counts"]
    original_total = sum(counts.values())

    response = input("Additional grasps for every current class [0]: ").strip()
    extra_for_all = int(response) if response else 0
    if extra_for_all < 0:
        raise ValueError("The additional grasp count cannot be negative")
    for class_name in classes:
        counts[class_name] += extra_for_all

    print("Enter a class name to add more grasps to just that class.")
    while True:
        class_name = input("Existing class name [Enter to continue]: ").strip().lower()
        if not class_name:
            break
        if class_name not in counts:
            print(f"Choose one of: {', '.join(classes)}")
            continue
        extra = int(input(f"Additional grasps for {class_name}: "))
        if extra < 1:
            raise ValueError("Enter at least one additional grasp")
        counts[class_name] += extra

    response = input("How many new classes? [0]: ").strip()
    new_classes = int(response) if response else 0
    if new_classes < 0:
        raise ValueError("The new class count cannot be negative")
    for index in range(new_classes):
        class_name = input(f"Name for new class {index + 1}: ").strip().lower()
        dataset_folder(class_name)
        if class_name in counts:
            raise ValueError(f"Class {class_name} already exists")
        response = input(f"Grasps for {class_name} [{PhysicalOnly.default_trials_per_class}]: ").strip()
        count = int(response) if response else PhysicalOnly.default_trials_per_class
        if count < 1:
            raise ValueError("Enter at least one grasp for the new class")
        classes.append(class_name)
        counts[class_name] = count

    if sum(counts.values()) == original_total:
        print("No new trials requested")
        return None
    write_collection(name, plan)
    print(f"Expanded collection plan saved for {name}")
    return collect_dataset(name)


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


def recognize_once(name, on_angles=None, on_grip=None):
    model = load_model(name)
    hand = open_hand()
    try:
        try:
            angles = grasp(hand, on_angles=on_angles, on_grip=on_grip)
            result = predict(angles, model["samples"])
            timestamp = datetime.now(timezone.utc)
            run_folder = dataset_folder(name) / "runs" / timestamp.strftime("%Y%m%d_%H%M%S_%f")
            save_grasp(run_folder)
        finally:
            release_hand(hand)
            if on_angles is not None:
                on_angles(Proprioception.grip[0]["start_angles"])
            if on_grip is not None:
                on_grip(1)
    finally:
        hand.close()
    output = dict(time_utc=timestamp.isoformat(), dataset=name,
                  final_angles_deg=angles, prediction=result)
    (run_folder / "recognition_result.json").write_text(json.dumps(output, indent=2) + "\n")
    return output


def train_neural_dataset(name):
    from utils.PhysicalOnlyNetwork import train_network

    samples, classes, counts = load_samples(name)
    folder = dataset_folder(name)
    model_path = folder / PhysicalOnly.network_model_file
    report_path = folder / PhysicalOnly.network_report_file
    model, report = train_network(samples, classes, model_path, report_path)
    print(f"Saved neural model to {model_path}")
    print(f"Held-out physical trials: {report['validation_correct']}/"
          f"{report['validation_total']} correct")
    print(f"Class trial counts: {counts}")
    return model, report


def load_neural_model(name):
    from utils.PhysicalOnlyNetwork import PhysicalOnlyNetwork, sample_fingerprint

    samples, classes, _ = load_samples(name)
    path = dataset_folder(name) / PhysicalOnly.network_model_file
    model = PhysicalOnlyNetwork.load(path, sample_fingerprint(samples))
    if model.class_names != classes:
        raise ValueError("The physical classes changed. Retrain the neural model")
    return model


def recognize_neural(name):
    model = load_neural_model(name)
    if PhysicalOnly.network_max_grasps < 1:
        raise ValueError("Set network_max_grasps to at least one")
    timestamp = datetime.now(timezone.utc)
    run_folder = dataset_folder(name) / "neural_runs" / timestamp.strftime("%Y%m%d_%H%M%S_%f")
    results = []
    hand = open_hand()
    try:
        for number in range(1, PhysicalOnly.network_max_grasps + 1):
            if number > 1:
                release_hand(hand)
            prompt = ("Place the object, then press Enter to grasp: " if number == 1
                      else "Reposition the object and press Enter, or type q to stop: ")
            if input(prompt).strip().lower() == "q":
                break
            angles = grasp(hand)
            grasp_folder = run_folder / f"grasp_{number}"
            save_grasp(grasp_folder)
            prediction = model.predict(angles)
            results.append(dict(angles_deg=angles, prediction=prediction))

            probabilities = [sum(item["prediction"]["probabilities"][class_name]
                                 for item in results) / len(results)
                             for class_name in model.class_names]
            distance = statistics.median(item["prediction"]["distance"]
                                         for item in results)
            combined = model.evaluate(probabilities, distance)
            print(f"After {number} grasp(s): {combined['class_name']}")
            if combined["reason"]:
                print(f"Reason: {combined['reason']}")
            if combined["class_name"] in model.class_names:
                break
    finally:
        hand.close()

    if not results:
        print("No grasps recorded")
        return None
    output = dict(time_utc=timestamp.isoformat(), dataset=name,
                  grasp_count=len(results), prediction=combined, grasps=results)
    (run_folder / "recognition_result.json").write_text(json.dumps(output, indent=2) + "\n")
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {combined['class_name']}".center(60))
    print("=" * 60 + "\n")
    if combined["reason"]:
        print(f"Reason: {combined['reason']}")
    print(f"Result saved to {run_folder}")
    return output


def ask_dataset_name(new=False):
    if new:
        name = input("Collection dataset name (new or existing): ").strip()
        if not name:
            raise ValueError("Enter a name for the new dataset")
        return name
    name = input(f"Dataset name [{PhysicalOnly.default_dataset_name}]: ").strip()
    return name or PhysicalOnly.default_dataset_name


def main():
    while True:
        print("\nPhysical-only proprioception")
        print("1. Collect or resume real grasps and train")
        print("2. Train from saved physical trials")
        print("3. Identify an object with the real hand")
        print("4. Quit")
        print("5. Add trials or classes to an existing dataset")
        print("6. Train the physical-only neural model")
        print("7. Identify an object with the neural model")
        print("8. Launch presentation UI")
        print("9. Retake a saved physical trial")
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
            elif choice == "5":
                extend_dataset(ask_dataset_name())
            elif choice == "6":
                train_neural_dataset(ask_dataset_name())
            elif choice == "7":
                recognize_neural(ask_dataset_name())
            elif choice == "8":
                from Physical_Only_UI import main as launch_ui
                launch_ui()
            elif choice == "9":
                name = ask_dataset_name()
                plan = read_collection(name)
                print("Classes: " + ", ".join(
                    f"{class_name} (1-{plan['trial_counts'][class_name]})"
                    for class_name in plan["classes"]))
                class_name = input("Class to retake: ").strip().lower()
                trial = int(input("Trial number to retake: "))
                input("Place the object and press Enter to retake the trial: ")
                retake_trial(name, class_name, trial)
            else:
                print("Choose 1 through 9")
        except (FileNotFoundError, FileExistsError, ValueError, RuntimeError,
                ModuleNotFoundError) as error:
            print(error)


if __name__ == "__main__":
    main()
