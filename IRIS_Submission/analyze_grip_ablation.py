"""Exploratory nearest-trial analysis of the saved physical trial angles.

This is a descriptive reuse of the same 75 or 100 trials, not a new
independent test set. The original nearest-trial method uses all 48 angles.
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "Data" / "Proprioception" / "Physical_Only"
SETS = ["Common Household Objects", "Household Objects_Extended"]
GRIPS = {
    "grip_1": [0],
    "grip_2": [1],
    "grip_3": [2],
    "grips_1_2": [0, 1],
    "grips_1_3": [0, 2],
    "grips_2_3": [1, 2],
    "all_three_grips": [0, 1, 2],
}


def analyze(dataset):
    saved = json.loads((ROOT / dataset / "model.json").read_text())
    samples = saved["samples"]
    angles = np.asarray([row["angles_deg"] for row in samples], dtype=float)
    labels = np.asarray([row["class_name"] for row in samples])
    assert angles.shape == (len(samples), 48)
    results = {}
    for name, grips in GRIPS.items():
        columns = [joint for grip in grips for joint in range(16 * grip, 16 * grip + 16)]
        features = angles[:, columns]
        distance = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        closest = np.argmin(distance, axis=1)
        correct = int(np.sum(labels[closest] == labels))
        results[name] = {"correct": correct, "total": len(samples),
                         "accuracy": correct / len(samples)}
    assert results["all_three_grips"]["correct"] == saved["leave_one_out_correct"]
    return results


if __name__ == "__main__":
    output = {name: analyze(name) for name in SETS}
    out_path = Path(__file__).with_name("grip_ablation_results.json")
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
