#!/usr/bin/env python3
"""Compare baseline body-pair penetration at previously audited worst-case poses."""
import argparse
import json
from pathlib import Path

from Grasp_Discovery import load_objects_from_config
from utils.GraspSimulator import GraspSimulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("/tmp/leap-fixed-library-grasps.json"))
    parser.add_argument("--audit", type=Path, default=Path("/tmp/leap-collision-audit.json"))
    parser.add_argument("--experimental", type=Path, default=Path("/tmp/leap-decomposed-hand/hand_decomposed.urdf"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/leap-decomposition-comparison.json"))
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in (args.results, args.audit, args.experimental)}:
        parser.error("Output must not overwrite inputs")
    results = json.loads(args.results.read_text())
    audit = json.loads(args.audit.read_text())
    geometries = load_objects_from_config(results["configuration"])
    rows = []
    for entry in audit["baseline_pairs"]:
        pose = entry["worst_pose"]
        row = {"body_names": entry["body_names"], "pose": pose, "models": {}}
        for label, path in [("original", results["config"]["hand_urdf"]),
                            ("experimental", str(args.experimental))]:
            sim = GraspSimulator(path, geometries[pose["object"]])
            try:
                sim.set_angles(pose["angles_deg"])
                depths = []
                for contact in sim.data.contact:
                    bodies = [sim.model.body(int(sim.model.geom_bodyid[g])).name
                              for g in (contact.geom1, contact.geom2)]
                    if sorted(bodies) == sorted(entry["body_names"]) and contact.dist < -1e-6:
                        depths.append(-float(contact.dist) * 1000)
                row["models"][label] = {"penetrating_contacts": len(depths),
                                         "max_penetration_mm": max(depths, default=0)}
            finally:
                sim.close()
        rows.append(row)
        print(f"{row['body_names']}: {row['models']}")
    report = {"comparisons": rows, "limitations": [
        "Only previously worst-case poses checked, not complete paths",
        "Body-pair comparisons can miss new collision pairs",
        "No physical safety or mesh fidelity certification"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
