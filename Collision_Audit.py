#!/usr/bin/env python3
"""Measure tolerated baseline penetrations without changing collision policy."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from Grasp_Discovery import load_objects_from_config
from Grasp_Viewer import RecordedSimulator, record_probe


def audit(results):
    geometries = load_objects_from_config(results["configuration"])
    pairs = {}
    errors = {}
    paths = 0
    frames_checked = 0
    for name in results["objects"]:
        sim = None
        try:
            sim = RecordedSimulator(results["config"]["hand_urdf"], geometries[name])
            baseline = {}
            for contact in sim.data.contact:
                pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                if pair in sim.baseline_self_pairs:
                    baseline[pair] = min(baseline.get(pair, 0), float(contact.dist))
            for pair in sorted(sim.baseline_self_pairs):
                key = ":".join(map(str, pair))
                bodies = [int(sim.model.geom_bodyid[g]) for g in pair]
                parents = [int(sim.model.body_parentid[b]) for b in bodies]
                pairs.setdefault(key, {
                    "geom_ids": list(pair),
                    "geom_names": [sim.model.geom(g).name for g in pair],
                    "body_names": [sim.model.body(b).name for b in bodies],
                    "mesh_names": [sim.model.mesh(int(sim.model.geom_dataid[g])).name
                                   if sim.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
                                   for g in pair],
                    "direct_parent_child": parents[0] == bodies[1] or parents[1] == bodies[0],
                    "baseline_penetration_mm": -1000 * baseline.get(pair, 0),
                    "max_penetration_mm": 0, "worst_pose": None,
                })
            for index, grasp in enumerate(results["grasps"]):
                frames, _ = record_probe(sim, grasp, name)
                for phase, angles in frames:
                    sim.data.qpos[sim.qpos_ids] = np.radians(angles)
                    mujoco.mj_forward(sim.model, sim.data)
                    frames_checked += 1
                    for contact in sim.data.contact:
                        pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                        if pair not in sim.baseline_self_pairs:
                            continue
                        entry = pairs[":".join(map(str, pair))]
                        depth = max(0, -1000 * float(contact.dist))
                        if depth > entry["max_penetration_mm"]:
                            entry["max_penetration_mm"] = depth
                            entry["worst_pose"] = {"object": name, "probe": index + 1,
                                                   "phase": phase, "angles_deg": angles.tolist()}
                paths += 1
        except Exception as exc:
            errors[name] = str(exc)
        finally:
            if sim is not None:
                sim.close()
    return {"paths_checked": paths, "frames_checked": frames_checked, "errors": errors,
            "baseline_pairs": list(pairs.values()),
            "limitations": ["MuJoCo contact distances for collision meshes, not physical clearance measurements",
                            "Baseline pairs remain tolerated; this audit does not certify safety",
                            "Recorded frames include contact-refinement trial poses"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("/tmp/leap-fixed-library-grasps.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/leap-collision-audit.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.results.resolve():
        parser.error("Output must not overwrite results")
    report = audit(json.loads(args.results.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"Checked {report['paths_checked']} paths, {report['frames_checked']} frames; errors={len(report['errors'])}")
    for pair in report["baseline_pairs"]:
        print(f"{pair['body_names']}: adjacent={pair['direct_parent_child']}; "
              f"baseline={pair['baseline_penetration_mm']:.3f} mm; maximum={pair['max_penetration_mm']:.3f} mm")
    print(f"Report: {args.output}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
