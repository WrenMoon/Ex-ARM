#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from utils.ProbeIdentifier import ProbeIdentifier


def validate(results, tolerance, trials, seed):
    from Grasp_Discovery import load_objects_from_config
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    identifier = ProbeIdentifier(results, tolerance)
    geometries = load_objects_from_config(results["configuration"])
    rng = np.random.default_rng(seed)
    fresh = {}
    errors = {}
    predictions = {}
    for name in results["objects"]:
        sim = None
        try:
            sim = GraspSimulator(results["config"]["hand_urdf"], geometries[name])
            observations = []
            for grasp in results["grasps"]:
                start = np.asarray(grasp["start_angles"], dtype=float)
                spec = GraspSpec(start, grasp["target_joint"], grasp["target_angle"], grasp["max_increment"])
                sim.move_clear(start)
                steps = int(np.ceil(abs(spec.end_angle - start[spec.joint_index]) / spec.max_increment)) + 2
                outcome = sim.execute_probe(spec, max_steps=steps)
                if outcome.termination_reason not in ("contact", "angle_limit"):
                    raise ValueError("Incomplete probe")
                returned = sim.retract_hand(start, outcome.final_angles)
                np.testing.assert_allclose(returned, 0, atol=1e-12, rtol=0)
                observations.append({"angle": float(outcome.achieved_angle[spec.joint_index]),
                                     "contact": bool(outcome.contact_detected)})
            fresh[name] = observations
            predictions[name] = identifier.identify(observations)
            print(f"{name}: {predictions[name]['status']} -> {predictions[name]['object_id']}", flush=True)
        except Exception as exc:
            errors[name] = str(exc)
        finally:
            if sim is not None:
                sim.close()
    noise_reports = []
    for amplitude in [0.0, 0.25, 0.5, 1.0, 2.0]:
        counts = dict(correct=0, wrong=0, ambiguous=0, no_match=0)
        per_object = {}
        for name, observations in fresh.items():
            local = dict(correct=0, wrong=0, ambiguous=0, no_match=0)
            for _ in range(trials):
                noisy = [{"contact": o["contact"], "angle": o["angle"] + float(rng.uniform(-amplitude, amplitude))}
                         for o in observations]
                prediction = identifier.identify(noisy)
                category = ("correct" if prediction["object_id"] == name else "wrong") if prediction["status"] == "identified" else prediction["status"]
                counts[category] += 1
                local[category] += 1
            per_object[name] = local
        noise_reports.append({"uniform_noise_bound_deg": amplitude, "counts": counts,
                              "per_object_counts": per_object})
    correct = sum(p["object_id"] == name for name, p in predictions.items())
    return {"complete": not errors and correct == len(results["objects"]),
            "object_count": len(results["objects"]), "fresh_correct": correct,
            "tolerance_deg": tolerance, "trials_per_object": trials, "seed": seed,
            "errors": errors, "fresh_observations": fresh, "predictions": predictions,
            "noise_reports": noise_reports,
            "limitations": ["Same geometry and calibration as references; not held-out physical validation",
                            "Noise is independent bounded uniform angle noise; contact flags are unchanged",
                            "Unknown objects can match known signatures; no universal novelty guarantee",
                            "Baseline self-overlaps remain tolerated; no hardware safety certification"]}


def main():
    parser = argparse.ArgumentParser(description="Identify objects from ordered probe observations or validate in simulation")
    parser.add_argument("--results", type=Path, default=Path("/tmp/leap-fixed-library-grasps.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observations", type=Path, help='JSON: {"observations": [{"angle": 42.0, "contact": true}, ...]}')
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Maximum per-probe angle error in degrees")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/leap-identification-results.json"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("Trials must be positive")
    if args.output.resolve() in {args.results.resolve(), args.observations.resolve() if args.observations else None}:
        parser.error("Output must not overwrite inputs")
    results = json.loads(args.results.read_text())
    identifier = ProbeIdentifier(results, args.tolerance)
    report = validate(results, args.tolerance, args.trials, args.seed) if args.validate else identifier.identify(json.loads(args.observations.read_text())["observations"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    if args.validate:
        print(f"Fresh simulation identification: {report['fresh_correct']}/{report['object_count']}; complete={report['complete']}")
        for row in report["noise_reports"]:
            print(f"Noise +/-{row['uniform_noise_bound_deg']} degrees: {row['counts']}")
        if report["errors"]:
            print("Errors:", report["errors"])
    else:
        print(json.dumps(report, indent=2))
    print(f"Results written to {args.output}")
    return 0 if (report.get("complete") if args.validate else report["status"] == "identified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
