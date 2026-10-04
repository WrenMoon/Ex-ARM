#!/usr/bin/env python3
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List

from datetime import datetime, timezone
from utils.GeometryLoader import GeometryLoader, PrimitiveGeometry, load_step_folder
from utils.GraspSimulator import GraspSimulator, GraspSpec, ProbeResult


def create_demo_grasp_spec(
    start_angles: np.ndarray, joint_index: int, end_angle: float
) -> GraspSpec:
    """Create a grasp specification for testing."""
    return GraspSpec(
        start_angles=start_angles,
        joint_index=joint_index,
        end_angle=end_angle,
        max_increment=0.5,
    )


def probe_object(
    hand_urdf: str,
    object_geom: PrimitiveGeometry,
    grasp_specs: List[GraspSpec],
    enable_viewer: bool = False,
) -> Dict:
    """
    Execute all grasps on a single object and return results.

    Parameters
    ----------
    hand_urdf : str
        Path to hand URDF
    object_geom : PrimitiveGeometry
        Object to probe
    grasp_specs : list of GraspSpec
        List of grasp specifications to try
    enable_viewer : bool
        Enable visualization

    Returns
    -------
    dict
        Results for the object with metadata and probe results
    """
    object_id = object_geom.metadata.get("object_id", "unknown")
    print(f"\nProbing {object_id} ({object_geom.shape}, scale={object_geom.scale:.4f}m)")

    sim = GraspSimulator(
        hand_urdf_path=hand_urdf,
        object_geom=object_geom,
        object_placement=object_geom.translation,
        enable_viewer=enable_viewer,
    )

    results = {
        "object_id": object_id,
        "shape": object_geom.shape,
        "scale_m": object_geom.scale,
        "metadata": object_geom.metadata,
        "probe_results": [],
    }

    for i, grasp_spec in enumerate(grasp_specs):
        try:
            sim.move_clear(grasp_spec.start_angles)
            probe_result = sim.execute_probe(grasp_spec, max_steps=1000)
            returned_angles = sim.retract_hand(grasp_spec.start_angles, probe_result.final_angles)

            result_dict = {
                "joint_status": probe_result.joint_status,
                "returned_angles": returned_angles.tolist(),
                "baseline_self_collision_pairs": sorted(sim.baseline_self_pairs),
                "grasp_index": i,
                "joint_index": grasp_spec.joint_index,
                "contact_detected": probe_result.contact_detected,
                "termination_reason": probe_result.termination_reason,
                "achieved_angle": probe_result.achieved_angle.tolist(),
                "contact_events": [
                    {
                        "step": ce.step,
                        "time": ce.time,
                        "joint_index": ce.joint_index,
                        "angles": ce.angles.tolist(),
                    }
                    for ce in probe_result.contact_events
                ],
            }
            results["probe_results"].append(result_dict)

            status = "CONTACT" if probe_result.contact_detected else "NO_CONTACT"
            print(
                f"  Grasp {i}: joint {grasp_spec.joint_index} -> {status} "
                f"(reason: {probe_result.termination_reason})"
            )
        except Exception as e:
            print(f"  Grasp {i}: ERROR - {e}")
            results["probe_results"].append({
                "grasp_index": i,
                "joint_index": grasp_spec.joint_index,
                "error": str(e),
            })

    sim.close()
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Simulate grasping and probing motions")
    parser.add_argument(
        "--hand-urdf",
        default="Data/mujoco_robot.urdf",
        help="Path to hand URDF file",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["cube_20mm", "sphere_10mm"],
        help="Object names from test suite",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable MuJoCo viewer",
    )
    parser.add_argument(
        "--output",
        default="probe_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--custom-object",
        choices=["cube", "sphere", "cylinder"],
        help="Create custom object instead of test suite",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.020,
        help="Scale in meters for custom object",
    )

    parser.add_argument("--object-folder", type=Path)
    parser.add_argument("--config", type=Path, help="Object metadata, shared mount and grasps JSON")
    args = parser.parse_args()
    if bool(args.object_folder) != bool(args.config):
        parser.error("--object-folder and --config must be supplied together")
    config = {}
    if args.config:
        with args.config.open() as handle:
            config = json.load(handle)

    hand_urdf = Path(args.hand_urdf)
    if not hand_urdf.exists():
        print(f"Error: URDF file not found: {hand_urdf}")
        return 1

    objects_to_probe = []

    if args.object_folder:
        objects_to_probe = list(load_step_folder(args.object_folder, config))
    elif args.custom_object:
        geom = GeometryLoader.create_primitive(
            shape=args.custom_object,
            scale=args.scale,
            metadata={"object_id": f"{args.custom_object}_{args.scale*1000:.0f}mm"},
        )
        objects_to_probe.append(geom)
    else:
        test_suite = GeometryLoader.generate_test_suite()
        for obj_name in args.objects:
            if obj_name in test_suite:
                objects_to_probe.append(test_suite[obj_name])
            else:
                parser.error(f"Unknown object: {obj_name}")

    if not objects_to_probe:
        print("No objects to probe")
        return 1

    if args.object_folder:
        if not config.get("grasps"):
            parser.error("STEP configuration must include explicit grasps")
        grasp_specs = [GraspSpec(np.asarray(g["start_angles"], dtype=float),
                                max_increment=g.get("max_increment", 0.5),
                                targets={int(j): a for j, a in g["targets"].items()})
                       for g in config["grasps"]]
    else:
        for geometry in objects_to_probe:
            geometry.translation = np.array([0.095, 0.008, -0.05])
        grasp_specs = [create_demo_grasp_spec(np.zeros(16), 2, 80.0)]

    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "mode": "geometry_only_simulation",
        "configuration": config,
        "hand_urdf": str(hand_urdf),
        "probe_count": 0,
        "objects": [],
    }

    for obj_geom in objects_to_probe:
        try:
            obj_results = probe_object(
                str(hand_urdf),
                obj_geom,
                grasp_specs,
                enable_viewer=args.viewer,
            )
            all_results["objects"].append(obj_results)
            all_results["probe_count"] += len(obj_results["probe_results"])
        except Exception as e:
            all_results["errors"].append(str(e))
            print(f"Error probing {obj_geom.metadata.get('object_id', 'unknown')}: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Total probes: {all_results['probe_count']}")

    failed = bool(all_results["errors"]) or any(
        "error" in probe or probe.get("termination_reason") == "max_steps"
        for obj in all_results["objects"] for probe in obj["probe_results"])
    return int(failed)


if __name__ == "__main__":
    exit(main())
