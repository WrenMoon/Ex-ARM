#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

from utils.GeometryLoader import GeometryLoader, load_step_folder
from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if "hand_urdf" not in cfg:
        cfg["hand_urdf"] = "Data/mujoco_robot.urdf"
    return cfg


def load_objects_from_config(config: dict) -> dict:
    geometries = {}

    if "object_folder" in config:
        folder = Path(config["object_folder"])
        for geom in load_step_folder(folder, config):
            obj_id = geom.metadata.get("object_id", "unknown")
            if obj_id in geometries:
                raise ValueError(f"Duplicate object ID: {obj_id}")
            geometries[obj_id] = geom
        return geometries

    if "primitives" in config:
        for prim in config["primitives"]:
            shape = prim.get("shape", "cube")
            scale = prim.get("scale", 0.025)
            obj_id = prim.get("object_id", f"{shape}_{scale*1000:.0f}mm")
            translation = prim.get("translation")
            if translation:
                translation = np.array(translation)
            geom = GeometryLoader.create_primitive(
                shape=shape,
                scale=scale,
                translation=translation,
                metadata={"object_id": obj_id}
            )
            if obj_id in geometries:
                raise ValueError(f"Duplicate object ID: {obj_id}")
            geometries[obj_id] = geom
        return geometries

    return geometries


def main():
    parser = argparse.ArgumentParser(
        description="Automated grasp probe discovery"
    )
    parser.add_argument("--config", type=Path, default=Path("Data/Objects/config.json"),
                        help="Config JSON file")
    parser.add_argument("--object-folder", type=Path,
                        help="Folder with STEP files; defaults to the config directory")
    parser.add_argument("--output", type=Path, default=Path("discovered_grasps.json"),
                        help="Output JSON file (default: discovered_grasps.json)")
    parser.add_argument("--library-count", "--max-candidates", type=int, default=100,
                        dest="library_count",
                        help="Library size: number of candidates to generate (default: 100)")
    parser.add_argument("--selected-count", type=int, default=5,
                        help="Number of grasps to select from library (default: 5)")
    parser.add_argument("--budget", type=float, default=180.0,
                        help="Time budget in seconds (default: 180)")
    parser.add_argument("--min-angle-separation", type=float, default=2.0,
                        help="Minimum angle separation in degrees (default: 2.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--hand-urdf", type=str, default="Data/mujoco_robot.urdf",
                        help="Hand URDF path (default: Data/mujoco_robot.urdf)")

    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    if args.object_folder:
        config["object_folder"] = str(args.object_folder)

    config["hand_urdf"] = args.hand_urdf

    if config.get("objects") and not config.get("object_folder"):
        config["object_folder"] = str(args.config.parent)
    if args.output.resolve() == args.config.resolve():
        parser.error("Output must not overwrite the input configuration")

    geometries = load_objects_from_config(config)

    if not geometries:
        result = {
            "error": "No objects loaded",
            "config": config,
            "timestamp": time.time(),
        }
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Error: No objects loaded. Results written to {output_path}")
        return 1

    discovery_config = DiscoveryConfig(
        hand_urdf=config["hand_urdf"],
        max_candidates=args.library_count,
        selected_count=args.selected_count,
        budget_sec=args.budget,
        min_angle_separation=args.min_angle_separation,
        seed=args.seed
    )

    print(f"Starting discovery with {len(geometries)} objects, "
          f"library size {args.library_count}, selecting {args.selected_count}, {args.budget}s budget")

    discovery = ProbeDiscovery(discovery_config)
    try:
        grasps, status = discovery.discover(geometries)
    finally:
        discovery.cleanup()

    result = {
        "timestamp": time.time(),
        "config": {
            "hand_urdf": discovery_config.hand_urdf,
            "library_count": discovery_config.max_candidates,
            "selected_count": discovery_config.selected_count,
            "budget_sec": discovery_config.budget_sec,
            "min_angle_separation": discovery_config.min_angle_separation,
            "seed": discovery_config.seed,
        },
        "configuration": config,
        "objects": list(geometries.keys()),
        "object_count": len(geometries),
        "grasp_count": len(grasps),
        "library": [{"start_angles": c.start_angles.tolist(),
                     "targets": {str(c.joint_index): c.end_angle},
                     "max_increment": c.max_increment,
                     "sampled_tip_positions_m": discovery.library_sweeps[i].tolist()}
                    for i, c in enumerate(discovery.candidates)],
        "grasps": [
            {
                "start_angles": g.start_angles,
                "target_joint": g.target_joint,
                "target_angle": g.target_angle,
                "targets": {str(g.target_joint): g.target_angle},
                "max_increment": g.max_increment,
                "measurements": g.measurements,
                "per_object_angles": g.per_object_angles,
                "per_object_contact": g.per_object_contact,
            }
            for g in grasps
        ],
        "status": status,
    }

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSearch finished: {len(grasps)} selected grasps (from {status.get('valid_count', 0)} valid out of {status.get('library_size', 0)} library)")
    print(f"Complete: {status.get('complete', False)}; reason: {status.get('reason')}; "
          f"evaluated: {status.get('tried', 0)}/{status.get('library_size', 0)}; "
          f"unresolved pairs: {len(status.get('unresolved_pairs', []))}; "
          f"no contact objects: {len(status.get('no_contact_objects', []))}")
    print(f"Results written to {output_path.absolute()}")

    return 0 if status.get("complete", False) else 1


if __name__ == "__main__":
    sys.exit(main())
