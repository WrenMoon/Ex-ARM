#!/usr/bin/env python3
"""Build a separate approximate convex-decomposition URDF; never edits source meshes."""
import argparse
import copy
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import numpy as np


def read_stl(path):
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Invalid binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f"Unsupported STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite STL coordinates")
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return vertices, indices.reshape(-1, 3)


def build(source, output_dir, names, threshold, max_parts):
    source = source.resolve()
    output_dir = output_dir.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.SubElement(root, "mujoco")
    compiler = mujoco_element.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_element, "compiler")
    compiler.set("strippath", "false")
    available = {Path(mesh.get("filename")).stem for mesh in root.findall("./link/collision/geometry/mesh")}
    if not names <= available:
        raise ValueError(f"Unknown collision meshes: {sorted(names - available)}")
    if output_dir == source.parent or (output_dir / source.name) == source:
        raise ValueError("Choose a separate output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pieces_by_source = {}
    for mesh in root.findall(".//mesh"):
        path = Path(mesh.get("filename"))
        mesh.set("filename", str((source.parent / path).resolve()))
    coacd.set_log_level("error")
    for link in root.findall("link"):
        for collision in list(link.findall("collision")):
            mesh = collision.find("geometry/mesh")
            if mesh is None or Path(mesh.get("filename")).stem not in names:
                continue
            path = Path(mesh.get("filename"))
            if path not in pieces_by_source:
                vertices, faces = read_stl(path)
                print(f"Decomposing {path.name}...", flush=True)
                parts = coacd.run_coacd(coacd.Mesh(vertices, faces), threshold=threshold,
                                        max_convex_hull=max_parts, preprocess_mode="off",
                                        resolution=1000, mcts_nodes=10, mcts_iterations=30,
                                        mcts_max_depth=3, seed=0)
                if not parts:
                    raise ValueError(f"Empty decomposition for {path}")
                paths = []
                for index, (points, triangles) in enumerate(parts):
                    if not np.isfinite(points).all():
                        raise ValueError("Decomposition returned nonfinite coordinates")
                    if np.any(points.min(axis=0) < vertices.min(axis=0) - 1e-8) or np.any(points.max(axis=0) > vertices.max(axis=0) + 1e-8):
                        raise ValueError(f"Decomposition expands outside source bounds: {path.name}, part {index}")
                    piece = output_dir / f"{path.stem}_part_{index:03d}.obj"
                    lines = [f"v {x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in points]
                    lines += [f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in triangles]
                    piece.write_text("".join(lines))
                    paths.append(piece)
                pieces_by_source[path] = paths
                manifest.append({"source": str(path), "part_count": len(paths)})
                print(f"  {len(paths)} pieces", flush=True)
            root_index = list(link).index(collision)
            link.remove(collision)
            for index, piece in enumerate(pieces_by_source[path]):
                replacement = copy.deepcopy(collision)
                replacement.set("name", f"{link.get('name')}_convex_{index}")
                replacement.find("geometry/mesh").set("filename", str(piece))
                link.insert(root_index + index, replacement)
    output = output_dir / "hand_decomposed.urdf"
    tree.write(output, encoding="unicode")
    report = {"source_urdf": str(source), "output_urdf": str(output),
              "threshold": threshold, "max_parts": max_parts, "seed": 0, "meshes": manifest,
              "preprocess_mode": "off",
              "limitations": ["Approximate convex decomposition; not certified conservative or exact",
                              "Preprocessing is disabled; source topology defects are not repaired",
                              "Preserved coordinate bounds do not guarantee preservation of concave interfaces",
                              "Unselected collision meshes retain their original convex hull approximation",
                              "New geometry requires fresh path validation and reference measurements"]}
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Data/mujoco_robot.urdf"))
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/leap-decomposed-hand"))
    parser.add_argument("--meshes", nargs="+", default=["palm_lower"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-parts", type=int, default=16)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold <= 0 or args.max_parts < 1:
        parser.error("Threshold and maximum part count must be positive")
    print(f"Experimental model: {build(args.source, args.output_dir, set(args.meshes), args.threshold, args.max_parts)}")


if __name__ == "__main__":
    main()
