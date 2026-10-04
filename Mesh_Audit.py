#!/usr/bin/env python3
"""Quantify STL concavity; this is not a physical-clearance test."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def inspect_mesh(path):
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"Not a binary STL: {path}")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Only standard binary STL supported: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84)["vertices"].astype(float)
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite mesh vertices")
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, edge_index, edge_count = np.unique(np.sort(directed, axis=1), axis=0,
                                             return_inverse=True, return_counts=True)
    orientation = np.bincount(edge_index, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
    closed_oriented = bool(np.all(edge_count == 2) and np.all(orientation == 0))
    hull = ConvexHull(vertices)
    interior_depth = np.concatenate([
        -np.max(np.einsum("ij,kj->ik", chunk, hull.equations[:, :3]) + hull.equations[:, 3], axis=1)
        for chunk in np.array_split(vertices, max(1, int(np.ceil(len(vertices) / 256))))])
    centered = triangles - vertices.mean(axis=0)
    volume = abs(float(np.einsum("ij,ij->i", centered[:, 0],
                                 np.cross(centered[:, 1], centered[:, 2])).sum() / 6))
    return {"mesh": path.name, "triangle_count": count, "vertex_count": len(vertices),
            "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "closed_consistently_oriented_edges": closed_oriented,
            "non_two_face_edges": int(np.count_nonzero(edge_count != 2)),
            "inconsistent_orientation_edges": int(np.count_nonzero(orientation)),
            "surface_vertices_inside_hull_by_over_0_1mm": int(np.count_nonzero(interior_depth > 0.0001)),
            "maximum_surface_vertex_hull_depth_mm": float(max(0, interior_depth.max()) * 1000),
            "signed_surface_volume_m3": volume, "convex_hull_volume_m3": float(hull.volume),
            "surface_to_hull_volume_ratio": volume / hull.volume if closed_oriented else None,
            "caution": "Closed edges do not prove a non-self-intersecting solid; volume ratios are diagnostics only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/leap-mesh-audit.json"))
    args = parser.parse_args()
    paths = sorted(args.mesh_dir.glob("*.stl"))
    if not paths:
        parser.error("No STL meshes found")
    if args.output.resolve() in {p.resolve() for p in paths}:
        parser.error("Output must not overwrite mesh inputs")
    meshes = [inspect_mesh(path) for path in paths]
    report = {"meshes": meshes, "limitations": [
        "MuJoCo mesh collision uses convex hulls, not the concave STL surface",
        "Concavity can produce false-positive overlap; these metrics do not establish physical clearance",
        "Assembly transforms and actual pair intersections are not measured here"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for mesh in meshes:
        ratio = mesh["surface_to_hull_volume_ratio"]
        print(f"{mesh['mesh']}: closed/oriented={mesh['closed_consistently_oriented_edges']}; "
              f"STL/hull volume={ratio:.3f}" if ratio is not None else
              f"{mesh['mesh']}: mesh topology prevents interpreting its volume ratio")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
