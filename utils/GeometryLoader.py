import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class PrimitiveGeometry:
    """
    Represents a primitive geometric object with scale and metadata.

    Attributes
    ----------
    shape : str
        'cube', 'sphere', or 'cylinder'
    scale : float
        Characteristic dimension (side length for cube, radius for sphere/cylinder)
    rotation : np.ndarray
        (3,3) rotation matrix in world frame (default: identity)
    translation : np.ndarray
        (3,) translation offset in world frame (default: origin)
    metadata : dict
        Custom metadata (e.g., object ID, size classification)
    """

    def __init__(
        self,
        shape: str,
        scale: float,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Initialize a primitive geometry.

        Parameters
        ----------
        shape : str
            'cube', 'sphere', or 'cylinder'
        scale : float
            Characteristic dimension in meters
        rotation : (3,3) array, optional
            Rotation matrix; defaults to identity
        translation : (3,) array, optional
            Translation vector; defaults to origin
        metadata : dict, optional
            Custom metadata
        """
        if shape not in ("cube", "sphere", "cylinder"):
            raise ValueError(f"Unknown shape: {shape}")

        self.shape = shape
        self.scale = float(scale)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)
        self.translation = np.zeros(3) if translation is None else np.asarray(translation)
        self.metadata = metadata or {}
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Object size must be finite and positive")
        if self.translation.shape != (3,) or not np.isfinite(self.translation).all():
            raise ValueError("Mount translation must contain three finite values")
        if (self.rotation.shape != (3, 3) or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3))
                or not np.isclose(np.linalg.det(self.rotation), 1)):
            raise ValueError("Mount rotation must be a proper rotation matrix")

    def add_to_spec(self, spec, placement=None):
        import mujoco

        position = self.translation if placement is None else np.asarray(placement)
        body = spec.worldbody.add_body(name="probe_object", pos=position,
                                       quat=self._rotation_to_quat(self.rotation))
        geom = self.get_mujoco_body_spec("probe_object")["geom"]
        kind = {"box": mujoco.mjtGeom.mjGEOM_BOX, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
                "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}[geom["type"]]
        body.add_geom(name="probe_object_geom", type=kind, size=geom["size"],
                      contype=1, conaffinity=1)

    def get_mujoco_body_spec(self, name: str, density: float = 1000.0) -> Dict:
        """
        Generate a MuJoCo body specification for this geometry.

        For cube: scale is side length; MuJoCo gets half-size per axis.
        For sphere: scale is radius; MuJoCo gets radius in size[0].
        For cylinder: scale is radius; height is scale (both in size).

        Parameters
        ----------
        name : str
            Name for the body in MuJoCo
        density : float
            Material density in kg/m^3

        Returns
        -------
        dict
            Specification dict with 'body' and 'geom' keys
        """
        if self.shape == "cube":
            half_size = self.scale / 2.0
            geom_size = [half_size, half_size, half_size]
            geom_type = "box"
        elif self.shape == "sphere":
            geom_size = [self.scale, 0, 0]
            geom_type = "sphere"
        elif self.shape == "cylinder":
            half_height = self.scale
            geom_size = [self.scale, half_height, 0]
            geom_type = "cylinder"

        quat = self._rotation_to_quat(self.rotation)

        return {
            "body": {
                "name": name,
                "pos": self.translation.tolist(),
                "quat": quat.tolist(),
            },
            "geom": {
                "name": f"{name}_geom",
                "type": geom_type,
                "size": geom_size,
                "density": density,
            },
        }

    @staticmethod
    def _rotation_to_quat(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion (w,x,y,z).

        Parameters
        ----------
        R : (3,3) array
            Rotation matrix

        Returns
        -------
        np.ndarray
            Quaternion [w, x, y, z]
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])


class GeometryLoader:
    """
    Load and manage geometric objects for simulation.

    Supports:
    - Primitive shapes (cube, sphere, cylinder) with scaling
    - Metadata storage (size classification, object ID, etc.)
    """

    @staticmethod
    def create_primitive(
        shape: str,
        scale: float,
        rotation: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
    ) -> PrimitiveGeometry:
        """
        Create a primitive geometric object.

        Parameters
        ----------
        shape : str
            'cube', 'sphere', or 'cylinder'
        scale : float
            Characteristic dimension in meters
        rotation : (3,3) array, optional
            Rotation matrix; defaults to identity
        translation : (3,) array, optional
            Translation vector; defaults to origin
        metadata : dict, optional
            Custom metadata

        Returns
        -------
        PrimitiveGeometry
            The created geometry
        """
        return PrimitiveGeometry(
            shape=shape,
            scale=scale,
            rotation=rotation,
            translation=translation,
            metadata=metadata,
        )

    @staticmethod
    def generate_test_suite() -> Dict[str, PrimitiveGeometry]:
        """
        Generate a test suite of primitive objects at multiple scales.

        Returns
        -------
        dict
            Maps object names to PrimitiveGeometry instances
            Format: '{shape}_{scale_mm}' -> geometry

        Objects generated:
        - cube: 20mm, 30mm, 50mm
        - sphere: 10mm, 15mm, 25mm
        - cylinder: radius 10mm/height 20mm, 15mm/30mm, 25mm/50mm
        """
        objects = {}

        cube_scales = [0.020, 0.030, 0.050]
        for scale in cube_scales:
            name = f"cube_{int(scale*1000)}mm"
            objects[name] = PrimitiveGeometry(
                shape="cube",
                scale=scale,
                metadata={"object_id": name, "size_mm": int(scale * 1000)},
            )

        sphere_scales = [0.010, 0.015, 0.025]
        for scale in sphere_scales:
            name = f"sphere_{int(scale*1000)}mm"
            objects[name] = PrimitiveGeometry(
                shape="sphere",
                scale=scale,
                metadata={"object_id": name, "size_mm": int(scale * 1000)},
            )

        cylinder_scales = [(0.010, 0.020), (0.015, 0.030), (0.025, 0.050)]
        for radius, height in cylinder_scales:
            name = f"cylinder_r{int(radius*1000)}_h{int(height*1000)}mm"
            objects[name] = PrimitiveGeometry(
                shape="cylinder",
                scale=radius,
                metadata={
                    "object_id": name,
                    "radius_mm": int(radius * 1000),
                    "height_mm": int(height * 1000),
                },
            )

        return objects

    @staticmethod
    def save_metadata(
        geometry: PrimitiveGeometry,
        output_path: Path,
    ) -> None:
        """
        Save geometry metadata to a JSON file.

        Parameters
        ----------
        geometry : PrimitiveGeometry
            Geometry to save
        output_path : Path
            Output JSON file path
        """
        data = {
            "shape": geometry.shape,
            "scale": geometry.scale,
            "metadata": geometry.metadata,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_metadata(metadata_path: Path) -> Dict:
        """
        Load geometry metadata from a JSON file.

        Parameters
        ----------
        metadata_path : Path
            Input JSON file path

        Returns
        -------
        dict
            Metadata dict with 'shape', 'scale', 'metadata' keys
        """
        with open(metadata_path, "r") as f:
            return json.load(f)


class StepGeometry(PrimitiveGeometry):
    def __init__(self, path, reference_size_m, scale_factor=1.0, rotation=None,
                 translation=None, metadata=None, allow_convex_hull=False):
        import cadquery as cq

        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("Scale factor must be finite and positive")
        if not allow_convex_hull:
            raise ValueError("STEP mesh collisions use a convex hull; explicitly enable "
                             "allow_convex_hull only for suitable objects")
        super().__init__("cube", reference_size_m * scale_factor, rotation, translation, metadata)
        self.shape = self.metadata.get("class", Path(path).stem)
        self.path = Path(path)
        self.scale_factor = scale_factor
        self.reference_size_m = reference_size_m
        imported = cq.importers.importStep(str(self.path))
        shape = cq.Compound.makeCompound(imported.vals())
        vertices, faces = shape.tessellate(0.05, 0.1)
        self.vertices = np.array([v.toTuple() for v in vertices]) * (0.001 * scale_factor)
        self.faces = np.asarray(faces, dtype=int)
        if len(self.vertices) < 4 or not np.isfinite(self.vertices).all():
            raise ValueError("STEP did not produce a valid solid mesh")
        self.metadata = dict(self.metadata, collision_representation="convex_hull",
                             reference_size_m=reference_size_m, scale_factor=scale_factor,
                             size_m=self.scale)

    def add_to_spec(self, spec, placement=None):
        import mujoco

        position = self.translation if placement is None else np.asarray(placement)
        spec.add_mesh(name="probe_mesh", uservert=self.vertices.ravel().tolist(),
                      userface=self.faces.ravel().tolist())
        body = spec.worldbody.add_body(name="probe_object", pos=position,
                                       quat=self._rotation_to_quat(self.rotation))
        body.add_geom(name="probe_object_geom", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname="probe_mesh", contype=1, conaffinity=1)


def load_step_folder(folder, config):
    folder = Path(folder).resolve()
    files = {p.name: p for p in folder.iterdir() if p.suffix.lower() in (".step", ".stp")}
    entries = {entry["file"]: entry for entry in config["objects"]}
    if not files or set(files) != set(entries):
        raise ValueError("Every STEP/STP file must have exactly one metadata entry")
    if len(entries) != len(config["objects"]):
        raise ValueError("Duplicate object metadata entries")
    rotation = np.eye(3)
    angles = np.radians(config.get("mount_rotation_rpy_deg", [0, 0, 0]))
    for axis, angle in zip(np.eye(3), angles):
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        rotation = (np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross) @ rotation
    scaling = config.get("scale_percent", {"start": 150, "stop": 50, "step": -5})
    start, stop, step = (scaling[key] for key in ("start", "stop", "step"))
    if (not np.isfinite([start, stop, step]).all() or min(start, stop) <= 0
            or step == 0 or (stop - start) * step < 0):
        raise ValueError("Invalid scale range")
    count = int(np.floor((stop - start) / step)) + 1
    if count > 10000:
        raise ValueError("Scale range exceeds 10000 samples")
    for name, path in sorted(files.items()):
        entry = entries[name]
        for index in range(count):
            factor = (start + index * step) / 100
            yield StepGeometry(path, entry["reference_size_m"], factor, rotation,
                               config.get("mount_translation_m", [0, 0, 0]),
                               dict(entry, object_id=f"{path.stem}@{factor:g}"),
                               config.get("allow_convex_hull", False))
