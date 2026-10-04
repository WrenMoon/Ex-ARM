import mujoco
import mujoco.viewer
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import os
import tempfile

from .GeometryLoader import PrimitiveGeometry


@dataclass
class GraspSpec:
    """Specification for a grasp probe motion."""
    start_angles: np.ndarray
    joint_index: int = 2
    end_angle: float = 80.0
    max_increment: float = 0.5
    targets: dict = field(default_factory=dict)


@dataclass
class ContactEvent:
    """Record of a contact event during probing."""
    step: int
    time: float
    joint_index: int
    finger_index: int
    angles: np.ndarray


@dataclass
class ProbeResult:
    """Result of a single probe execution."""
    grasp_spec: GraspSpec
    object_name: str
    success: bool
    contact_detected: bool
    contact_events: List[ContactEvent] = field(default_factory=list)
    final_angles: np.ndarray = field(default_factory=lambda: np.zeros(16))
    termination_reason: str = "unknown"
    achieved_angle: np.ndarray = field(default_factory=lambda: np.zeros(16))
    joint_status: dict = field(default_factory=dict)


class GraspSimulator:
    """
    Geometry-only grasp simulator using MuJoCo for contact detection.

    Handles:
    - Loading hand URDF with object attached to worldbody
    - Executing probe motions with geometry-only kinematics (qpos + mj_forward)
    - Per-finger independent stopping with contact or angle-limit detection
    - Retracting hand along predefined path
    """

    SIM_TO_REAL = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])
    REAL_TO_SIM = np.argsort(SIM_TO_REAL)

    def __init__(
        self,
        hand_urdf_path: str,
        object_geom: PrimitiveGeometry,
        object_placement: np.ndarray = None,
        enable_viewer: bool = False,
    ):
        """
        Initialize grasp simulator.

        Parameters
        ----------
        hand_urdf_path : str
            Path to hand URDF file
        object_geom : PrimitiveGeometry
            Object geometry to attach to world
        object_placement : (3,) array, optional
            Object center position; defaults to [0, 0, 0.05]
        enable_viewer : bool
            If True, launch MuJoCo viewer
        """
        self.hand_urdf_path = hand_urdf_path
        self.object_geom = object_geom
        self.object_placement = (
            object_placement if object_placement is not None
            else object_geom.translation
        )
        self.enable_viewer = enable_viewer

        self.model = None
        self.data = None
        self.viewer = None
        self.object_body_id = None
        self.object_geom_id = None
        self.finger_bodies = {}
        self.joint_ranges = {}

        self._load_model()

    def _load_model(self) -> None:
        spec = mujoco.MjSpec.from_file(str(Path(self.hand_urdf_path).resolve()))
        spec.meshdir = str(Path(self.hand_urdf_path).resolve().parent)
        hand_geoms = list(spec.geoms)
        for index, geom in enumerate(hand_geoms):
            if not geom.name:
                geom.name = f"hand_collision_{index}"
        self.object_geom.add_to_spec(spec, self.object_placement)
        for index, geom in enumerate(hand_geoms):
            spec.add_pair(name=f"object_hand_{index}", geomname1="probe_object_geom",
                          geomname2=geom.name)
        self.model = spec.compile()
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.data = mujoco.MjData(self.model)
        self.object_geom_id = self.model.geom("probe_object_geom").id
        self.object_body_id = int(self.model.geom_bodyid[self.object_geom_id])
        self.joint_ids = np.array([self.model.joint(str(i)).id for i in range(16)])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.limits = np.degrees(self.model.jnt_range[self.joint_ids])
        self._find_finger_bodies()
        self.fixed_hand_geoms = [i for i in range(self.model.ngeom)
                                 if i != self.object_geom_id
                                 and int(self.model.geom_bodyid[i]) not in self.finger_bodies]
        self.set_angles(np.zeros(16))
        self.baseline_self_pairs = self.self_collision_pairs()
        if self.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _find_finger_bodies(self) -> None:
        self.finger_bodies = {}
        for joint, body in enumerate(self.model.jnt_bodyid[self.joint_ids]):
            self.finger_bodies[int(body)] = joint // 4
        for body in range(1, self.model.nbody):
            parent = body
            while parent and parent not in self.finger_bodies:
                parent = int(self.model.body_parentid[parent])
            if parent in self.finger_bodies:
                self.finger_bodies[body] = self.finger_bodies[parent]

    def validate_angles(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (16,) or not np.isfinite(angles).all():
            raise ValueError("Expected 16 finite joint angles in degrees")
        if np.any(angles < self.limits[:, 0]) or np.any(angles > self.limits[:, 1]):
            raise ValueError("Joint angles exceed URDF limits")
        return angles.copy()

    def set_angles(self, angles):
        self.data.qpos[self.qpos_ids] = np.radians(self.validate_angles(angles))
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def get_angles(self):
        return np.degrees(self.data.qpos[self.qpos_ids]).copy()

    def self_collision_pairs(self):
        return {tuple(sorted((int(c.geom1), int(c.geom2))))
                for c in self.data.contact if c.dist < -1e-6
                and self.object_geom_id not in (c.geom1, c.geom2)}

    def contact_fingers(self):
        fingers = set()
        for contact in self.data.contact:
            if contact.dist > 0 or self.object_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom_id else contact.geom1
            fingers.add(self.finger_bodies.get(int(self.model.geom_bodyid[other]), -1))
        if any(mujoco.mj_geomDistance(self.model, self.data, geom, self.object_geom_id,
                                     0.001, None) <= 0 for geom in self.fixed_hand_geoms):
            fingers.add(-1)
        return fingers

    def check_self_collision(self):
        if self.self_collision_pairs() - self.baseline_self_pairs:
            raise ValueError("New hand self-collision along probe path")

    def execute_probe(
        self,
        grasp_spec: GraspSpec,
        max_steps: int = 1000,
        dt: float = 0.001,
    ) -> ProbeResult:
        """
        Execute probe motion with geometry-only kinematics.

        Updates qpos incrementally, calls mj_forward only (no mj_step),
        stops on contact or angle limit. Returns status and results.

        Parameters
        ----------
        grasp_spec : GraspSpec
            Start angles, target joint, target angle, increment per step
        max_steps : int
            Maximum steps before termination
        dt : float
            Not used (geometry-only); for API compatibility

        Returns
        -------
        ProbeResult
            Contact status, achieved angle, termination reason
        """
        current_angles = self.validate_angles(grasp_spec.start_angles)
        targets = grasp_spec.targets or {grasp_spec.joint_index: grasp_spec.end_angle}
        if any(not isinstance(j, int) or j < 0 or j >= 16 for j in targets):
            raise ValueError("Moving joint indices must be integers from 0 to 15")
        if len({j // 4 for j in targets}) != len(targets):
            raise ValueError("Only one moving joint per finger is supported")
        if not np.isfinite(grasp_spec.max_increment) or not 0 < grasp_spec.max_increment <= 1:
            raise ValueError("Increment must be positive and at most one degree")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        end = current_angles.copy()
        for joint, target in targets.items():
            end[joint] = target
        self.validate_angles(end)
        self.set_angles(current_angles)
        if self.contact_fingers():
            raise ValueError("Starting pose intersects the object")
        self.check_self_collision()
        contact_events = []
        status = {joint: "moving" for joint in targets}

        for step in range(max_steps):
            for joint, target in targets.items():
                if status[joint] != "moving":
                    continue
                candidate = current_angles.copy()
                candidate[joint] += np.clip(target - candidate[joint],
                                            -grasp_spec.max_increment, grasp_spec.max_increment)
                self.set_angles(candidate)
                self.check_self_collision()
                contacts = self.contact_fingers()
                if contacts - {joint // 4}:
                    self.set_angles(current_angles)
                    raise ValueError("Unexpected contact on a stationary finger or palm")
                if joint // 4 in contacts:
                    candidate = self.refine_contact(current_angles, candidate)
                    status[joint] = "contact"
                    contact_events.append(ContactEvent(step + 1, 0.0, joint, joint // 4,
                                                       candidate.copy()))
                elif abs(candidate[joint] - target) < 1e-10:
                    status[joint] = "angle_limit"
                current_angles = candidate
            if "moving" not in status.values():
                break
        status = {j: "max_steps" if s == "moving" else s for j, s in status.items()}
        contact_detected = "contact" in status.values()
        termination_reason = "max_steps" if "max_steps" in status.values() else (
            "contact" if contact_detected else "angle_limit")
        achieved_angle = self.get_angles()

        return ProbeResult(
            grasp_spec=grasp_spec,
            object_name=self.object_geom.metadata.get("object_id", "unknown"),
            success=contact_detected,
            contact_detected=contact_detected,
            contact_events=contact_events,
            final_angles=achieved_angle.copy(),
            termination_reason=termination_reason,
            achieved_angle=achieved_angle,
            joint_status=status,
        )

    def refine_contact(self, clear, blocked):
        for _ in range(20):
            middle = (clear + blocked) / 2
            self.set_angles(middle)
            if self.contact_fingers():
                blocked = middle
            else:
                clear = middle
        self.set_angles(clear)
        return clear

    def retract_hand(
        self,
        start_angles: np.ndarray,
        current_angles: np.ndarray,
        max_steps: int = 500,
        dt: float = 0.001,
    ) -> np.ndarray:
        """
        Retract hand: current -> start_angles -> zero.

        Executes path in two phases with small increments, stops on error.

        Parameters
        ----------
        start_angles : (16,) array
            Grasp start position in degrees
        current_angles : (16,) array
            Current position in degrees
        max_steps : int
            Max steps per phase
        dt : float
            Not used

        Returns
        -------
        np.ndarray
            Final angles (should be near zero if successful)
        """
        self.set_angles(current_angles)
        self.move_clear(start_angles, max_steps)
        self.move_clear(np.zeros(16), max_steps)
        return self.get_angles()

    def move_clear(self, target, max_steps=500):
        target = self.validate_angles(target)
        current = self.get_angles()
        steps = max(1, int(np.ceil(np.max(np.abs(target - current)) / 0.5)))
        if steps > max_steps:
            raise ValueError("Transition exceeds step budget")
        origin = current.copy()
        for step in range(1, steps + 1):
            candidate = origin + (target - origin) * (step / steps)
            self.set_angles(candidate)
            try:
                self.check_self_collision()
                if self.contact_fingers():
                    raise ValueError("Transition intersects the object")
            except ValueError:
                self.set_angles(current)
                raise
            current = candidate
        return self.get_angles()

    def close(self) -> None:
        """Close viewer if open."""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.close()
