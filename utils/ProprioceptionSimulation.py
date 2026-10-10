"""Run the real grasp motion against fixed STEP objects in MuJoCo."""

import csv
from pathlib import Path

import mujoco
import numpy as np

from utils.Constants import Connection, Proprioception


def object_scales():
    start = Proprioception.scale_start_percent
    stop = Proprioception.scale_stop_percent
    step = Proprioception.scale_step_percent
    if step == 0 or (stop - start) * step < 0:
        raise ValueError("Invalid object scale sweep")
    return range(start, stop + (1 if step > 0 else -1), step)


class GraspSimulation:
    def __init__(self, object_info, scale_percent):
        import cadquery as cq

        object_path = Path(Proprioception.object_folder) / object_info["file"]
        imported = cq.importers.importStep(str(object_path))
        shape = cq.Compound.makeCompound(imported.vals())
        vertices, faces = shape.tessellate(0.05, 0.1)
        vertices = np.asarray([vertex.toTuple() for vertex in vertices])
        faces = np.asarray(faces, dtype=int)
        if len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid STEP mesh: {object_path}")
        vertices *= 0.001 * scale_percent / 100  # STEP files are in millimetres.

        hand_path = Path(Connection.model_path).resolve()
        spec = mujoco.MjSpec.from_file(str(hand_path))
        spec.meshdir = str(hand_path.parent)
        hand_geoms = list(spec.geoms)
        for index, geom in enumerate(hand_geoms):
            if not geom.name:
                geom.name = f"hand_collision_{index}"

        spec.add_mesh(name="object_mesh", uservert=vertices.ravel().tolist(),
                      userface=faces.ravel().tolist())
        quat = np.empty(4)
        mujoco.mju_euler2Quat(quat, np.radians(Proprioception.mount_rotation_rpy_deg), "XYZ")
        body = spec.worldbody.add_body(name="fixed_object",
                                       pos=Proprioception.mount_translation_m,
                                       quat=quat)
        body.add_geom(name="object_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname="object_mesh", contype=1, conaffinity=1)
        for index, geom in enumerate(hand_geoms):
            spec.add_pair(name=f"object_hand_{index}", geomname1="object_collision",
                          geomname2=geom.name)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.object_geom_id = self.model.geom("object_collision").id
        self.joint_ids = np.asarray([self.model.joint(str(joint)).id for joint in range(16)])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.joint_limits = np.degrees(self.model.jnt_range[self.joint_ids])

        finger_bodies = {}
        for joint, body_id in enumerate(self.model.jnt_bodyid[self.joint_ids]):
            finger_bodies[int(body_id)] = joint // 4
        for body_id in range(1, self.model.nbody):
            parent = body_id
            while parent and parent not in finger_bodies:
                parent = int(self.model.body_parentid[parent])
            if parent in finger_bodies:
                finger_bodies[body_id] = finger_bodies[parent]
        self.finger_bodies = finger_bodies
        self.palm_geoms = [geom for geom in range(self.model.ngeom)
                           if geom != self.object_geom_id
                           and int(self.model.geom_bodyid[geom]) not in finger_bodies]
        mujoco.mj_forward(self.model, self.data)
        self.object_low, self.object_high = self.geom_bounds(self.object_geom_id)
        self.palm_bounds = {geom: self.geom_bounds(geom) for geom in self.palm_geoms}

    def geom_bounds(self, geom):
        rotation = self.data.geom_xmat[geom].reshape(3, 3)
        if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(self.model.geom_dataid[geom])
            start = int(self.model.mesh_vertadr[mesh])
            count = int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[start:start + count]
            points = np.einsum("ij,kj->ik", vertices, rotation, optimize=False)
            points += self.data.geom_xpos[geom]
            return points.min(axis=0), points.max(axis=0)
        bounds = self.model.geom_aabb[geom]
        center = self.data.geom_xpos[geom] + rotation @ bounds[:3]
        extent = np.abs(rotation) @ bounds[3:]
        return center - extent, center + extent

    def set_angles(self, angles):
        angles = np.asarray(angles, dtype=float)
        offsets = np.asarray(Proprioception.sim_joint_offsets_deg, dtype=float)
        if angles.shape != (16,) or offsets.shape != (16,):
            raise ValueError("Expected 16 grasp angles and 16 simulation offsets")
        sim_angles = angles + offsets
        if not np.isfinite(sim_angles).all():
            raise ValueError("Non-finite simulation joint angle")
        if np.any(sim_angles < self.joint_limits[:, 0] - 1e-6) or np.any(
                sim_angles > self.joint_limits[:, 1] + 1e-6):
            bad = np.flatnonzero((sim_angles < self.joint_limits[:, 0] - 1e-6)
                             | (sim_angles > self.joint_limits[:, 1] + 1e-6))
            raise ValueError(f"Grasp angles exceed URDF joint limits at joints {bad.tolist()}; "
                             "check the simulation angle offsets and URDF calibration")
        self.data.qpos[self.qpos_ids] = np.radians(sim_angles)
        mujoco.mj_forward(self.model, self.data)

    def contact_fingers(self):
        fingers = set()
        for contact in self.data.contact:
            if contact.dist > 0 or self.object_geom_id not in (contact.geom1, contact.geom2):
                continue
            hand_geom = contact.geom2 if contact.geom1 == self.object_geom_id else contact.geom1
            body_id = int(self.model.geom_bodyid[hand_geom])
            fingers.add(self.finger_bodies.get(body_id, -1))

        for geom in self.palm_geoms:
            low, high = self.palm_bounds[geom]
            gap = max(float(np.max(low - self.object_high)),
                      float(np.max(self.object_low - high)))
            if gap > 0:
                continue
            distance = mujoco.mj_geomDistance(self.model, self.data, geom,
                                              self.object_geom_id, 0.001, None)
            if distance <= 0:
                fingers.add(-1)
        return fingers

    def grasp_one(self, grip, grip_number):
        angles = np.asarray(grip["start_angles"], dtype=float).copy()
        targets = np.asarray(grip["max_angles"], dtype=float)
        increments = np.asarray(grip["step_sizes"], dtype=float)
        if targets.shape != (16,) or increments.shape != (16,):
            raise ValueError("Expected a 16-joint grasp")
        offsets = np.asarray(Proprioception.sim_joint_offsets_deg, dtype=float)
        endpoints = np.stack((angles + offsets, targets + offsets))
        outside = (endpoints < self.joint_limits[:, 0]) | (endpoints > self.joint_limits[:, 1])
        if outside.any():
            bad = np.flatnonzero(outside.any(axis=0))
            raise ValueError(f"Grasp endpoints exceed URDF joint limits at joints "
                             f"{bad.tolist()}; check the simulation angle offsets "
                             "and URDF calibration")
        self.set_angles(angles)
        if self.contact_fingers():
            raise ValueError("Object touches the hand before the grasp starts")

        stopped_fingers = [False] * 4
        stop_reasons = [""] * 4
        self.trace.append((grip_number, 0, angles.copy(), stop_reasons.copy()))
        for step in range(Proprioception.max_grasp_steps):
            for finger in range(4):
                if stopped_fingers[finger]:
                    continue
                joints = slice(finger * 4, finger * 4 + 4)
                candidate = angles.copy()
                for joint in range(finger * 4, finger * 4 + 4):
                    if increments[joint] > 0:
                        candidate[joint] = min(angles[joint] + increments[joint], targets[joint])
                    else:
                        candidate[joint] = max(angles[joint] + increments[joint], targets[joint])
                if np.array_equal(candidate[joints], angles[joints]):
                    stopped_fingers[finger] = True
                    stop_reasons[finger] = "angle_limit"
                    continue

                self.set_angles(candidate)
                contacts = self.contact_fingers()
                if -1 in contacts or contacts - {finger}:
                    raise ValueError(f"Unexpected palm or stationary finger contact: {contacts}")
                if finger in contacts:
                    clear = angles.copy()
                    blocked = candidate.copy()
                    for _ in range(8):
                        middle = (clear + blocked) / 2
                        self.set_angles(middle)
                        if finger in self.contact_fingers():
                            blocked = middle
                        else:
                            clear = middle
                    angles = clear
                    self.set_angles(angles)
                    stopped_fingers[finger] = True
                    stop_reasons[finger] = "contact"
                else:
                    angles = candidate
                    if np.allclose(angles[joints], targets[joints], atol=1e-6):
                        stopped_fingers[finger] = True
                        stop_reasons[finger] = "angle_limit"
            self.trace.append((grip_number, step + 1, angles.copy(), stop_reasons.copy()))
            if all(stopped_fingers):
                return angles, stop_reasons
        raise RuntimeError("Simulation grasp exceeded the maximum number of steps")

    def grasp(self):
        if not Proprioception.grip:
            raise ValueError("Add at least one grip to Proprioception.grip")
        self.trace = []
        all_angles = []
        all_reasons = []
        for grip_number, grip in enumerate(Proprioception.grip, 1):
            angles, reasons = self.grasp_one(grip, grip_number)
            all_angles.extend(angles)
            all_reasons.extend(reasons)
        return all_angles, all_reasons


def run_simulations():
    rows = []
    summary_path = Path(Proprioception.simulation_path)
    steps_path = Path(Proprioception.training_steps_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    angle_columns = [f"joint_{joint}" for joint in range(16 * len(Proprioception.grip))]
    stop_columns = [f"grip_{grip}_finger_{finger}_stop"
                    for grip in range(1, len(Proprioception.grip) + 1)
                    for finger in range(4)]
    step_angles = [f"joint_{joint}" for joint in range(16)]
    step_stops = [f"finger_{finger}_stop" for finger in range(4)]

    with summary_path.open("w", newline="") as summary_file, \
            steps_path.open("w", newline="") as steps_file:
        summary_writer = csv.writer(summary_file)
        steps_writer = csv.writer(steps_file)
        summary_writer.writerow(["class", "size_name", "size_m", "scale_percent"]
                                + angle_columns + stop_columns)
        steps_writer.writerow(["class", "size_name", "size_m", "scale_percent",
                               "grip", "step"] + step_angles + step_stops)

        for object_info in Proprioception.objects:
            for scale_percent in object_scales():
                simulation = GraspSimulation(object_info, scale_percent)
                angles, stop_reasons = simulation.grasp()
                size_m = object_info["reference_size_m"] * scale_percent / 100
                row = [object_info["class_name"], object_info["size_name"],
                       size_m, scale_percent, *angles, *stop_reasons]
                summary_writer.writerow(row)
                rows.append(row)
                for grip_number, step, angles_at_step, step_reasons in simulation.trace:
                    steps_writer.writerow([object_info["class_name"],
                                           object_info["size_name"], size_m,
                                           scale_percent, grip_number, step,
                                           *angles_at_step, *step_reasons])
                summary_file.flush()
                steps_file.flush()
                print(f"{object_info['class_name']} at {scale_percent}%: {stop_reasons}")
    return rows


if __name__ == "__main__":
    run_simulations()
