import argparse
import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np

from utils.GeometryLoader import load_step_folder


class Placement:
    def __init__(self, folder, config_path, hand_urdf, object_name, scale_percent=100):
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text())
        if not np.isfinite(scale_percent) or scale_percent <= 0:
            raise ValueError("Scale percentage must be finite and positive")
        preview = dict(config, scale_percent={"start": scale_percent,
                                             "stop": scale_percent, "step": 1})
        self.geometry = next((g for g in load_step_folder(folder, preview)
                              if g.path.name == object_name), None)
        if self.geometry is None:
            raise ValueError(f"Object not found in configuration: {object_name}")
        spec = mujoco.MjSpec.from_file(str(Path(hand_urdf).resolve()))
        spec.meshdir = str(Path(hand_urdf).resolve().parent)
        self.geometry.add_to_spec(spec)
        spec.worldbody.add_light(pos=[0, 0, 0.5], dir=[0, 0, -1],
                                 diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3])
        next(g for g in spec.geoms if g.name == "probe_object_geom").rgba = [0.2, 0.65, 0.9, 1]
        body = next(b for b in spec.bodies if b.name == "probe_object")
        body.mocap = True
        body.add_site(name="mount_origin", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[0.002, 0.002, 0.002], rgba=[1, 0.2, 0.1, 1])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.mocap_id = int(self.model.body_mocapid[self.model.body("probe_object").id])
        self.translation = np.asarray(config.get("mount_translation_m", [0, 0, 0]), dtype=float)
        self.rotation = np.asarray(config.get("mount_rotation_rpy_deg", [0, 0, 0]), dtype=float)
        self.apply()

    def apply(self):
        for values in (self.translation, self.rotation):
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError("Mount translation and rotation must each contain three finite values")
        quaternion = np.empty(4)
        mujoco.mju_euler2Quat(quaternion, np.radians(self.rotation), "XYZ")
        self.data.mocap_pos[self.mocap_id] = self.translation
        self.data.mocap_quat[self.mocap_id] = quaternion
        mujoco.mj_forward(self.model, self.data)

    def adjust(self, key, translation_step, rotation_step):
        translations = {262: (0, 1), 263: (0, -1), 265: (1, 1), 264: (1, -1),
                        93: (2, 1), 91: (2, -1)}
        rotations = {49: (0, -1), 50: (0, 1), 51: (1, -1), 52: (1, 1),
                     53: (2, -1), 54: (2, 1)}
        if key in translations:
            axis, direction = translations[key]
            self.translation[axis] += direction * translation_step
        elif key in rotations:
            axis, direction = rotations[key]
            self.rotation[axis] += direction * rotation_step
        else:
            return False
        self.apply()
        return True

    def mount_values(self):
        return {"mount_translation_m": np.round(self.translation, 9).tolist(),
                "mount_rotation_rpy_deg": np.round(self.rotation, 6).tolist()}

    def save(self):
        config = json.loads(self.config_path.read_text())
        config.update(self.mount_values())
        self.config_path.write_text(json.dumps(config, indent=2) + "\n")

    def reload(self):
        config = json.loads(self.config_path.read_text())
        translation = self.translation.copy()
        rotation = self.rotation.copy()
        try:
            self.translation = np.asarray(config.get("mount_translation_m", [0, 0, 0]), dtype=float)
            self.rotation = np.asarray(config.get("mount_rotation_rpy_deg", [0, 0, 0]), dtype=float)
            self.apply()
        except ValueError:
            self.translation, self.rotation = translation, rotation
            raise


def main():
    parser = argparse.ArgumentParser(description="Placement-only STEP viewer; no probing or hardware.")
    parser.add_argument("--object-folder", type=Path, default=Path("Data/Objects"))
    parser.add_argument("--config", type=Path, default=Path("Data/Objects/config.json"))
    parser.add_argument("--hand-urdf", type=Path, default=Path("Data/mujoco_robot.urdf"))
    parser.add_argument("--object", default="cube.step")
    parser.add_argument("--scale-percent", type=float, default=100)
    parser.add_argument("--step-mm", type=float, default=1)
    parser.add_argument("--step-deg", type=float, default=1)
    args = parser.parse_args()
    if not np.isfinite([args.step_mm, args.step_deg]).all() or min(args.step_mm, args.step_deg) <= 0:
        parser.error("Adjustment steps must be finite and positive")
    placement = Placement(args.object_folder, args.config, args.hand_urdf,
                          args.object, args.scale_percent)
    print("Placement only: hand stays at zero; no physics, probing, or hardware commands.", flush=True)
    print("Arrows: X/Y | [ / ]: -Z/+Z | 1/2: roll -/+ | 3/4: pitch -/+ | 5/6: yaw -/+", flush=True)
    print("S: save mount to supplied config | R: reload mount | P: print | Esc: close", flush=True)
    print(f"Steps: {args.step_mm} mm, {args.step_deg} degrees. Red dot: CAD mounting origin.", flush=True)
    print("Axes: hand/world coordinates, not camera coordinates. Site frames show mount axes.", flush=True)
    print(json.dumps(placement.mount_values()), flush=True)
    keys = SimpleQueue()
    with mujoco.viewer.launch_passive(placement.model, placement.data,
                                      key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = (placement.translation + np.array([-0.04, -0.03, 0])) / 2
            viewer.cam.distance = 0.45
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -25
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            with viewer.lock():
                while not keys.empty():
                    key = keys.get()
                    try:
                        if key == 83:
                            placement.save()
                            print(f"Saved shared mounting transform to {args.config}", flush=True)
                        elif key == 82:
                            placement.reload()
                        elif key == 256:
                            return
                        changed = placement.adjust(key, args.step_mm / 1000, args.step_deg)
                        if changed or key in (80, 82, 83):
                            print(json.dumps(placement.mount_values()), flush=True)
                    except (ValueError, OSError) as error:
                        print(f"Placement update failed: {error}", flush=True)
                placement.apply()
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
