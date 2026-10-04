#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from queue import SimpleQueue

import mujoco.viewer
import numpy as np

from Grasp_Discovery import load_objects_from_config
from utils.GraspSimulator import GraspSimulator, GraspSpec


class RecordedSimulator(GraspSimulator):
    def __init__(self, *args):
        self.frames = []
        self.phase = "Zero"
        super().__init__(*args)

    def set_angles(self, angles):
        super().set_angles(angles)
        self.frames.append((self.phase, self.get_angles()))


def record_probe(sim, grasp, object_id):
    sim.frames = []
    sim.phase = "Zero"
    sim.set_angles(np.zeros(16))
    start = np.asarray(grasp["start_angles"], dtype=float)
    spec = GraspSpec(start, grasp["target_joint"], grasp["target_angle"],
                     grasp["max_increment"])
    sim.phase = "Approach"
    sim.move_clear(start)
    sim.phase = "Probe"
    steps = int(np.ceil(abs(spec.end_angle - start[spec.joint_index]) / spec.max_increment)) + 2
    outcome = sim.execute_probe(spec, max_steps=steps)
    saved = grasp["measurements"][object_id]
    if outcome.termination_reason not in ("contact", "angle_limit"):
        raise ValueError("Replay did not finish")
    if outcome.contact_detected != saved["contact"]:
        raise ValueError("Replay contact outcome differs from saved measurement")
    np.testing.assert_allclose(outcome.final_angles, saved["final_angles"], atol=1e-6, rtol=0)
    sim.phase = "Stop: " + outcome.termination_reason
    sim.set_angles(outcome.final_angles)
    sim.phase = "Return to start"
    sim.move_clear(start)
    sim.phase = "Return to zero"
    sim.move_clear(np.zeros(16))
    np.testing.assert_allclose(sim.get_angles(), 0, atol=1e-12, rtol=0)
    return sim.frames.copy(), outcome


def main():
    parser = argparse.ArgumentParser(description="Replay selected simulation probes from discovery results")
    parser.add_argument("--results", type=Path, default=Path("/tmp/leap-fixed-library-grasps.json"))
    parser.add_argument("--object", help="Initial object ID, e.g. cylinder@0.6")
    parser.add_argument("--grasp", type=int, default=1, help="Initial probe number (1-based)")
    parser.add_argument("--speed", type=float, default=20, help="Playback degrees per second")
    parser.add_argument("--check-only", action="store_true", help="Check initial object's probes without opening a window")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("Speed must be finite and positive")
    result = json.loads(args.results.read_text())
    grasps = result["grasps"]
    if not 1 <= args.grasp <= len(grasps):
        parser.error("Grasp number outside saved probe range")
    geometries = load_objects_from_config(result["configuration"])
    ids = result["objects"]
    if not ids or any(name not in geometries for name in ids):
        parser.error("Saved objects could not be loaded")
    if args.object and args.object not in ids:
        parser.error("Unknown object ID; choose one from the results object's list")
    obj_index = ids.index(args.object) if args.object else 0
    grasp_index = args.grasp - 1
    urdf = result["config"]["hand_urdf"]
    if args.check_only:
        sim = RecordedSimulator(urdf, geometries[ids[obj_index]])
        try:
            for i, grasp in enumerate(grasps):
                frames, outcome = record_probe(sim, grasp, ids[obj_index])
                print(f"Verified {ids[obj_index]} probe {i + 1}: {outcome.termination_reason}; {len(frames)} frames")
        finally:
            sim.close()
        return

    print("SPACE: pause/resume | R: restart | LEFT/RIGHT: object | UP/DOWN: probe", flush=True)
    print("ESC: close viewer. Close window to exit. Simulation only; baseline self-overlaps remain tolerated.", flush=True)
    keys = SimpleQueue()
    while True:
        name = ids[obj_index]
        sim = RecordedSimulator(urdf, geometries[name])
        try:
            frames, outcome = record_probe(sim, grasps[grasp_index], name)
            print(f"Viewing {name}, probe {grasp_index + 1}/{len(grasps)}: "
                  f"{outcome.termination_reason}, joint {grasps[grasp_index]['target_joint']}, "
                  f"stop {outcome.achieved_angle[grasps[grasp_index]['target_joint']]:.3f} degrees", flush=True)
            sim.set_angles(frames[0][1])
            with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
                viewer.cam.lookat[:] = sim.data.geom_xpos[sim.object_geom_id]
                viewer.cam.distance = 0.45
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                frame = 0
                paused = False
                last_phase = None
                next_scene = False
                while viewer.is_running():
                    while not keys.empty():
                        key = keys.get()
                        if key == 32:
                            paused = not paused
                        elif key in (82, 114):
                            frame = 0
                        elif key in (262, 263):
                            obj_index = (obj_index + (1 if key == 262 else -1)) % len(ids)
                            next_scene = True
                        elif key in (264, 265):
                            grasp_index = (grasp_index + (1 if key == 264 else -1)) % len(grasps)
                            next_scene = True
                    if next_scene:
                        break
                    phase, angles = frames[frame]
                    if phase != last_phase:
                        print("  " + phase, flush=True)
                        last_phase = phase
                    with viewer.lock():
                        sim.data.qpos[sim.qpos_ids] = np.radians(angles)
                        mujoco.mj_forward(sim.model, sim.data)
                    viewer.sync()
                    time.sleep(1.0 if phase.startswith("Stop:") and not paused else 0.5 / args.speed)
                    if not paused and frame < len(frames) - 1:
                        frame += 1
                    elif frame == len(frames) - 1:
                        paused = True
                if not next_scene:
                    return
        finally:
            sim.close()


if __name__ == "__main__":
    main()
