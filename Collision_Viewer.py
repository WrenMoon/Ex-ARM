#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from queue import SimpleQueue

import mujoco
import mujoco.viewer

from Grasp_Discovery import load_objects_from_config
from utils.GraspSimulator import GraspSimulator


def main():
    parser = argparse.ArgumentParser(description='Inspect reported collision pairs at static worst-case poses; simulation only')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    report = json.loads(args.audit.read_text())
    pairs = sorted([p for p in report['collision_pairs'] if not p['same_body']],
                   key=lambda p: p['deepening_mm'], reverse=True)
    if not pairs:
        parser.error('No inter-body collision pairs in report')
    geometries = load_objects_from_config(results['configuration'])
    keys = SimpleQueue()
    index = 0
    print('Static simulation only. LEFT/RIGHT: pair; close window: exit. Red/orange: reported pair.', flush=True)
    while True:
        pair = pairs[index]
        sim = GraspSimulator(results['config']['hand_urdf'], geometries[pair['object']])
        try:
            sim.set_angles(pair['angles_deg'])
            ids = [sim.model.geom(name).id for name in pair['geom_names']]
            if ids != pair['geom_ids']:
                raise ValueError('Audit geometry IDs do not match selected model')
            print(f"Pair {index + 1}/{len(pairs)}: {pair['body_names']}; {pair['object']}; "
                  f"probe {pair['probe']}; {pair['phase']}; baseline {pair['baseline_mm']:.3f} mm; "
                  f"maximum {pair['maximum_mm']:.3f} mm; angles {pair['angles_deg']}", flush=True)
            sim.model.geom_rgba[:, :] = [0.65, 0.65, 0.65, 0.15]
            sim.model.geom_rgba[ids[0], :] = [1, 0.1, 0.1, 0.85]
            sim.model.geom_rgba[ids[1], :] = [1, 0.65, 0, 0.85]
            if args.check_only:
                index += 1
                if index == len(pairs):
                    print(f'Checked {index} static poses')
                    return
                continue
            next_pair = False
            with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put) as viewer:
                viewer.cam.lookat[:] = sim.data.geom_xpos[ids].mean(axis=0)
                viewer.cam.distance = 0.25
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                while viewer.is_running():
                    while not keys.empty():
                        key = keys.get()
                        if key in (262, 263):
                            index = (index + (1 if key == 262 else -1)) % len(pairs)
                            next_pair = True
                    if next_pair:
                        break
                    viewer.sync()
                    time.sleep(0.02)
            if not next_pair:
                return
        finally:
            sim.close()


if __name__ == '__main__':
    main()
