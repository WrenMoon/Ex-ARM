#!/usr/bin/env python3
"""Diagnostic full-path replay with fresh measurements and explicit clearance failures."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import mujoco

from Grasp_Discovery import load_objects_from_config
from utils.GraspSimulator import GraspSimulator, GraspSpec
from utils.ProbeDiscovery import DiscoveredGrasp, select_grasps
from utils.ProbeIdentifier import ProbeIdentifier


class DiagnosticSimulator(GraspSimulator):
    """Continue through hand overlaps to measure them; never a hardware controller."""

    def __init__(self, *args):
        self.phase = "Zero"
        self.pairs = {}
        self.frames = 0
        self.baseline = None
        super().__init__(*args)
        self.baseline = self.depths()

    def depths(self):
        depths = {}
        for c in self.data.contact:
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            if self.object_geom_id not in pair and c.dist < -1e-6:
                depths[pair] = max(depths.get(pair, 0), -float(c.dist) * 1000)
        return depths

    def set_angles(self, angles):
        super().set_angles(angles)
        self.frames += 1
        for pair, depth in self.depths().items():
            key = ':'.join(map(str, pair))
            if key not in self.pairs or depth > self.pairs[key]['maximum_mm']:
                self.pairs[key] = {
                    'geom_ids': list(pair),
                    'geom_names': [self.model.geom(g).name for g in pair],
                    'body_names': [self.model.body(int(self.model.geom_bodyid[g])).name for g in pair],
                    'baseline_mm': (self.baseline or {}).get(pair, depth if self.baseline is None else 0),
                    'maximum_mm': depth, 'phase': self.phase, 'angles_deg': self.get_angles().tolist()}

    def check_self_collision(self):
        # Diagnostic-only: all overlaps are reported, rather than stopping at the first one.
        pass


def validate(results, model):
    fresh = copy.deepcopy(results)
    fresh['config']['hand_urdf'] = str(Path(model).resolve())
    fresh['library'] = []  # Old library measurements are incompatible with the new geometry.
    for grasp in fresh['grasps']:
        targets = grasp.get('targets') or {str(grasp['target_joint']): grasp['target_angle']}
        if {int(j): float(a) for j, a in targets.items()} != {grasp['target_joint']: grasp['target_angle']}:
            raise ValueError('This validator requires single-joint probe signatures')
        for key in ('measurements', 'per_object_angles', 'per_object_contact'):
            grasp[key] = {}
    geometries = load_objects_from_config(results['configuration'])
    paths, errors, pairs = [], [], {}
    zero_pose_contacts = {}
    attempted_frames = 0
    for name in results['objects']:
        sim = DiagnosticSimulator(str(model), geometries[name])
        try:
            zero_pose_contacts[name] = []
            for geom, distance in sim.fixed_object_distances().items():
                if distance <= 0:
                    zero_pose_contacts[name].append({'geom': sim.model.geom(geom).name,
                        'body': sim.model.body(int(sim.model.geom_bodyid[geom])).name,
                        'signed_distance_mm': float(distance * 1000)})
            for index, grasp in enumerate(fresh['grasps']):
                sim.pairs = {}
                sim.frames = 0
                try:
                    sim.phase = 'Zero'
                    sim.set_angles(np.zeros(16))
                    start = np.asarray(grasp['start_angles'], dtype=float)
                    spec = GraspSpec(start, grasp['target_joint'], grasp['target_angle'], grasp['max_increment'])
                    sim.phase = 'Approach'
                    sim.move_clear(start)
                    sim.phase = 'Probe (includes refinement trials)'
                    steps = int(np.ceil(abs(spec.end_angle - start[spec.joint_index]) / spec.max_increment)) + 2
                    outcome = sim.execute_probe(spec, max_steps=steps)
                    if outcome.termination_reason not in ('contact', 'angle_limit'):
                        raise ValueError('Incomplete probe')
                    sim.phase = 'Return to start'
                    sim.move_clear(start)
                    sim.phase = 'Return to zero'
                    returned = sim.move_clear(np.zeros(16))
                    np.testing.assert_allclose(returned, 0, atol=1e-12, rtol=0)
                    angle = float(outcome.final_angles[spec.joint_index])
                    grasp['per_object_angles'][name] = angle
                    grasp['per_object_contact'][name] = bool(outcome.contact_detected)
                    grasp['measurements'][name] = {'angle': angle, 'contact': bool(outcome.contact_detected),
                        'success': bool(outcome.contact_detected), 'final_angles': outcome.final_angles.tolist(),
                        'returned_angles': returned.tolist(), 'joint_status': outcome.joint_status,
                        'path_verified': False, 'diagnostic_replay_completed': True}
                    paths.append({'object': name, 'probe': index + 1, 'frames': sim.frames})
                except Exception as exc:
                    errors.append({'object': name, 'probe': index + 1, 'phase': sim.phase, 'error': str(exc)})
                attempted_frames += sim.frames
                for key, pair in sim.pairs.items():
                    if key not in pairs or pair['maximum_mm'] > pairs[key]['maximum_mm']:
                        pairs[key] = dict(pair, object=name, probe=index + 1)
            print(f"{name}: replayed", flush=True)
        finally:
            sim.close()
    for pair in pairs.values():
        pair['same_body'] = len(set(pair['body_names'])) == 1
        pair['new_pair'] = pair['baseline_mm'] == 0
        pair['deepening_mm'] = max(0, pair['maximum_mm'] - pair['baseline_mm'])
    status = {'complete': False, 'reason': 'replay_errors'}
    recognition = None
    if not errors:
        candidates = [DiscoveredGrasp(**{key: value for key, value in g.items()
                                        if key in DiscoveredGrasp.__dataclass_fields__})
                      for g in fresh['grasps']]
        _, status = select_grasps(candidates, fresh['objects'], results['config']['min_angle_separation'], len(candidates))
        identifier = ProbeIdentifier(fresh)
        correct = 0
        for name in fresh['objects']:
            observation = [{'angle': g['per_object_angles'][name], 'contact': g['per_object_contact'][name]} for g in fresh['grasps']]
            correct += identifier.identify(observation)['object_id'] == name
        recognition = {'correct': correct, 'total': len(fresh['objects']),
                       'note': 'Fresh signatures against regenerated references, not held-out physical validation'}
    fresh['status'] = dict(status, collision_clearance_verified=False, experimental=True)
    report = {'completed_paths': len(paths), 'expected_paths': len(results['objects']) * len(results['grasps']),
              'frames_checked': attempted_frames, 'errors': errors,
              'zero_pose_fixed_object_contacts': zero_pose_contacts,
              'collision_pairs': list(pairs.values()), 'discrimination': status, 'recognition': recognition,
              'collision_clearance_verified': False,
              'limitations': ['Diagnostic replay bypasses self-collision rejection to report all encountered overlaps',
                              'Object contact and transition checks remain enabled',
                              'Pair inspection uses MuJoCo-generated contacts, not exhaustive exact mesh intersections',
                              'Refinement trial poses are included; sampling does not establish continuous clearance',
                              'Baseline overlaps require assessment even when they do not deepen']}
    return report, fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('/tmp/leap-fixed-library-grasps.json'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/tmp/leap-v2-validation.json'))
    parser.add_argument('--fresh-results', type=Path, default=Path('/tmp/leap-v2-fresh-grasps.json'))
    args = parser.parse_args()
    inputs = {args.results.resolve(), args.model.resolve()}
    outputs = [args.output.resolve(), args.fresh_results.resolve()]
    if len(set(outputs)) != 2 or any(p in inputs for p in outputs):
        parser.error('Outputs must be distinct and must not overwrite inputs')
    report, fresh = validate(json.loads(args.results.read_text()), args.model)
    for path, data in [(args.output, report), (args.fresh_results, fresh)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, allow_nan=False))
    print(f"Paths: {report['completed_paths']}/{report['expected_paths']}; errors: {len(report['errors'])}")
    print('Discrimination:', report['discrimination'])
    print('Recognition:', report['recognition'])
    print('New pairs:', sum(p['new_pair'] for p in report['collision_pairs']))
    print('Deepened pairs:', sum(p['deepening_mm'] > 0.001 for p in report['collision_pairs']))
    print('Clearance NOT certified. Report:', args.output)
    return 1 if report['errors'] or not report['discrimination']['complete'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
