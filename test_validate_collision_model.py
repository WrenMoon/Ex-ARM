import json
from pathlib import Path

from utils.GeometryLoader import GeometryLoader
from Validate_Collision_Model import DiagnosticSimulator, validate


def test_diagnostic_records_baseline_and_does_not_certify_clearance():
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = DiagnosticSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        assert sim.baseline
        sim.set_angles([0.] * 16)
        assert sim.pairs
        assert all(p['baseline_mm'] > 0 for p in sim.pairs.values())
        assert all(p['maximum_mm'] >= p['baseline_mm'] for p in sim.pairs.values())
    finally:
        sim.close()


def test_fresh_replay_does_not_require_old_stopping_angle(monkeypatch):
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    monkeypatch.setattr('Validate_Collision_Model.load_objects_from_config', lambda config: {'far': geometry})
    grasp = {'start_angles': [0.] * 16, 'target_joint': 2, 'target_angle': 1., 'max_increment': 0.5,
             'measurements': {'far': {'angle': -999}}, 'per_object_angles': {'far': -999},
             'per_object_contact': {'far': True}, 'targets': {'2': 1.}}
    original = {'config': {'hand_urdf': 'Data/mujoco_robot.urdf', 'min_angle_separation': 2},
                'configuration': {}, 'objects': ['far'], 'grasps': [grasp], 'library': []}
    report, fresh = validate(original, Path('Data/mujoco_robot.urdf'))
    assert report['completed_paths'] == 1 and not report['errors']
    assert not report['collision_clearance_verified']
    measurement = fresh['grasps'][0]['measurements']['far']
    assert measurement['angle'] == 1 and measurement['contact'] is False
    assert not measurement['path_verified']
    assert original['grasps'][0]['per_object_angles']['far'] == -999
    assert report['discrimination']['no_contact_objects'] == ['far']


def test_separated_fixed_mesh_rejects_spurious_distance(monkeypatch):
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = DiagnosticSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        def forbidden(*args):
            raise AssertionError('Separated bounds must not invoke narrow-phase distance')
        monkeypatch.setattr('utils.GraspSimulator.mujoco.mj_geomDistance', forbidden)
        assert all(distance > 0 for distance in sim.fixed_object_distances().values())
        assert sim.contact_fingers() == set()
    finally:
        sim.close()


def test_overlapping_bounds_keep_narrow_phase(monkeypatch):
    geometry = GeometryLoader.create_primitive('sphere', 1., translation=[0, 0, 0])
    sim = DiagnosticSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        calls = []
        def distance(*args):
            calls.append(args)
            return -0.001
        monkeypatch.setattr('utils.GraspSimulator.mujoco.mj_geomDistance', distance)
        assert any(d < 0 for d in sim.fixed_object_distances().values())
        assert calls
        assert -1 in sim.contact_fingers()
    finally:
        sim.close()
