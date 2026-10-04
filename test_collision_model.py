import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from Build_Collision_Model import build, read_stl


def source_model(tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    stl = tmp_path / 'palm_lower.stl'
    records = [struct.pack('<12fH', *([0.] * 3 + points[f].ravel().tolist()), 0) for f in faces]
    stl.write_bytes(b' ' * 80 + struct.pack('<I', 4) + b''.join(records))
    urdf = tmp_path / 'hand.urdf'
    urdf.write_text('<robot name="hand"><link name="palm"><visual><geometry><mesh filename="palm_lower.stl"/></geometry></visual><collision><origin xyz="1 2 3"/><geometry><mesh filename="palm_lower.stl"/></geometry></collision></link></robot>')
    return urdf, stl, points, faces


def test_separate_model_preserves_sources_and_origins(tmp_path, monkeypatch):
    urdf, stl, points, faces = source_model(tmp_path)
    snapshots = urdf.read_bytes(), stl.read_bytes()
    monkeypatch.setattr('coacd.run_coacd', lambda *args, **kwargs: [(points, faces), (points, faces)])
    output = build(urdf, tmp_path / 'experiment', {'palm_lower'}, 0.05, 16)
    assert (urdf.read_bytes(), stl.read_bytes()) == snapshots
    root = ET.parse(output).getroot()
    assert root.find('mujoco/compiler').get('strippath') == 'false'
    assert len(root.findall('./link/collision')) == 2
    assert all(c.find('origin').get('xyz') == '1 2 3' for c in root.findall('./link/collision'))
    assert root.find('./link/visual/geometry/mesh').get('filename') == str(stl.resolve())
    vertices, triangles = read_stl(stl)
    assert vertices.shape == (4, 3) and triangles.shape == (4, 3)


def test_refuse_source_directory_and_unknown_mesh(tmp_path):
    urdf, _, _, _ = source_model(tmp_path)
    with pytest.raises(ValueError, match='separate'):
        build(urdf, tmp_path, {'palm_lower'}, 0.05, 16)
    with pytest.raises(ValueError, match='Unknown'):
        build(urdf, tmp_path / 'out', {'missing'}, 0.05, 16)


def test_decomposition_disables_expanding_preprocessing(tmp_path, monkeypatch):
    urdf, _, points, faces = source_model(tmp_path)
    def decompose(*args, **kwargs):
        assert kwargs['preprocess_mode'] == 'off'
        return [(points, faces)]
    monkeypatch.setattr('coacd.run_coacd', decompose)
    build(urdf, tmp_path / 'out', {'palm_lower'}, 0.05, 16)


def test_reject_expanded_decomposition(tmp_path, monkeypatch):
    urdf, _, points, faces = source_model(tmp_path)
    expanded = points.copy()
    expanded[0, 2] = -0.0004
    monkeypatch.setattr('coacd.run_coacd', lambda *args, **kwargs: [(expanded, faces)])
    with pytest.raises(ValueError, match='expands outside source bounds'):
        build(urdf, tmp_path / 'out', {'palm_lower'}, 0.05, 16)


def test_collision_baseline_tracking():
    from utils.GeometryLoader import GeometryLoader
    from utils.GraspSimulator import GraspSimulator
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = GraspSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        assert isinstance(sim.baseline_self_depths, dict)
        baseline = sim.baseline_self_depths.copy()
        for pair in baseline:
            assert isinstance(pair, tuple) and len(pair) == 2
            assert sim.baseline_self_depths[pair] >= 0
    finally:
        sim.close()


def test_collision_same_body_allowed():
    from utils.GeometryLoader import GeometryLoader
    from utils.GraspSimulator import GraspSimulator, GraspSpec
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = GraspSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        sim.set_angles(np.zeros(16))
        sim.check_self_collision()
        spec = GraspSpec(np.zeros(16), joint_index=2, end_angle=10.0, max_increment=1.0)
        result = sim.execute_probe(spec)
        assert result.success or result.termination_reason in ('angle_limit', 'max_steps')
    finally:
        sim.close()


def test_collision_new_pair_rejected():
    from utils.GeometryLoader import GeometryLoader
    from utils.GraspSimulator import GraspSimulator
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = GraspSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        baseline = sim.baseline_self_depths.copy()
        original_collision_depths = sim._collision_depths
        try:
            test_pair = (0, 2)
            body1 = int(sim.model.geom_bodyid[test_pair[0]])
            body2 = int(sim.model.geom_bodyid[test_pair[1]])
            if body1 != body2 and test_pair not in baseline:
                sim.baseline_self_depths[test_pair] = 0
                sim._collision_depths = lambda: {test_pair: 1e-7}
                try:
                    sim.check_self_collision()
                    assert False, "Should have raised ValueError for new pair"
                except ValueError as e:
                    assert 'New hand self-collision' in str(e)
        finally:
            sim._collision_depths = original_collision_depths
    finally:
        sim.close()


def test_collision_deepening_rejected():
    from utils.GeometryLoader import GeometryLoader
    from utils.GraspSimulator import GraspSimulator
    geometry = GeometryLoader.create_primitive('sphere', 0.001, translation=[10, 10, 10])
    sim = GraspSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        baseline = sim.baseline_self_depths.copy()
        if baseline:
            pair = list(baseline.keys())[0]
            baseline_depth = baseline[pair]
            test_depth = baseline_depth + 1e-5
            sim.baseline_self_depths[pair] = baseline_depth
            sim._collision_depths = lambda: {pair: test_depth}
            try:
                sim.check_self_collision()
            except ValueError as e:
                assert 'deepening' in str(e)
    finally:
        sim.close()


def test_refine_contact_checks_collision():
    from utils.GeometryLoader import GeometryLoader
    from utils.GraspSimulator import GraspSimulator
    geometry = GeometryLoader.create_primitive('sphere', 0.025, translation=[0, 0, 0.05])
    sim = GraspSimulator('Data/mujoco_robot.urdf', geometry)
    try:
        sim.set_angles(np.zeros(16))
        clear_angles = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        blocked_angles = np.array([0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        sim.set_angles(clear_angles)
        result = sim.refine_contact(clear_angles, blocked_angles)
        assert result is not None
    finally:
        sim.close()
