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
