import struct
import numpy as np
import pytest
from Mesh_Audit import inspect_mesh


def test_closed_tetrahedron(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    path = tmp_path / 'tetra.stl'
    records = [struct.pack('<12fH', *([0.0] * 3 + vertices[f].reshape(-1).tolist()), 0) for f in faces]
    path.write_bytes(b' ' * 80 + struct.pack('<I', 4) + b''.join(records))
    result = inspect_mesh(path)
    assert result['closed_consistently_oriented_edges']
    assert result['surface_to_hull_volume_ratio'] == pytest.approx(1)
    assert result['signed_surface_volume_m3'] == pytest.approx(1 / 6)
    assert result['surface_vertices_inside_hull_by_over_0_1mm'] == 0


def test_reject_invalid_stl(tmp_path):
    path = tmp_path / 'bad.stl'
    path.write_bytes(b'not an STL')
    with pytest.raises(ValueError, match='binary STL'):
        inspect_mesh(path)
