import json
from pathlib import Path

import numpy as np
import pytest

from Object_Placement import Placement
from utils.GeometryLoader import load_step_folder
from utils.GraspSimulator import GraspSimulator


@pytest.fixture
def placement(tmp_path):
    config = tmp_path / "config.json"
    values = json.loads(Path("Data/Objects/config.json").read_text())
    values.update(mount_translation_m=[0, 0, 0], mount_rotation_rpy_deg=[0, 0, 0])
    config.write_text(json.dumps(values))
    return Placement("Data/Objects", config, "Data/mujoco_robot.urdf", "cube.step")


def test_controls_save_reload(placement):
    original = json.loads(placement.config_path.read_text())
    for key in (262, 265, 93, 50, 52, 54):
        assert placement.adjust(key, 0.001, 1)
    np.testing.assert_allclose(placement.translation, [0.001] * 3)
    np.testing.assert_allclose(placement.rotation, [1] * 3)
    placement.save()
    saved = json.loads(placement.config_path.read_text())
    for key in ("objects", "scale_percent", "grasps", "allow_convex_hull"):
        assert saved[key] == original[key]
    placement.adjust(262, 0.001, 1)
    placement.reload()
    assert saved["mount_translation_m"] == placement.translation.tolist()
    for key in (263, 264, 91, 49, 51, 53):
        assert placement.adjust(key, 0.001, 1)
    np.testing.assert_allclose(placement.translation, 0, atol=1e-12)
    np.testing.assert_allclose(placement.rotation, 0, atol=1e-12)
    assert placement.data.time == 0
    assert not placement.adjust(0, 0.001, 1)


def test_viewer_matches_simulator_transform(placement):
    placement.translation[:] = [0.02, 0.03, -0.05]
    placement.rotation[:] = [23, -17, 41]
    placement.apply()
    placement.save()
    config = json.loads(placement.config_path.read_text())
    config["scale_percent"] = {"start": 100, "stop": 100, "step": 1}
    geometry = next(load_step_folder("Data/Objects", config))
    sim = GraspSimulator("Data/mujoco_robot.urdf", geometry)
    geom = placement.model.geom("probe_object_geom").id
    np.testing.assert_allclose(placement.data.geom_xpos[geom],
                               sim.data.geom_xpos[sim.object_geom_id], atol=1e-9)
    np.testing.assert_allclose(placement.data.geom_xmat[geom],
                               sim.data.geom_xmat[sim.object_geom_id], atol=1e-9)
    site = placement.model.site("mount_origin").id
    np.testing.assert_allclose(placement.data.site_xpos[site], placement.translation)


@pytest.mark.parametrize("name", ["cube.step", "sphere.step", "cylinder.step"])
def test_actual_objects_load(name):
    placement = Placement("Data/Objects", "Data/Objects/config.json",
                          "Data/mujoco_robot.urdf", name, 50)
    assert placement.geometry.scale_factor == 0.5
    assert placement.model.nmocap == 1
    assert placement.data.time == 0
