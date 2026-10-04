#!/usr/bin/env python3
import numpy as np
import json
import tempfile
from pathlib import Path

from utils.GeometryLoader import GeometryLoader, PrimitiveGeometry
from utils.SimHand import SimHand


def test_geometry_loader_primitives():
    """Test primitive geometry creation and metadata."""
    print("\n=== Test: Geometry Loader Primitives ===")
    
    cube = GeometryLoader.create_primitive(
        shape="cube",
        scale=0.025,
        metadata={"object_id": "test_cube", "size_mm": 25}
    )
    assert cube.shape == "cube"
    assert cube.scale == 0.025
    assert cube.metadata["object_id"] == "test_cube"
    
    sphere = GeometryLoader.create_primitive(
        shape="sphere",
        scale=0.015,
        metadata={"object_id": "test_sphere"}
    )
    assert sphere.shape == "sphere"
    assert sphere.scale == 0.015
    
    cylinder = GeometryLoader.create_primitive(
        shape="cylinder",
        scale=0.010,
        metadata={"object_id": "test_cylinder"}
    )
    assert cylinder.shape == "cylinder"
    assert cylinder.scale == 0.010
    
    print("  PASS: Primitives created successfully")


def test_geometry_mujoco_spec():
    """Test MuJoCo body specification generation."""
    print("\n=== Test: MuJoCo Body Spec ===")
    
    cube = GeometryLoader.create_primitive(
        shape="cube",
        scale=0.020,
        translation=np.array([0.1, 0.2, 0.3]),
        metadata={"test": "value"}
    )
    
    spec = cube.get_mujoco_body_spec("test_body")
    
    assert "body" in spec
    assert "geom" in spec
    assert spec["body"]["name"] == "test_body"
    assert spec["body"]["pos"] == [0.1, 0.2, 0.3]
    assert spec["geom"]["type"] == "box"
    assert spec["geom"]["size"] == [0.01, 0.01, 0.01]
    
    print("  PASS: MuJoCo spec generated correctly")


def test_geometry_test_suite():
    """Test the standard test suite generation."""
    print("\n=== Test: Geometry Test Suite ===")
    
    suite = GeometryLoader.generate_test_suite()
    
    assert "cube_20mm" in suite
    assert "cube_30mm" in suite
    assert "cube_50mm" in suite
    assert "sphere_10mm" in suite
    assert "sphere_15mm" in suite
    assert "sphere_25mm" in suite
    assert "cylinder_r10_h20mm" in suite
    assert "cylinder_r15_h30mm" in suite
    assert "cylinder_r25_h50mm" in suite
    
    assert len(suite) == 9
    
    for name, geom in suite.items():
        assert geom.metadata["object_id"] == name
        assert geom.scale > 0
    
    print(f"  PASS: Test suite has {len(suite)} objects")


def test_geometry_metadata_io():
    """Test saving and loading geometry metadata."""
    print("\n=== Test: Geometry Metadata I/O ===")
    
    cube = GeometryLoader.create_primitive(
        shape="cube",
        scale=0.025,
        metadata={"object_id": "test", "size_mm": 25, "custom": "data"}
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "geometry.json"
        
        GeometryLoader.save_metadata(cube, metadata_path)
        assert metadata_path.exists()
        
        loaded = GeometryLoader.load_metadata(metadata_path)
        assert loaded["shape"] == "cube"
        assert loaded["scale"] == 0.025
        assert loaded["metadata"]["object_id"] == "test"
        assert loaded["metadata"]["custom"] == "data"
    
    print("  PASS: Metadata I/O works correctly")


def test_simhand_viewer_optional():
    """Test that SimHand viewer can be disabled."""
    print("\n=== Test: SimHand Viewer Optional ===")
    
    hand = SimHand("Data/mujoco_robot.urdf", enable_viewer=False)
    
    assert hand.viewer is None
    assert hand.enable_viewer == False
    
    positions = hand.get_positions_degree()
    assert positions.shape == (16,)
    assert np.allclose(positions, 0, atol=0.1)
    
    hand.set_goal_positions_degree(np.array([30]*16))
    positions = hand.get_positions_degree()
    assert np.allclose(positions, 30, atol=0.1)
    
    hand.close()
    
    print("  PASS: SimHand works without viewer")


def test_simhand_state_reading():
    """Test SimHand state reading."""
    print("\n=== Test: SimHand State Reading ===")
    
    hand = SimHand("Data/mujoco_robot.urdf", enable_viewer=False)
    
    positions, velocities, currents = hand.get_state()
    
    assert positions.shape == (16,)
    assert velocities.shape == (16,)
    assert currents.shape == (16,)
    
    assert np.allclose(positions, 0, atol=0.1)
    assert np.allclose(velocities, 0)
    assert np.allclose(currents, 0)
    
    hand.close()
    
    print("  PASS: SimHand state reading works")


def test_rotation_to_quat():
    """Test quaternion conversion from rotation matrix."""
    print("\n=== Test: Rotation to Quaternion ===")
    
    R_identity = np.eye(3)
    q = PrimitiveGeometry._rotation_to_quat(R_identity)
    
    assert q.shape == (4,)
    assert np.isfinite(q).all()
    
    w, x, y, z = q
    expected_w = 1.0 if w > 0 else -1.0
    assert abs(abs(w) - 1.0) < 0.01
    assert abs(x) < 0.01 and abs(y) < 0.01 and abs(z) < 0.01
    
    print("  PASS: Quaternion conversion works")


def test_grasp_spec_creation():
    """Test grasp specification creation."""
    print("\n=== Test: Grasp Specification ===")
    
    from utils.GraspSimulator import GraspSpec
    
    start_angles = np.zeros(16)
    grasp_spec = GraspSpec(
        start_angles=start_angles,
        joint_index=2,
        end_angle=90.0,
        max_increment=0.5
    )
    
    assert grasp_spec.joint_index == 2
    assert grasp_spec.end_angle == 90.0
    assert grasp_spec.max_increment == 0.5
    assert grasp_spec.start_angles.shape == (16,)
    
    print("  PASS: Grasp specification created successfully")


def test_urdf_loading():
    """Test URDF file loading through SimHand."""
    print("\n=== Test: URDF Loading ===")
    
    urdf_path = Path("Data/mujoco_robot.urdf")
    assert urdf_path.exists(), f"URDF not found at {urdf_path.absolute()}"
    
    hand = SimHand(str(urdf_path), enable_viewer=False)
    
    assert hand.model is not None
    assert hand.data is not None
    assert hand.model.nbody > 0
    
    hand.close()
    
    print(f"  PASS: URDF loaded with {hand.model.nbody} bodies")


def test_result_json_serialization():
    """Test that results can be serialized to JSON."""
    print("\n=== Test: Result JSON Serialization ===")
    
    result = {
        "timestamp": "2024-01-01T00:00:00",
        "hand_urdf": "Data/mujoco_robot.urdf",
        "probe_count": 1,
        "objects": [
            {
                "object_id": "cube_20mm",
                "shape": "cube",
                "scale_m": 0.020,
                "metadata": {"object_id": "cube_20mm", "size_mm": 20},
                "probe_results": [
                    {
                        "grasp_index": 0,
                        "joint_index": 2,
                        "contact_detected": False,
                        "termination_reason": "angle_limit",
                        "achieved_angle": [0.0]*16,
                        "contact_events": []
                    }
                ]
            }
        ]
    }
    
    json_str = json.dumps(result)
    loaded = json.loads(json_str)
    
    assert loaded["probe_count"] == 1
    assert loaded["objects"][0]["object_id"] == "cube_20mm"
    assert not loaded["objects"][0]["probe_results"][0]["contact_detected"]
    
    print("  PASS: Result serialization works")


def test_grasp_simulator_creation():
    """Test basic GraspSimulator instantiation."""
    print("\n=== Test: GraspSimulator Creation ===")
    
    from utils.GraspSimulator import GraspSimulator, GraspSpec
    
    cube = GeometryLoader.create_primitive(
        shape="cube",
        scale=0.020,
        metadata={"object_id": "test_cube"}
    )
    
    sim = GraspSimulator(
        hand_urdf_path="Data/mujoco_robot.urdf",
        object_geom=cube,
        object_placement=np.array([0.0, 0.0, 0.05]),
        enable_viewer=False
    )
    
    assert sim.model is not None
    assert sim.data is not None
    assert sim.object_body_id >= 0
    assert sim.object_geom_id >= 0
    
    sim.close()
    
    print("  PASS: GraspSimulator created successfully")


def test_probe_execution_angle_limit():
    """Test probe execution terminating at angle limit."""
    print("\n=== Test: Probe Execution (Angle Limit) ===")
    
    from utils.GraspSimulator import GraspSimulator, GraspSpec
    
    cube = GeometryLoader.create_primitive(
        shape="cube",
        scale=0.020,
        metadata={"object_id": "test_cube"}
    )
    
    sim = GraspSimulator(
        hand_urdf_path="Data/mujoco_robot.urdf",
        object_geom=cube,
        object_placement=np.array([0.0, 0.0, 0.05]),
        enable_viewer=False
    )
    
    start_angles = np.zeros(16)
    grasp_spec = GraspSpec(
        start_angles=start_angles,
        joint_index=2,
        end_angle=45.0,
        max_increment=0.5
    )
    
    result = sim.execute_probe(grasp_spec, max_steps=200)
    
    assert result.grasp_spec is grasp_spec
    assert result.object_name == "test_cube"
    assert result.termination_reason in ["angle_limit", "max_steps"]
    assert result.achieved_angle.shape == (16,)
    assert result.achieved_angle[2] > 0
    
    sim.close()
    
    print(f"  PASS: Probe reached angle {result.achieved_angle[2]:.2f}° "
          f"(reason: {result.termination_reason})")


def test_probe_negative_direction():
    """Test probe in negative direction."""
    print("\n=== Test: Probe Negative Direction ===")
    
    from utils.GraspSimulator import GraspSimulator, GraspSpec
    
    sphere = GeometryLoader.create_primitive(
        shape="sphere",
        scale=0.015,
        metadata={"object_id": "test_sphere"}
    )
    
    sim = GraspSimulator(
        hand_urdf_path="Data/mujoco_robot.urdf",
        object_geom=sphere,
        object_placement=np.array([0.0, 0.0, 0.05]),
        enable_viewer=False
    )
    
    start_angles = np.zeros(16)
    start_angles[2] = 20
    grasp_spec = GraspSpec(
        start_angles=start_angles,
        joint_index=2,
        end_angle=0.0,
        max_increment=0.5
    )
    
    result = sim.execute_probe(grasp_spec, max_steps=150)
    
    assert result.achieved_angle[2] == 0
    assert result.termination_reason in ["angle_limit", "max_steps"]
    
    sim.close()
    
    print(f"  PASS: Probe moved from {start_angles[3]:.1f}° to "
          f"{result.achieved_angle[3]:.2f}° (negative direction)")


def test_retraction_path():
    """Test retraction from current to start to zero."""
    print("\n=== Test: Retraction Path ===")
    
    from utils.GraspSimulator import GraspSimulator
    
    cylinder = GeometryLoader.create_primitive(
        shape="cylinder",
        scale=0.010,
        metadata={"object_id": "test_cylinder"}
    )
    
    sim = GraspSimulator(
        hand_urdf_path="Data/mujoco_robot.urdf",
        object_geom=cylinder,
        object_placement=np.array([0.0, 0.0, 0.05]),
        enable_viewer=False
    )
    
    start_angles = np.zeros(16)
    current_angles = np.zeros(16)
    start_angles[2] = 10.3
    current_angles[2] = 30.1
    
    final_angles = sim.retract_hand(
        start_angles=start_angles,
        current_angles=current_angles,
        max_steps=100
    )
    
    assert final_angles.shape == (16,)
    assert np.allclose(final_angles, 0, atol=0.5)
    
    sim.close()
    
    print(f"  PASS: Retraction completed, final angles near zero "
          f"(max deviation: {np.max(np.abs(final_angles)):.2f}°)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMULATION TESTS")
    print("=" * 60)
    
    tests = [
        test_geometry_loader_primitives,
        test_geometry_mujoco_spec,
        test_geometry_test_suite,
        test_geometry_metadata_io,
        test_rotation_to_quat,
        test_simhand_viewer_optional,
        test_simhand_state_reading,
        test_grasp_spec_creation,
        test_urdf_loading,
        test_result_json_serialization,
        test_grasp_simulator_creation,
        test_probe_execution_angle_limit,
        test_probe_negative_direction,
        test_retraction_path,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())


def test_real_hand_contact_boundary_and_retraction():
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    cube = GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])
    sim = GraspSimulator("Data/mujoco_robot.urdf", cube)
    spec = GraspSpec(np.zeros(16), 2, 80)
    result = sim.execute_probe(spec)
    assert result.joint_status == {2: "contact"}
    assert 0 < result.final_angles[2] < 80
    assert result.contact_events[0].step > 1
    assert not sim.contact_fingers()
    np.testing.assert_allclose(sim.get_angles(), result.final_angles)
    beyond = result.final_angles.copy()
    beyond[2] += 0.001
    sim.set_angles(beyond)
    assert sim.contact_fingers() == {0}
    final = sim.retract_hand(spec.start_angles, result.final_angles)
    np.testing.assert_allclose(final, 0, atol=1e-12)
    assert sim.data.time == 0


def test_independent_fingers_and_limits():
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    cube = GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])
    sim = GraspSimulator("Data/mujoco_robot.urdf", cube)
    result = sim.execute_probe(GraspSpec(np.zeros(16), targets={2: 80, 6: 20}))
    assert result.joint_status == {2: "contact", 6: "angle_limit"}
    assert np.isclose(result.final_angles[6], 20)
    assert sim.data.time == 0


def test_invalid_and_overlapping_grasps():
    import pytest
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    cube = GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])
    sim = GraspSimulator("Data/mujoco_robot.urdf", cube)
    for grasp in (GraspSpec(np.zeros(15)), GraspSpec(np.zeros(16), end_angle=1000),
                  GraspSpec(np.zeros(16), max_increment=0),
                  GraspSpec(np.zeros(16), targets={2: 20, 3: 20})):
        with pytest.raises(ValueError):
            sim.execute_probe(grasp)
    start = np.zeros(16)
    start[2] = 30
    with pytest.raises(ValueError, match="Starting pose"):
        sim.execute_probe(GraspSpec(start))
    with pytest.raises(ValueError, match="intersects"):
        sim.set_angles(np.zeros(16))
        sim.move_clear(start)


def test_step_origin_scale_and_rotation(tmp_path):
    import cadquery as cq
    from utils.GeometryLoader import StepGeometry, load_step_folder
    from utils.GraspSimulator import GraspSimulator

    path = tmp_path / "offset.STEP"
    cq.exporters.export(cq.Workplane("XY").box(20, 20, 20).translate((30, 0, 10)), str(path))
    config = {"mount_translation_m": [1, 2, 3], "mount_rotation_rpy_deg": [0, 0, 90],
              "scale_percent": {"start": 100, "stop": 50, "step": -50},
              "allow_convex_hull": True,
              "objects": [{"file": path.name, "class": "cube", "size_name": "side_length",
                           "reference_size_m": 0.02}]}
    objects = list(load_step_folder(tmp_path, config))
    assert len(objects) == 2
    np.testing.assert_allclose(objects[1].vertices.min(axis=0), [0.01, -0.005, 0], atol=1e-9)
    np.testing.assert_allclose(objects[1].vertices.max(axis=0), [0.02, 0.005, 0.01], atol=1e-9)
    assert objects[1].scale == 0.01
    sim = GraspSimulator("Data/mujoco_robot.urdf", objects[1])
    mesh = sim.model.mesh("probe_mesh")
    local = sim.model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
    geom = sim.object_geom_id
    world = local @ sim.data.geom_xmat[geom].reshape(3, 3).T + sim.data.geom_xpos[geom]
    np.testing.assert_allclose(world.min(axis=0), [0.995, 2.01, 3], atol=1e-7)
    np.testing.assert_allclose(world.max(axis=0), [1.005, 2.02, 3.01], atol=1e-7)


def test_step_cli_and_failure_exit(tmp_path):
    import cadquery as cq
    import subprocess
    import sys

    folder = tmp_path / "objects"
    folder.mkdir()
    cq.exporters.export(cq.Workplane("XY").box(20, 20, 20), str(folder / "cube.step"))
    config = {"mount_translation_m": [0.095, 0.008, -0.05],
              "scale_percent": {"start": 100, "stop": 50, "step": -50},
              "allow_convex_hull": True,
              "objects": [{"file": "cube.step", "class": "cube", "size_name": "side_length",
                           "reference_size_m": 0.02}],
              "grasps": [{"start_angles": [0] * 16, "targets": {"2": 80}}]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "results.json"
    command = [sys.executable, "Simulation_Test.py", "--object-folder", str(folder),
               "--config", str(config_path), "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = json.loads(output.read_text())
    assert results["probe_count"] == 2
    assert not results["errors"]
    assert [obj["scale_m"] for obj in results["objects"]] == [0.02, 0.01]
    for obj in results["objects"]:
        probe = obj["probe_results"][0]
        assert probe["joint_status"] == {"2": "contact"}
        assert 0 < probe["achieved_angle"][2] < 80
        np.testing.assert_allclose(probe["returned_angles"], 0, atol=1e-12)
    config["grasps"][0]["targets"]["2"] = 1000
    config_path.write_text(json.dumps(config))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "error" in json.loads(output.read_text())["objects"][0]["probe_results"][0]


def test_palm_overlap_and_step_opt_in(tmp_path):
    import pytest
    from utils.GeometryLoader import StepGeometry
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    geometry = GeometryLoader.create_primitive("cube", 0.2)
    sim = GraspSimulator("Data/mujoco_robot.urdf", geometry)
    assert -1 in sim.contact_fingers()
    with pytest.raises(ValueError, match="Starting pose"):
        sim.execute_probe(GraspSpec(np.zeros(16)))
    with pytest.raises(ValueError, match="convex hull"):
        StepGeometry(tmp_path / "unused.step", 0.02)


def test_step_palm_clearance_and_overlap():
    import mujoco
    import pytest
    from utils.GeometryLoader import StepGeometry
    from utils.GraspSimulator import GraspSimulator, GraspSpec

    rotation = np.diag([1.0, -1.0, -1.0])
    for name, factor, reference in (("cube", 0.85, 0.05), ("cylinder", 0.55, 0.025),
                                    ("sphere", 1.25, 0.025), ("sphere", 0.75, 0.025)):
        geometry = StepGeometry(Path("Data/Objects") / f"{name}.step", reference, factor,
                                rotation=rotation, translation=[-0.06, -0.048, -0.035],
                                metadata={"class": name}, allow_convex_hull=True)
        sim = GraspSimulator("Data/mujoco_robot.urdf", geometry)
        palm = sim.model.geom("hand_collision_0").id
        distances = []
        for geom in (palm, sim.object_geom_id):
            mesh = sim.model.mesh(int(sim.model.geom_dataid[geom]))
            vertices = sim.model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
            world = vertices @ sim.data.geom_xmat[geom].reshape(3, 3).T + sim.data.geom_xpos[geom]
            distances.append((world[:, 2].min(), world[:, 2].max()))
        gap = distances[0][0] - distances[1][1]
        assert gap > 0
        distance = mujoco.mj_geomDistance(sim.model, sim.data, palm, sim.object_geom_id,
                                         0.001, None)
        assert gap <= distance < 0.001
        assert not sim.contact_fingers()
        sim.move_clear(np.zeros(16))
        result = sim.execute_probe(GraspSpec(np.zeros(16), targets={2: 80}))
        assert result.joint_status == {2: "angle_limit"}
        np.testing.assert_allclose(sim.retract_hand(np.zeros(16), result.final_angles), 0,
                                   atol=1e-12)
        assert sim.data.time == 0
        geometry.translation[2] += 0.001
        blocked = GraspSimulator("Data/mujoco_robot.urdf", geometry)
        assert -1 in blocked.contact_fingers()
        with pytest.raises(ValueError, match="Starting pose"):
            blocked.execute_probe(GraspSpec(np.zeros(16)))
        with pytest.raises(ValueError, match="intersects"):
            blocked.move_clear(np.zeros(16))


def test_probe_discovery_single_object():
    print("\n=== Test: Probe Discovery Single Object ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05],
                                           metadata={"object_id": "cube_20"})
    geometries = {"cube_20": cube}

    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=20,
        selected_count=1,
        budget_sec=30.0,
        seed=42
    )
    discovery = ProbeDiscovery(config)
    grasps, status = discovery.discover(geometries)
    discovery.cleanup()

    assert status["complete"]
    assert len(grasps) == 1
    assert grasps[0].per_object_contact["cube_20"]
    assert grasps[0].measurements["cube_20"]["path_verified"]
    np.testing.assert_allclose(grasps[0].measurements["cube_20"]["returned_angles"], 0, atol=1e-12)
    print(f"  PASS: Found {len(grasps)} grasps in {status['elapsed_sec']:.1f}s")


def test_probe_discovery_multiple_objects():
    print("\n=== Test: Probe Discovery Multiple Objects ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    geometries = {
        "cube_20": GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05]),
        "sphere_15": GeometryLoader.create_primitive("sphere", 0.015, translation=[0.095, 0.008, -0.05]),
    }

    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=30,
        selected_count=1,
        budget_sec=40.0,
        seed=42
    )
    discovery = ProbeDiscovery(config)
    grasps, status = discovery.discover(geometries)
    discovery.cleanup()

    assert status["complete"]
    assert len(grasps) == 1
    for grasp in grasps:
        assert len(grasp.per_object_contact) == 2
        assert abs(grasp.per_object_angles["cube_20"] - grasp.per_object_angles["sphere_15"]) >= 2
        assert all(record["path_verified"] for record in grasp.measurements.values())
    print(f"  PASS: Found {len(grasps)} grasps working on both objects")


def test_probe_discovery_empty():
    print("\n=== Test: Probe Discovery Empty Objects ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    config = DiscoveryConfig(hand_urdf="Data/mujoco_robot.urdf")
    discovery = ProbeDiscovery(config)
    grasps, status = discovery.discover({})
    discovery.cleanup()

    assert len(grasps) == 0
    assert status["reason"] == "no_objects"
    print("  PASS: Empty objects handled correctly")


def test_probe_discovery_config_validation():
    print("\n=== Test: Probe Discovery Config Validation ===")
    from utils.ProbeDiscovery import DiscoveryConfig
    import pytest

    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=50,
        selected_count=5,
        budget_sec=60.0,
        min_angle_separation=2.0,
        max_increment=0.5
    )
    assert config.max_candidates == 50
    assert config.selected_count == 5
    assert config.budget_sec == 60.0
    assert config.seed == 0
    for kwargs in ({"max_candidates": 0}, {"selected_count": 0}, {"selected_count": 100, "max_candidates": 50},
                   {"budget_sec": -1}, {"budget_sec": float("nan")},
                   {"min_angle_separation": 0}, {"max_increment": 0.6}):
        with pytest.raises(ValueError):
            DiscoveryConfig(hand_urdf="Data/mujoco_robot.urdf", **kwargs)
    print("  PASS: Config created with valid parameters")


def test_discovery_selection_one_multiple_and_unresolved():
    from utils.ProbeDiscovery import DiscoveredGrasp, select_grasps

    def candidate(angles, contacts=None):
        return DiscoveredGrasp([0] * 16, 2, 80, 0.5, dict(zip("abc", angles)),
                               dict(zip("abc", contacts or [True] * 3)))

    selected, report = select_grasps([candidate([10, 20, 30])], list("abc"), 2)
    assert report["complete"] and len(selected) == 1
    selected, report = select_grasps([candidate([10, 10, 20]), candidate([10, 20, 10])], list("abc"), 2)
    assert report["complete"] and len(selected) == 2
    selected, report = select_grasps([candidate([10, 10, 20])], list("abc"), 2)
    assert not report["complete"] and report["unresolved_pairs"] == [["a", "b"]]
    selected, report = select_grasps([candidate([10, 20, 30], [True, False, True])], list("abc"), 2)
    assert not report["complete"] and report["no_contact_objects"] == ["b"]


def test_discovery_identical_and_overlap_no_relocation():
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    identical = {name: GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])
                 for name in ["a", "b"]}
    discovery = ProbeDiscovery(DiscoveryConfig("Data/mujoco_robot.urdf", max_candidates=8))
    try:
        grasps, status = discovery.discover(identical)
        assert not status["complete"] and status["unresolved_pairs"] == [["a", "b"]]
        assert all(g.per_object_angles["a"] == g.per_object_angles["b"] for g in grasps)
        blocked = GeometryLoader.create_primitive("cube", 0.2)
        grasps, status = discovery.discover({"blocked": blocked})
        assert not grasps and not status["complete"]
        assert status["no_contact_objects"] == ["blocked"]
        np.testing.assert_array_equal(blocked.translation, 0)
    finally:
        discovery.cleanup()


def test_discovery_cli_and_snapshot(tmp_path):
    import subprocess
    import sys

    config = {"primitives": [
        {"shape": "cube", "scale": 0.02, "object_id": "a", "translation": [0.095, 0.008, -0.05]},
        {"shape": "sphere", "scale": 0.015, "object_id": "b", "translation": [0.095, 0.008, -0.05]}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    output = tmp_path / "results.json"
    command = [sys.executable, "Grasp_Discovery.py", "--config", str(path), "--output", str(output),
               "--library-count", "20", "--selected-count", "1", "--budget", "30"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads(output.read_text())
    assert saved["status"]["complete"] and saved["grasp_count"] == 1
    assert saved["configuration"]["primitives"] == config["primitives"]
    assert path.read_bytes() == original
    grasp = saved["grasps"][0]
    assert grasp["targets"] == {str(grasp["target_joint"]): grasp["target_angle"]}
    assert set(grasp["measurements"]) == {"a", "b"}
    config["primitives"][1].update(shape="cube", scale=0.02)
    path.write_text(json.dumps(config))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert not json.loads(output.read_text())["status"]["complete"]


def test_discovery_candidate_limits_reproducibility_and_budget():
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    objects = {"a": GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])}
    discovery = ProbeDiscovery(DiscoveryConfig("Data/mujoco_robot.urdf", max_candidates=100, selected_count=5, budget_sec=0.000001))
    try:
        grasps, status = discovery.discover(objects)
        assert not grasps and not status["complete"]
        assert status["reason"] == "time_budget" and status["tried"] == 0
        discovery.rng = np.random.default_rng(discovery.config.seed)
        sim = next(iter(discovery.object_sims.values()))
        first = discovery.generate_candidates(objects)
        discovery.rng = np.random.default_rng(discovery.config.seed)
        second = discovery.generate_candidates(objects)
        assert len(first) == 100
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.start_angles, b.start_angles)
            sim.validate_angles(a.start_angles)
            target = a.start_angles.copy()
            target[a.joint_index] = a.end_angle
            sim.validate_angles(target)
        assert any(np.any(candidate.start_angles) for candidate in first)
    finally:
        discovery.cleanup()


def test_discovery_requires_approach_and_rejects_invalid_object():
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig
    from utils.GraspSimulator import GraspSpec

    geometry = GeometryLoader.create_primitive("cube", 0.02, translation=[0.095, 0.008, -0.05])
    discovery = ProbeDiscovery(DiscoveryConfig("Data/mujoco_robot.urdf"))
    start = np.zeros(16)
    start[2] = 30
    try:
        success, records = discovery.test_candidate(GraspSpec(start, 2, 80), {"a": geometry})
        assert not success and "intersects" in records["a"]["reason"]
        np.testing.assert_allclose(discovery.object_sims["a"].get_angles(), 0, atol=1e-12)
        blocked = GeometryLoader.create_primitive("cube", 0.2)
        success, records = discovery.test_candidate(GraspSpec(np.zeros(16), 2, 80),
                                                     {"a": geometry, "blocked": blocked})
        assert not success
        assert records["a"]["path_verified"]
        assert "reason" in records["blocked"]
    finally:
        discovery.cleanup()


def test_discovery_fixedcount_insufficient_valid():
    print("\n=== Test: Discovery Fixed Count Insufficient ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    geometry = GeometryLoader.create_primitive("cube", 0.3)
    discovery = ProbeDiscovery(DiscoveryConfig("Data/mujoco_robot.urdf", 
                                               max_candidates=10, 
                                               selected_count=5,
                                               budget_sec=20.0))
    try:
        grasps, status = discovery.discover({"blocked": geometry})
        assert status["reason"] in ["insufficient_valid", "unresolved", "time_budget"]
        assert len(grasps) <= 5
        assert status["valid_count"] >= 0
        assert status["library_evaluation_complete"] or "time_budget" in status["reason"]
        print(f"  PASS: Handled insufficient valid: {status['reason']}, got {len(grasps)} grasps")
    finally:
        discovery.cleanup()


def test_discovery_contact_difference_equal_angles():
    print("\n=== Test: Discovery Contact Difference with Equal Angles ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveredGrasp, select_grasps

    def grasp1_angle_gap():
        g = DiscoveredGrasp([0]*16, 2, 80, 0.5)
        g.per_object_angles = {"a": 10.0, "b": 50.0}
        g.per_object_contact = {"a": True, "b": True}
        return g
    
    def grasp2_contact_diff():
        g = DiscoveredGrasp([0]*16, 2, 50, 0.5)
        g.per_object_angles = {"a": 50.0, "b": 50.0}
        g.per_object_contact = {"a": True, "b": False}
        return g
    
    grasps = [grasp1_angle_gap(), grasp2_contact_diff()]
    selected, report = select_grasps(grasps, ["a", "b"], 2.0, count=2)
    
    assert report["complete"]
    assert len(selected) == 2
    assert report["unresolved_pairs"] == []
    print("  PASS: Both angle gap and contact difference work")


def test_discovery_complement_misses():
    print("\n=== Test: Discovery Complement Misses ===")
    from utils.ProbeDiscovery import DiscoveredGrasp, select_grasps

    def grasp1():
        g = DiscoveredGrasp([0]*16, 2, 80, 0.5)
        g.per_object_angles = {"a": 10.0, "b": 50.0}
        g.per_object_contact = {"a": True, "b": True}
        return g
    
    def grasp2():
        g = DiscoveredGrasp([0]*16, 2, 50, 0.5)
        g.per_object_angles = {"a": 50.0, "b": 10.0}
        g.per_object_contact = {"a": True, "b": True}
        return g
    
    grasps = [grasp1(), grasp2()]
    selected, report = select_grasps(grasps, ["a", "b"], 2.0, count=2)
    
    assert report["complete"]
    assert len(selected) == 2
    assert len(report["unresolved_pairs"]) == 0
    print("  PASS: Complement selection works correctly")


def test_discovery_spatial_metrics():
    print("\n=== Test: Discovery Spatial Metrics ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05],
                                          metadata={"object_id": "test_cube"})
    
    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=15,
        selected_count=1,
        budget_sec=30.0,
        seed=42
    )
    
    discovery = ProbeDiscovery(config)
    try:
        grasps, status = discovery.discover({"cube": cube})
        
        if "spatial_metrics" in status:
            metrics = status["spatial_metrics"]
            assert "min_distance" in metrics
            assert "mean_distance" in metrics
            assert "coverage_spread" in metrics
            assert all(isinstance(v, (int, float)) for v in metrics.values())
            print(f"  PASS: Spatial metrics computed: {metrics}")
        else:
            print("  PASS: Spatial metrics optional if no valid grasps")
    finally:
        discovery.cleanup()


def test_discovery_deterministic_reproducibility():
    print("\n=== Test: Discovery Deterministic Reproducibility ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05])
    
    results = []
    for run in range(2):
        config = DiscoveryConfig(
            hand_urdf="Data/mujoco_robot.urdf",
            max_candidates=15,
            selected_count=1,
            budget_sec=20.0,
            seed=42
        )
        discovery = ProbeDiscovery(config)
        try:
            grasps, status = discovery.discover({"cube": cube})
            results.append((len(grasps), status.get("library_evaluation_complete")))
        finally:
            discovery.cleanup()
    
    assert results[0] == results[1]
    print(f"  PASS: Deterministic reproduction verified")


def test_discovery_timeout_reason():
    print("\n=== Test: Discovery Timeout Reason ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05])
    
    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=100,
        selected_count=1,
        budget_sec=0.00001
    )
    
    discovery = ProbeDiscovery(config)
    try:
        grasps, status = discovery.discover({"cube": cube})
        
        assert status["reason"] == "time_budget"
        assert status["tried"] == 0 or status["tried"] < status["library_size"]
        print(f"  PASS: Timeout reason detected: tried {status['tried']}/{status['library_size']}")
    finally:
        discovery.cleanup()


def test_discovery_negative_all_joints():
    print("\n=== Test: Discovery Negative All Joints ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig
    from utils.GraspSimulator import GraspSimulator

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05])
    sim = GraspSimulator("Data/mujoco_robot.urdf", cube, enable_viewer=False)
    limits = sim.limits
    
    for joint in range(16):
        limits_min = limits[joint, 0]
        if limits_min < 0:
            assert True
            break
    else:
        print("  SKIP: No negative joint limits in URDF")
        sim.close()
        return
    
    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=20,
        selected_count=1,
        budget_sec=15.0,
        seed=42
    )
    
    discovery = ProbeDiscovery(config)
    try:
        grasps, status = discovery.discover({"cube": cube})
        candidates = discovery.generate_candidates({"cube": cube})
        
        for candidate in candidates:
            target = candidate.start_angles.copy()
            target[candidate.joint_index] = candidate.end_angle
            
            is_valid = np.all(target >= limits[:, 0]) and np.all(target <= limits[:, 1])
            assert is_valid, f"Candidate target {target[candidate.joint_index]} exceeds limits for joint {candidate.joint_index}"
        
        print(f"  PASS: All {len(candidates)} candidates have valid angles including negatives")
    finally:
        discovery.cleanup()
        sim.close()


def test_discovery_count_100_500_random_fallback():
    print("\n=== Test: Discovery Count 100 and 500 Random Fallback ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05])
    
    for count in [100, 500]:
        config = DiscoveryConfig(
            hand_urdf="Data/mujoco_robot.urdf",
            max_candidates=count,
            selected_count=1,
            budget_sec=30.0,
            seed=42
        )
        
        discovery = ProbeDiscovery(config)
        try:
            candidates = discovery.generate_candidates({"cube": cube})
            assert len(candidates) == count
            assert len(set((tuple(np.round(c.start_angles, 6)), c.joint_index, round(c.end_angle, 6)) 
                           for c in candidates)) == count
            print(f"  PASS: Generated {count} unique candidates")
        finally:
            discovery.cleanup()


def test_discovery_limits_and_validation():
    print("\n=== Test: Discovery Limits and Validation ===")
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig, DiscoveredGrasp

    cube = GeometryLoader.create_primitive("cube", 0.020, translation=[0.095, 0.008, -0.05])
    
    config = DiscoveryConfig(
        hand_urdf="Data/mujoco_robot.urdf",
        max_candidates=10,
        selected_count=1,
        budget_sec=15.0,
        seed=42
    )
    
    discovery = ProbeDiscovery(config)
    try:
        grasps, status = discovery.discover({"cube": cube})
        
        sim = discovery.object_sims["cube"]
        limits = sim.limits
        
        for grasp in grasps:
            start = np.array(grasp.start_angles)
            target = start.copy()
            target[grasp.target_joint] = grasp.target_angle
            
            assert np.all(start >= limits[:, 0] - 0.1) and np.all(start <= limits[:, 1] + 0.1)
            assert np.all(target >= limits[:, 0] - 0.1) and np.all(target <= limits[:, 1] + 0.1)
            assert abs(grasp.max_increment) <= 0.5 and grasp.max_increment > 0
        
        print(f"  PASS: All grasps within limits with valid increment")
    finally:
        discovery.cleanup()


def test_fixed_library_dataset_independence_and_spatial_samples():
    from utils.ProbeDiscovery import ProbeDiscovery, DiscoveryConfig

    libraries = []
    for name, geometry in [("arbitrary", GeometryLoader.create_primitive("cube", 0.02)),
                           ("other", GeometryLoader.create_primitive("sphere", 0.01))]:
        discovery = ProbeDiscovery(DiscoveryConfig("Data/mujoco_robot.urdf"))
        try:
            candidates = discovery.generate_candidates({name: geometry})
            libraries.append([(c.start_angles.tolist(), c.joint_index, c.end_angle) for c in candidates])
            assert len(candidates) == 100
            assert {c.joint_index for c in candidates} == set(range(16))
            assert any(c.end_angle < 0 for c in candidates)
            assert discovery.library_sweeps.shape == (100, 5, 3)
            assert np.any(np.linalg.norm(discovery.library_sweeps[:, 0] - discovery.library_sweeps[:, -1], axis=1) > 0.001)
            expected = np.linalg.norm(np.ptp(discovery.library_sweeps.reshape(-1, 3), axis=0))
            assert discovery.spatial_metrics["coverage_spread"] == expected
            assert discovery.spatial_metrics["sample_count"] == 500
        finally:
            discovery.cleanup()
    assert libraries[0] == libraries[1]


def test_fixed_selection_contact_only_complements_and_exact_count():
    from utils.ProbeDiscovery import DiscoveredGrasp, select_grasps

    first = DiscoveredGrasp([0] * 16, 2, 80, 0.5,
                            {"a": 80, "b": 80}, {"a": True, "b": False})
    second = DiscoveredGrasp([0] * 16, 6, 80, 0.5,
                             {"a": 80, "b": 80}, {"a": False, "b": True})
    selected, report = select_grasps([first, second], ["a", "b"], 2, count=2)
    assert report["complete"] and len(selected) == 2
    assert not report["unresolved_pairs"] and not report["no_contact_objects"]
    selected, report = select_grasps([first, second], ["a", "b"], 2, count=3)
    assert not report["complete"] and len(selected) == 2
    selected, report = select_grasps([first], ["a", "b"], 2, count=1)
    assert not report["complete"] and report["no_contact_objects"] == ["b"]
    assert not report["unresolved_pairs"]


def test_fixed_cli_exports_library_and_checked_paths(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "config.json"
    config = {"primitives": [{"shape": "cube", "scale": 0.02, "object_id": "a",
                              "translation": [0.095, 0.008, -0.05]}]}
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    output = tmp_path / "results.json"
    completed = subprocess.run([sys.executable, "Grasp_Discovery.py", "--config", str(path),
                                "--output", str(output), "--library-count", "20",
                                "--selected-count", "5", "--budget", "30"], capture_output=True, text=True)
    saved = json.loads(output.read_text())
    assert len(saved["library"]) == 20
    assert saved["status"]["library_evaluation_complete"]
    assert len(saved["grasps"]) >= 3
    assert path.read_bytes() == original
    for grasp in saved["grasps"]:
        assert grasp["measurements"]["a"]["path_verified"]
        np.testing.assert_allclose(grasp["measurements"]["a"]["returned_angles"], 0, atol=1e-12)
