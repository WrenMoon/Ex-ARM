import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import time
from dataclasses import dataclass, field, asdict
from .GeometryLoader import GeometryLoader
from .GraspSimulator import GraspSimulator, GraspSpec, ProbeResult


@dataclass
class DiscoveredGrasp:
    start_angles: List[float]
    target_joint: int
    target_angle: float
    max_increment: float
    per_object_angles: Dict[str, float] = field(default_factory=dict)
    per_object_contact: Dict[str, bool] = field(default_factory=dict)
    measurements: Dict = field(default_factory=dict)


def select_grasps(candidates, object_ids, separation, count=None):
    pairs = {(a, b) for i, a in enumerate(object_ids) for b in object_ids[i + 1:]}
    required = {("pair", a, b) for a, b in pairs} | {("contact", a) for a in object_ids}
    coverage = []
    
    for grasp in candidates:
        covered = set()
        for a, b in pairs:
            contact_diff = grasp.per_object_contact[a] != grasp.per_object_contact[b]
            angle_gap = abs(grasp.per_object_angles[a] - grasp.per_object_angles[b]) >= separation
            if contact_diff or angle_gap:
                covered.add(("pair", a, b))
        for a in object_ids:
            if grasp.per_object_contact[a]:
                covered.add(("contact", a))
        coverage.append(covered)
    
    selected = []
    missing = required.copy()
    
    if count is None:
        while missing:
            best = max(range(len(candidates)), key=lambda i: len(coverage[i] & missing), default=None)
            if best is None or not coverage[best] & missing:
                break
            selected.append(best)
            missing -= coverage[best]
        for index in selected[::-1]:
            other = set().union(*(coverage[i] for i in selected if i != index))
            if required - other == missing:
                selected.remove(index)
    else:
        while len(selected) < count and missing:
            best = max(range(len(candidates)), key=lambda i: len(coverage[i] & missing), default=None)
            if best is None or not coverage[best] & missing:
                break
            selected.append(best)
            missing -= coverage[best]
        while len(selected) < count:
            best = max((i for i in range(len(candidates)) if i not in selected),
                      key=lambda i: len(coverage[i]), default=None)
            if best is None:
                break
            selected.append(best)
    
    complete = (not missing) and (count is None or len(selected) == count)
    
    return [candidates[i] for i in selected], {
        "complete": complete,
        "unresolved_pairs": [list(item[1:]) for item in sorted(missing) if item[0] == "pair"],
        "no_contact_objects": [item[1] for item in sorted(missing) if item[0] == "contact"],
    }


@dataclass
class DiscoveryConfig:
    hand_urdf: str
    max_candidates: int = 100
    selected_count: int = 5
    budget_sec: float = 180.0
    min_angle_separation: float = 2.0
    max_increment: float = 0.5
    seed: Optional[int] = 0

    def __post_init__(self):
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            raise ValueError("Library count must be a positive integer")
        if not isinstance(self.selected_count, int) or self.selected_count < 1:
            raise ValueError("Selected count must be a positive integer")
        if self.selected_count > self.max_candidates:
            raise ValueError("Selected count must be <= library count")
        if not np.isfinite([self.budget_sec, self.min_angle_separation, self.max_increment]).all():
            raise ValueError("Discovery settings must be finite")
        if self.budget_sec <= 0 or self.min_angle_separation <= 0 or not 0 < self.max_increment <= 0.5:
            raise ValueError("Budgets and separation must be positive; increment must be <= 0.5")


class ProbeDiscovery:
    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.candidates = []
        self.object_sims = {}
        self.start_time = None
        self.swept_points_cache = {}

    def _compute_swept_points(self, sim: GraspSimulator, candidate: GraspSpec, n_samples: int = 5) -> np.ndarray:
        joint = candidate.joint_index
        geoms = [i for i in range(sim.model.ngeom)
                 if sim.finger_bodies.get(int(sim.model.geom_bodyid[i])) == joint // 4]
        tip = geoms[-1]
        origin = sim.get_angles()
        points = []
        try:
            for fraction in np.linspace(0, 1, n_samples):
                angles = candidate.start_angles.copy()
                angles[joint] += (candidate.end_angle - angles[joint]) * fraction
                sim.set_angles(angles)
                points.append(sim.data.geom_xpos[tip].copy())
        finally:
            sim.set_angles(origin)
        return np.asarray(points)
    
    def _compute_spatial_diversity(self, swept_points_list: List[np.ndarray]) -> Dict[str, float]:
        points = np.vstack(swept_points_list)
        descriptors = np.asarray(swept_points_list).reshape(len(swept_points_list), -1)
        distances = np.linalg.norm(descriptors[:, None] - descriptors[None, :], axis=2) / np.sqrt(5)
        pairs = distances[np.triu_indices(len(descriptors), 1)]
        return {
            "min_distance": float(pairs.min()) if pairs.size else 0.0,
            "mean_distance": float(pairs.mean()) if pairs.size else 0.0,
            "coverage_spread": float(np.linalg.norm(np.ptp(points, axis=0))),
        }

    def _generate_base_candidates(self, limits):
        order = [5, 9, 12, 1, 6, 10, 13, 2, 7, 11, 14, 3, 15, 0, 4, 8]
        for joint in order:
            for direction in [1, 0]:
                target = limits[joint, direction]
                if abs(target) > 1e-6:
                    yield np.zeros(16), joint, float(target)
        for fraction in [0.15, 0.30, 0.50, 0.70]:
            for offset in [1, 2, 3]:
                for joint in order:
                    sibling = joint // 4 * 4 + (joint % 4 + offset) % 4
                    for direction in [1, 0]:
                        start = np.zeros(16)
                        start[sibling] = limits[sibling, direction] * fraction
                        yield start, joint, float(limits[joint, direction])

    def generate_candidates(self, geometries: Dict[str, any]) -> List[GraspSpec]:
        if not self.object_sims and geometries:
            obj_id, geometry = next(iter(geometries.items()))
            self.object_sims[obj_id] = GraspSimulator(self.config.hand_urdf, geometry)
        reference = GeometryLoader.create_primitive("sphere", 0.001, translation=[10, 10, 10])
        sim = GraspSimulator(self.config.hand_urdf, reference)
        limits = sim.limits
        candidates = []
        seen = set()
        pool_size = 4 * self.config.max_candidates
        rng = np.random.default_rng(self.config.seed)
        for start, joint, target in self._generate_base_candidates(limits):
            if len(candidates) >= pool_size:
                break
            key = (tuple(np.round(start, 6)), joint, round(float(target), 6))
            if key not in seen:
                seen.add(key)
                candidates.append(GraspSpec(start.copy(), joint_index=joint, end_angle=target,
                                           max_increment=self.config.max_increment))
        
        while len(candidates) < pool_size:
            joint = int(rng.integers(0, 16))
            base = joint // 4 * 4
            
            start = np.zeros(16)
            for sibling in range(base, base + 4):
                if sibling != joint and rng.random() < 0.3:
                    frac = rng.uniform(0.1, 0.9)
                    start[sibling] = limits[sibling, 0] + (limits[sibling, 1] - limits[sibling, 0]) * frac
            
            frac = rng.uniform(0.0, 1.0)
            target = limits[joint, 0] + (limits[joint, 1] - limits[joint, 0]) * frac
            target = float(target)
            
            key = (tuple(np.round(start, 6)), joint, round(target, 6))
            if key not in seen:
                seen.add(key)
                candidates.append(GraspSpec(start.copy(), joint_index=joint, end_angle=target,
                                           max_increment=self.config.max_increment))
        
        try:
            sweeps = np.asarray([self._compute_swept_points(sim, c) for c in candidates])
            descriptors = sweeps.reshape(len(candidates), -1)
            chosen = list(range(min(16, self.config.max_candidates)))
            distance = np.full(len(candidates), np.inf)
            for index in chosen:
                distance = np.minimum(distance, np.linalg.norm(descriptors - descriptors[index], axis=1))
            distance[chosen] = -1
            while len(chosen) < self.config.max_candidates:
                index = int(np.argmax(distance))
                chosen.append(index)
                distance = np.minimum(distance, np.linalg.norm(descriptors - descriptors[index], axis=1))
                distance[chosen] = -1
            self.candidates = [candidates[i] for i in chosen]
            self.library_sweeps = sweeps[chosen]
            self.spatial_metrics = self._compute_spatial_diversity(self.library_sweeps)
            self.spatial_metrics["sample_count"] = int(self.library_sweeps.size // 3)
            return self.candidates
        finally:
            sim.close()

    def test_candidate(self, candidate: GraspSpec, geometries: Dict[str, any]) -> Tuple[bool, Dict]:
        results = {}
        for obj_id, geom in geometries.items():
            if obj_id not in self.object_sims:
                try:
                    self.object_sims[obj_id] = GraspSimulator(
                        hand_urdf_path=self.config.hand_urdf,
                        object_geom=geom,
                        enable_viewer=False
                    )
                except Exception as e:
                    results[obj_id] = {"success": False, "reason": f"sim_init: {str(e)}"}
                    return False, results

            if self.start_time is not None and time.monotonic() - self.start_time >= self.config.budget_sec:
                return False, {obj_id: {"reason": "time_budget"}}
            sim = self.object_sims[obj_id]
            try:
                sim.set_angles(np.zeros(16))
                sim.move_clear(candidate.start_angles)
                steps = int(np.ceil(abs(candidate.end_angle - candidate.start_angles[candidate.joint_index])
                                    / candidate.max_increment)) + 2
                probe_result = sim.execute_probe(candidate, max_steps=steps)
                if probe_result.termination_reason not in ("contact", "angle_limit"):
                    raise ValueError("Incomplete probe")
                returned = sim.retract_hand(candidate.start_angles, probe_result.final_angles)
                if not np.allclose(returned, 0, atol=1e-12, rtol=0) or sim.data.time != 0:
                    raise ValueError("Failed geometry-only return to zero")
                if sim.contact_fingers():
                    raise ValueError("Object overlap at zero")
                results[obj_id] = {
                    "success": True, "contact": bool(probe_result.contact_detected),
                    "angle": float(probe_result.achieved_angle[candidate.joint_index]),
                    "joint_status": probe_result.joint_status,
                    "final_angles": probe_result.final_angles.tolist(),
                    "returned_angles": returned.tolist(), "path_verified": True,
                }
            except ValueError as e:
                results[obj_id] = {"success": False, "reason": str(e)}
                return False, results
            finally:
                sim.set_angles(np.zeros(16))

        return True, results

    def discover(self, geometries: Dict[str, any]) -> Tuple[List[DiscoveredGrasp], Dict]:
        if not geometries:
            return [], {"complete": False, "reason": "no_objects"}

        self.cleanup()
        self.object_sims = {}
        self.swept_points_cache = {}
        self.rng = np.random.default_rng(self.config.seed)
        self.start_time = time.monotonic()
        
        for obj_id, geometry in geometries.items():
            self.object_sims[obj_id] = GraspSimulator(self.config.hand_urdf, geometry)
        
        candidates = self.generate_candidates(geometries)
        library_size = len(candidates)
        
        valid = []
        tried = 0
        rejected_reasons = {}
        
        for candidate in candidates:
            if time.monotonic() - self.start_time >= self.config.budget_sec:
                break
            
            success, results = self.test_candidate(candidate, geometries)
            if any(res.get("reason") == "time_budget" for res in results.values()):
                break
            tried += 1
            
            if success:
                grasp = DiscoveredGrasp(
                    start_angles=candidate.start_angles.tolist(),
                    target_joint=candidate.joint_index,
                    target_angle=candidate.end_angle,
                    max_increment=candidate.max_increment,
                    per_object_angles={obj: res["angle"] for obj, res in results.items()},
                    per_object_contact={obj: res["contact"] for obj, res in results.items()},
                    measurements=results,
                )
                valid.append(grasp)
            elif not success:
                for obj, res in results.items():
                    if "reason" in res:
                        reason = res["reason"]
                        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
        
        elapsed = time.monotonic() - self.start_time
        library_evaluation_complete = (tried >= library_size)
        
        selected, selection_report = select_grasps(
            valid, list(geometries), self.config.min_angle_separation,
            count=self.config.selected_count
        )
        
        if library_evaluation_complete and not selection_report["complete"]:
            if len(valid) < self.config.selected_count:
                reason = "insufficient_valid"
            else:
                reason = "unresolved"
        elif not library_evaluation_complete:
            reason = "time_budget"
        elif selection_report["complete"]:
            reason = "distinguished"
        else:
            reason = "unresolved"
        
        report = {
            "complete": selection_report["complete"],
            "reason": reason,
            "tried": tried,
            "library_size": library_size,
            "library_evaluation_complete": library_evaluation_complete,
            "valid_count": len(valid),
            "selected_count": len(selected),
            "elapsed_sec": elapsed,
            "unresolved_pairs": selection_report.get("unresolved_pairs", []),
            "no_contact_objects": selection_report.get("no_contact_objects", []),
            "rejected_reasons": rejected_reasons,
            "spatial_metrics": self.spatial_metrics,
            "limitations": ["Sampled fingertip sweep diversity is not full workspace coverage",
                            "Library candidates require dataset-specific path verification",
                            "Contact flags and stopping angles are simulation observables, not hardware accuracy",
                            "Existing baseline self-collision pairs are tolerated by the simulator",
                            "Greedy fixed-count selection is not guaranteed globally optimal"],
            "baseline_self_collision_pairs": {
                obj: [list(pair) for pair in sorted(sim.baseline_self_depths.keys())]
                for obj, sim in self.object_sims.items()
            },
        }
        
        return selected, report

    def cleanup(self):
        for sim in self.object_sims.values():
            sim.close()
