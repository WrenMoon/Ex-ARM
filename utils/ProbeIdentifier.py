import numpy as np


class ProbeIdentifier:
    """Match ordered probe observations using a bounded angular error model."""

    def __init__(self, results, tolerance_deg=0.5):
        if not np.isfinite(tolerance_deg) or tolerance_deg < 0:
            raise ValueError("Tolerance must be finite and nonnegative")
        self.tolerance = float(tolerance_deg)
        self.object_ids = list(results["objects"])
        grasps = results["grasps"]
        if not self.object_ids or not grasps or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("Reference must contain unique objects and at least one probe")
        self.probe_count = len(grasps)
        self.angles = np.array([[g["per_object_angles"][obj] for g in grasps]
                                for obj in self.object_ids], dtype=float)
        contacts = [[g["per_object_contact"][obj] for g in grasps] for obj in self.object_ids]
        if not np.isfinite(self.angles).all() or any(type(c) is not bool for row in contacts for c in row):
            raise ValueError("Reference requires finite angles and boolean contact flags")
        self.contacts = np.array(contacts, dtype=bool)

    def identify(self, observations):
        if len(observations) != self.probe_count:
            raise ValueError(f"Expected {self.probe_count} observations in saved probe order")
        angles = np.array([o["angle"] for o in observations], dtype=float)
        contacts = [o["contact"] for o in observations]
        if angles.shape != (self.probe_count,) or not np.isfinite(angles).all():
            raise ValueError("Observed angles must be finite scalars")
        if any(type(c) is not bool for c in contacts):
            raise ValueError("Observed contact flags must be booleans")
        residuals = np.max(np.abs(self.angles - angles), axis=1)
        compatible = np.all(self.contacts == contacts, axis=1) & (residuals <= self.tolerance)
        indices = np.flatnonzero(compatible)
        candidates = [{"object_id": self.object_ids[i], "max_angle_error_deg": float(residuals[i])}
                      for i in indices]
        status = "identified" if len(indices) == 1 else "ambiguous" if len(indices) else "no_match"
        return {"status": status,
                "object_id": candidates[0]["object_id"] if status == "identified" else None,
                "candidates": candidates, "tolerance_deg": self.tolerance}
