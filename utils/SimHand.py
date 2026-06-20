import mujoco
import mujoco.viewer
import numpy as np


class SimHand:
    SIM_TO_REAL = np.array([
        1, 0, 2, 3,
        5, 4, 6, 7,
        9, 8, 10, 11,
        12, 13, 14, 15
    ])

    REAL_TO_SIM = np.argsort(SIM_TO_REAL)

    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.custom_marker_positions = np.zeros((0, 3), dtype=np.float64)

        self.update_viewer()

    def update_viewer(self):
        mujoco.mj_forward(self.model, self.data)
        self._draw_custom_markers()
        if self.viewer.is_running():
            self.viewer.sync()

    def set_goal_positions_degree(self, q_deg):
        q_deg = np.asarray(q_deg)
        q_rad = np.radians(q_deg)
        sim_q = q_rad[self.REAL_TO_SIM]
        self.data.qpos[:] = sim_q
        self.update_viewer()

    def set_goal_positions_radian(self, q_rad):
        q_rad = np.asarray(q_rad)
        sim_q = q_rad[self.REAL_TO_SIM]
        self.data.qpos[:] = sim_q
        self.update_viewer()

    def get_positions_radian(self):
        sim_q = self.data.qpos.copy()
        return sim_q[self.SIM_TO_REAL]

    def get_positions_degree(self):
        return np.degrees(self.get_positions_radian())

    def get_velocities(self):
        sim_v = self.data.qvel.copy()
        return sim_v[self.SIM_TO_REAL]

    def get_currents(self):
        return np.zeros(16)

    def get_state(self):
        return (
            self.get_positions_degree(),
            self.get_velocities(),
            self.get_currents()
        )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.update_viewer()

    def close(self):
        if self.viewer.is_running():
            self.viewer.close()

    def set_custom_markers(self, positions):
        """
        positions: (N, 3) array in world coordinates.
        """
        self.custom_marker_positions = np.asarray(positions, dtype=np.float64)
        self.update_viewer()

    def _draw_custom_markers(self):
        """
        Draw custom spheres into the viewer's user scene.
        """
        if not hasattr(self.viewer, "user_scn"):
            print("[SimHand] viewer has no user_scn attribute in this mujoco version.")
            return

        scn = self.viewer.user_scn
        scn.ngeom = 0

        marker_size = np.array([0.005, 0.0, 0.0], dtype=np.float64)
        marker_mat = np.eye(3).flatten()

        for pos in self.custom_marker_positions:
            if scn.ngeom >= scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                marker_size,
                pos,
                marker_mat,
                np.array([1.0, 0.0, 0.0, 0.85], dtype=np.float64)
            )
            scn.ngeom += 1