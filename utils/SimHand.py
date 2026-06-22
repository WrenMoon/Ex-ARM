"""
SimHand.py — MuJoCo simulation backend for the LEAP Hand.

Wraps a MuJoCo model loaded from a URDF/XML file and provides the same
control API as LeapHand (set_goal_positions_degree, get_state, reset, close)
so that ExArm can use either backend interchangeably.

The simulation viewer is launched in passive (non-blocking) mode and synced
after every state change. Optional marker spheres can be rendered in the
viewer to visualise IK target positions.
"""

import mujoco
import mujoco.viewer
import numpy as np


class SimHand:
    """
    MuJoCo-backed LEAP Hand simulator.

    Handles the mismatch between the logical joint ordering used by the rest
    of the codebase and the ordering that the URDF imposes in MuJoCo's qpos.
    """

    # Permutation: logical index → MuJoCo qpos index.
    # The URDF joint ordering differs from the logical [index, middle, ring, thumb]
    # convention, so this array re-routes each logical joint to its qpos slot.
    SIM_TO_REAL = np.array([
        1, 0, 2, 3,
        5, 4, 6, 7,
        9, 8, 10, 11,
        12, 13, 14, 15
    ])

    # Inverse permutation: MuJoCo qpos index → logical index (used when reading back)
    REAL_TO_SIM = np.argsort(SIM_TO_REAL)

    def __init__(self, model_path):
        """
        Load the MuJoCo model and launch the passive viewer.

        Parameters
        ----------
        model_path : str — path to the URDF or MuJoCo XML file.
        """
        self.model  = mujoco.MjModel.from_xml_path(model_path)
        self.data   = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Holds (N, 3) marker positions for optional IK target visualisation
        self.custom_marker_positions = np.zeros((0, 3), dtype=np.float64)

        self.update_viewer()

    # ------------------------------------------------------------------ #
    # Internal viewer update                                               #
    # ------------------------------------------------------------------ #

    def update_viewer(self):
        """
        Run a forward dynamics pass, redraw any custom markers, and sync
        the viewer so the latest state is visible.
        """
        mujoco.mj_forward(self.model, self.data)
        self._draw_custom_markers()
        if self.viewer.is_running():
            self.viewer.sync()

    # ------------------------------------------------------------------ #
    # Goal position                                                        #
    # ------------------------------------------------------------------ #

    def set_goal_positions_degree(self, q_deg):
        """
        Set joint positions directly in MuJoCo's qpos (degrees → radians).

        Note: MuJoCo simulation does not model dynamics here; positions are
        set instantaneously (kinematic mode).

        Parameters
        ----------
        q_deg : (16,) array-like — joint angles in degrees, *logical* order.
        """
        q_deg   = np.asarray(q_deg)
        q_rad   = np.radians(q_deg)
        # Reorder from logical → MuJoCo qpos order
        sim_q   = q_rad[self.REAL_TO_SIM]
        self.data.qpos[:] = sim_q
        self.update_viewer()

    def set_goal_positions_radian(self, q_rad):
        """
        Set joint positions in radians (logical order).

        Parameters
        ----------
        q_rad : (16,) array-like — joint angles in radians, *logical* order.
        """
        q_rad = np.asarray(q_rad)
        sim_q = q_rad[self.REAL_TO_SIM]
        self.data.qpos[:] = sim_q
        self.update_viewer()

    # ------------------------------------------------------------------ #
    # State reading                                                        #
    # ------------------------------------------------------------------ #

    def get_positions_radian(self):
        """
        Read current joint positions from qpos and return in logical order (radians).
        """
        sim_q = self.data.qpos.copy()
        return sim_q[self.SIM_TO_REAL]

    def get_positions_degree(self):
        """Return current joint positions in degrees (logical order)."""
        return np.degrees(self.get_positions_radian())

    def get_velocities(self):
        """Return current joint velocities in logical order (raw MuJoCo qvel units)."""
        sim_v = self.data.qvel.copy()
        return sim_v[self.SIM_TO_REAL]

    def get_currents(self):
        """
        Return a zero current array.

        MuJoCo does not model motor currents; this stub keeps the API
        consistent with LeapHand so ExArm can treat both backends uniformly.
        """
        return np.zeros(16)

    def get_state(self):
        """
        Return (positions_deg, velocities, currents) in logical joint order.

        Mirrors the LeapHand.get_state() signature for backend interoperability.
        """
        return (
            self.get_positions_degree(),
            self.get_velocities(),
            self.get_currents()
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset the simulation to its initial state and refresh the viewer."""
        mujoco.mj_resetData(self.model, self.data)
        self.update_viewer()

    def close(self):
        """Close the MuJoCo viewer window if it is still open."""
        if self.viewer.is_running():
            self.viewer.close()

    # ------------------------------------------------------------------ #
    # Custom marker rendering                                              #
    # ------------------------------------------------------------------ #

    def set_custom_markers(self, positions):
        """
        Store marker positions and trigger a viewer redraw.

        Used to render IK fingertip targets as small red spheres in the sim,
        giving a live visual comparison between commanded and actual fingertips.

        Parameters
        ----------
        positions : (N, 3) array-like — marker XYZ coordinates in world frame (metres).
        """
        self.custom_marker_positions = np.asarray(positions, dtype=np.float64)
        self.update_viewer()

    def _draw_custom_markers(self):
        """
        Inject custom sphere geometries into the viewer's user scene buffer.

        Each marker is drawn as a 5 mm radius red sphere. The buffer is
        cleared and rebuilt every frame to avoid accumulating stale markers.
        Silently no-ops on MuJoCo versions that do not expose user_scn.
        """
        if not hasattr(self.viewer, "user_scn"):
            print("[SimHand] viewer has no user_scn attribute in this mujoco version.")
            return

        scn       = self.viewer.user_scn
        scn.ngeom = 0

        marker_size = np.array([0.005, 0.0, 0.0], dtype=np.float64)
        marker_mat  = np.eye(3).flatten()

        for pos in self.custom_marker_positions:
            if scn.ngeom >= scn.maxgeom:
                break

            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                marker_size,
                pos,
                marker_mat,
                np.array([1.0, 0.0, 0.0, 0.85], dtype=np.float64)   # RGBA: opaque red
            )
            scn.ngeom += 1