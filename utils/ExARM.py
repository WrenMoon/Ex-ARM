from utils.LeapHand import LeapHand
from utils.SimHand import SimHand


class ExArm:
    """
    Unified interface for controlling the LEAP Hand in real hardware mode,
    MuJoCo simulation mode, or both simultaneously.

    In 'both' mode, every command is dispatched to both backends in parallel.
    State queries return a dict with 'real' and 'sim' keys. In single-backend
    modes they return the scalar result directly.
    """

    def __init__(
        self,
        mode="real",

        # LeapHand args
        ids=None,
        port=None,
        baudrate=None,
        offsets=None,

        # Sim args
        model_path=None,
    ):
        """
        Parameters
        ----------
        mode        : 'real' | 'sim' | 'both'
        ids         : list of 16 Dynamixel servo IDs (hardware only)
        port        : serial port string, e.g. 'COM5' or '/dev/ttyUSB0' (hardware only)
        baudrate    : Dynamixel baud rate, e.g. 4000000 (hardware only)
        offsets     : (16,) per-joint angle offsets in degrees (hardware only)
        model_path  : path to the MuJoCo URDF/XML model file (sim only)
        """
        mode = mode.lower()

        if mode not in ("real", "sim", "both"):
            raise ValueError(
                "mode must be 'real', 'sim', or 'both'"
            )

        self.mode = mode

        self.real = None
        self.sim = None

        if mode in ("real", "both"):
            self.real = LeapHand(
                ids=ids,
                port=port,
                baudrate=baudrate,
                offsets=offsets
            )

        if mode in ("sim", "both"):
            self.sim = SimHand(
                model_path=model_path
            )

    # ------------------------------------------------------------------ #
    # Explicit backend accessors                                           #
    # ------------------------------------------------------------------ #

    def get_real(self):
        """Return the underlying LeapHand (hardware) instance, or None."""
        return self.real

    def get_sim(self):
        """Return the underlying SimHand (MuJoCo) instance, or None."""
        return self.sim

    # ------------------------------------------------------------------ #
    # Common control API                                                   #
    # ------------------------------------------------------------------ #

    def set_goal_positions_degree(self, positions):
        """
        Send target joint angles (degrees) to whichever backends are active.

        Parameters
        ----------
        positions : array-like of shape (16,) — one angle per joint in degrees,
                    in logical order [index(4), middle(4), ring(4), thumb(4)].
        """
        if self.real:
            self.real.set_goal_positions_degree(positions)

        if self.sim:
            self.sim.set_goal_positions_degree(positions)

    def get_state(self):
        """
        Read the current hand state from active backend(s).

        Returns
        -------
        In 'real' or 'sim' mode : (positions, velocities, currents) tuple.
        In 'both' mode          : {'real': tuple, 'sim': tuple}.
        """
        if self.mode == "real":
            return self.real.get_state()

        if self.mode == "sim":
            return self.sim.get_state()

        return {
            "real": self.real.get_state(),
            "sim": self.sim.get_state()
        }

    def reset(self):
        """Reset all active backends to their default/zero configuration."""
        if self.real and hasattr(self.real, "reset"):
            self.real.reset()

        if self.sim and hasattr(self.sim, "reset"):
            self.sim.reset()

    def close(self):
        """Gracefully shut down all active backends (close port / viewer)."""
        if self.real and hasattr(self.real, "close_port"):
            self.real.close_port()

        if self.sim and hasattr(self.sim, "close"):
            self.sim.close()

    def set_target_markers(self, targets):
        """
        Forward fingertip target marker positions to the simulation viewer.

        Renders small red spheres at the IK target positions so the user can
        visually verify calibration. No-op when running in 'real'-only mode.

        Parameters
        ----------
        targets : array-like of shape (4, 3) — target XYZ positions
                  in the LEAP palm frame (metres).
        """
        if self.sim and hasattr(self.sim, "set_target_markers"):
            return self.sim.set_target_markers(np.asarray(targets))
        return None

    # ------------------------------------------------------------------ #
    # Automatic method forwarding                                          #
    # ------------------------------------------------------------------ #

    def __getattr__(self, name):
        """
        Transparently forward any unrecognised method call to the active
        backend(s). This allows backend-specific methods (e.g. torque helpers)
        to be called directly on the ExArm instance without boilerplate.

        In 'both' mode the results are returned as {'real': ..., 'sim': ...}.
        """
        def wrapper(*args, **kwargs):

            results = {}

            if self.real and hasattr(self.real, name):
                results["real"] = getattr(self.real, name)(*args, **kwargs)

            if self.sim and hasattr(self.sim, name):
                results["sim"] = getattr(self.sim, name)(*args, **kwargs)

            if self.mode == "real":
                return results.get("real")

            if self.mode == "sim":
                return results.get("sim")

            return results

        return wrapper