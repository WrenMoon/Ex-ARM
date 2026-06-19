from utils.LeapHand import LeapHand
from utils.SimHand import SimHand


class ExArm:

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

    # ==================================================
    # Explicit Access
    # ==================================================

    def get_real(self):
        return self.real

    def get_sim(self):
        return self.sim

    # ==================================================
    # Common Functions
    # ==================================================

    def set_goal_positions_degree(self, positions):

        if self.real:
            self.real.set_goal_positions_degree(
                positions
            )

        if self.sim:
            self.sim.set_goal_positions_degree(
                positions
            )

    def get_state(self):

        if self.mode == "real":
            return self.real.get_state()

        if self.mode == "sim":
            return self.sim.get_state()

        # BOTH MODE
        return {
            "real": self.real.get_state(),
            "sim": self.sim.get_state()
        }

    def reset(self):

        if self.real and hasattr(self.real, "reset"):
            self.real.reset()

        if self.sim and hasattr(self.sim, "reset"):
            self.sim.reset()

    def close(self):

        if self.real and hasattr(self.real, "close_port"):
            self.real.close_port()

        if self.sim and hasattr(self.sim, "close"):
            self.sim.close()

    # ==================================================
    # Automatic Function Forwarding
    # ==================================================

    def __getattr__(self, name):

        def wrapper(*args, **kwargs):

            results = {}

            if self.real and hasattr(self.real, name):
                results["real"] = getattr(
                    self.real,
                    name
                )(*args, **kwargs)

            if self.sim and hasattr(self.sim, name):
                results["sim"] = getattr(
                    self.sim,
                    name
                )(*args, **kwargs)

            if self.mode == "real":
                return results.get("real")

            if self.mode == "sim":
                return results.get("sim")

            return results

        return wrapper