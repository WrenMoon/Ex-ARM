"""
LeapKinematics.py
-----------------
Forward and Inverse Kinematics for the LEAP Hand.

Joint ordering (logical / real-hand space, 16 joints):
  Index  : 0=MCP_abduct, 1=MCP_flex, 2=PIP, 3=DIP
  Middle : 4=MCP_abduct, 5=MCP_flex, 6=PIP, 7=DIP
  Ring   : 8=MCP_abduct, 9=MCP_flex, 10=PIP, 11=DIP
  Thumb  : 12=base_rot,  13=MCP,     14=PIP, 15=DIP

All public methods accept/return angles in RADIANS unless
the method name ends in _degree.

FK returns fingertip positions in the palm frame (metres).
IK accepts a target fingertip position and returns joint
angles for that finger.
"""

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Joint limits (radians) — extracted directly from the URDF
# ---------------------------------------------------------------------------

JOINT_LIMITS = np.array([
    # Index
    [-1.047,  1.047],   # 0  MCP abduction
    [-0.314,  2.230],   # 1  MCP flexion
    [-0.506,  1.885],   # 2  PIP
    [-0.366,  2.042],   # 3  DIP
    # Middle
    [-1.047,  1.047],   # 4  MCP abduction
    [-0.314,  2.230],   # 5  MCP flexion
    [-0.506,  1.885],   # 6  PIP
    [-0.366,  2.042],   # 7  DIP
    # Ring
    [-1.047,  1.047],   # 8  MCP abduction
    [-0.314,  2.230],   # 9  MCP flexion
    [-0.506,  1.885],   # 10 PIP
    [-0.366,  2.042],   # 11 DIP
    # Thumb
    [-2.443,  0.470],   # 12 base rotation
    [-2.094,  0.349],   # 13 MCP
    [-1.200,  1.900],   # 14 PIP
    [-1.340,  1.880],   # 15 DIP
])

# ---------------------------------------------------------------------------
# Finger base transforms in the palm frame (from URDF joint origins)
# Each entry: (xyz translation [m], rpy [rad]) of the MCP-flex joint
# relative to palm_lower_left.
# ---------------------------------------------------------------------------

# Helper: build a 4x4 homogeneous transform from xyz + rpy (URDF convention)
def _rpy_to_rot(r, p, y):
    """Roll-Pitch-Yaw → 3x3 rotation matrix (URDF: Rz·Ry·Rx)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def _tf(xyz, rpy):
    """Build 4×4 homogeneous transform."""
    T = np.eye(4)
    T[:3,:3] = _rpy_to_rot(*rpy)
    T[:3, 3] = xyz
    return T

def _rot_z(theta):
    """4×4 rotation about local Z axis (used for revolute joints)."""
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3,:3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return T

# ---------------------------------------------------------------------------
# Finger kinematic parameters
# Extracted from URDF joint origins for each finger chain.
#
# Each finger has the chain:
#   palm → [joint_flex] → mcp_joint → [joint_abduct] → pip → [joint_pip]
#           → dip → [joint_dip] → fingertip
#
# URDF joint ordering per finger:
#   joint N+1 : palm → mcp_joint   (MCP flexion,  axis z=-1)
#   joint N+0 : mcp_joint → pip    (MCP abduction, axis z=-1)
#   joint N+2 : pip → dip          (PIP flexion,   axis z=-1)
#   joint N+3 : dip → fingertip    (DIP flexion,   axis z=-1)
#
# Logical ordering we expose: [abduct, flex, pip, dip]
# ---------------------------------------------------------------------------

# --- Index finger (joints 0,1,2,3) ---
# URDF joint 1: palm → mcp_joint
_IDX_J1_xyz = np.array([-0.082574224938, -0.085740011501, 0.007800888584])
_IDX_J1_rpy = np.array([ 1.5707963268,    1.5707963268,   0.0])
# URDF joint 0: mcp_joint → pip
_IDX_J0_xyz = np.array([-0.012200005979,  0.038100001921, 0.014499998512])
_IDX_J0_rpy = np.array([-1.5707963268,    0.0,            1.5707963268])
# URDF joint 2: pip → dip
_IDX_J2_xyz = np.array([ 0.014999961281,  0.014299993272,-0.013000025424])
_IDX_J2_rpy = np.array([ 1.5707963268,   -1.5707963268,   0.0])
# URDF joint 3: dip → fingertip
_IDX_J3_xyz = np.array([ 0.0,            -0.036100004210, 0.000200006124])
_IDX_J3_rpy = np.array([ 0.0,             0.0,            0.0])
# Fingertip offset (visual origin used as tip position)
_IDX_TIP_xyz = np.array([0.013286424109, -0.006114238387, 0.014499995133])

# --- Middle finger (joints 4,5,6,7) ---
_MID_J5_xyz = np.array([-0.082574224939, -0.040290011501, 0.007800888584])
_MID_J5_rpy = np.array([ 1.5707963268,    1.5707963268,   0.0])
_MID_J4_xyz = np.array([-0.012200002509,  0.038099998804, 0.014499990010])
_MID_J4_rpy = np.array([-1.5707963268,    0.0,            1.5707963268])
_MID_J6_xyz = np.array([ 0.014999964750,  0.014299998633,-0.013000025424])
_MID_J6_rpy = np.array([ 1.5707963268,   -1.5707963268,   0.0])
_MID_J7_xyz = np.array([ 0.0,            -0.036099998752, 0.000200007512])
_MID_J7_rpy = np.array([ 0.0,             0.0,            0.0])
_MID_TIP_xyz = np.array([0.013286424109, -0.006114238387, 0.014499994924])

# --- Ring finger (joints 8,9,10,11) ---
_RNG_J9_xyz  = np.array([-0.082574224938,  0.005159988499, 0.007800888584])
_RNG_J9_rpy  = np.array([ 1.5707963268,    1.5707963268,   0.0])
_RNG_J8_xyz  = np.array([-0.012200002509,  0.038100001921, 0.014500002470])
_RNG_J8_rpy  = np.array([-1.5707963268,    0.0,            1.5707963268])
_RNG_J10_xyz = np.array([ 0.014999968220,  0.014299998980,-0.013000025424])
_RNG_J10_rpy = np.array([ 1.5707963268,   -1.5707963268,   0.0])
_RNG_J11_xyz = np.array([ 0.0,            -0.036099998683, 0.000200007859])
_RNG_J11_rpy = np.array([ 0.0,             0.0,            0.0])
_RNG_TIP_xyz = np.array([0.013286424109, -0.006114238387, 0.014499994959])

# --- Thumb (joints 12,13,14,15) ---
# URDF joint 13: palm → thumb_left_temp_base
_THB_J13_xyz = np.array([-0.144874224938, -0.090040011501, 0.004900888584])
_THB_J13_rpy = np.array([ 0.0,             1.5707963268,   0.0])
# URDF joint 12: thumb_left_temp_base → thumb_pip
_THB_J12_xyz = np.array([ 0.0,            -0.014100001263,-0.012999956035])
_THB_J12_rpy = np.array([-1.5707963268,   -1.5707963268,   0.0])
# URDF joint 14: thumb_pip → thumb_dip
_THB_J14_xyz = np.array([ 0.0,             0.014499996778,-0.017000042855])
_THB_J14_rpy = np.array([-1.5707963268,    0.0,            0.0])
# URDF joint 15: thumb_dip → thumb_fingertip
_THB_J15_xyz = np.array([ 0.0,             0.046599996833, 0.000200006545])
_THB_J15_rpy = np.array([ 0.0,             0.0,           -3.1415926536])
_THB_TIP_xyz = np.array([0.062559538463,   0.078459682911, 0.048992911807])


# ---------------------------------------------------------------------------
# Core FK: single finger
# ---------------------------------------------------------------------------

def _finger_fk(j0_tf, j1_tf, j2_tf, j3_tf, tip_xyz, q):
    """
    Generic 4-DOF finger FK.

    Parameters
    ----------
    j0_tf : (xyz, rpy) of joint-0 (abduction) relative to its parent
    j1_tf : (xyz, rpy) of joint-1 (flexion)   relative to palm
    j2_tf : (xyz, rpy) of joint-2 (PIP)        relative to pip link
    j3_tf : (xyz, rpy) of joint-3 (DIP)        relative to dip link
    tip_xyz : fingertip offset in fingertip link frame
    q : [q_abduct, q_flex, q_pip, q_dip]  (radians)

    Returns
    -------
    tip_pos : (3,) fingertip position in palm frame
    T_list  : list of 4×4 transforms [T_mcp, T_pip, T_dip, T_tip]
    """
    q_abduct, q_flex, q_pip, q_dip = q

    # palm → mcp_joint frame (joint 1 = MCP flexion)
    T_palm_to_j1 = _tf(*j1_tf)
    T_j1_rot     = _rot_z(-q_flex)          # axis z=-1 → negate
    T_mcp        = T_palm_to_j1 @ T_j1_rot

    # mcp_joint → pip frame (joint 0 = MCP abduction)
    T_mcp_to_j0  = _tf(*j0_tf)
    T_j0_rot     = _rot_z(-q_abduct)
    T_pip        = T_mcp @ T_mcp_to_j0 @ T_j0_rot

    # pip → dip frame (joint 2 = PIP)
    T_pip_to_j2  = _tf(*j2_tf)
    T_j2_rot     = _rot_z(-q_pip)
    T_dip        = T_pip @ T_pip_to_j2 @ T_j2_rot

    # dip → fingertip frame (joint 3 = DIP)
    T_dip_to_j3  = _tf(*j3_tf)
    T_j3_rot     = _rot_z(-q_dip)
    T_tip_frame  = T_dip @ T_dip_to_j3 @ T_j3_rot

    # fingertip position
    tip_h   = np.array([*tip_xyz, 1.0])
    tip_pos = (T_tip_frame @ tip_h)[:3]

    return tip_pos, [T_mcp, T_pip, T_dip, T_tip_frame]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LeapKinematics:
    """
    Forward and Inverse Kinematics for the LEAP Hand.

    All angles are in radians unless the method name ends in _degree.

    Finger indices
    --------------
    0 = Index, 1 = Middle, 2 = Ring, 3 = Thumb

    Joint ordering (logical, 16-element vector)
    -------------------------------------------
    [0..3]  Index  : abduct, flex, pip, dip
    [4..7]  Middle : abduct, flex, pip, dip
    [8..11] Ring   : abduct, flex, pip, dip
    [12..15] Thumb : base_rot, mcp, pip, dip
    """

    FINGER_NAMES = ["index", "middle", "ring", "thumb"]

    # Finger parameter table: (j_flex_tf, j_abduct_tf, j_pip_tf, j_dip_tf, tip_xyz)
    # Note: for thumb the "abduct" slot is the base rotation (joint 12)
    _FINGER_PARAMS = [
        # Index
        ((_IDX_J1_xyz, _IDX_J1_rpy),
         (_IDX_J0_xyz, _IDX_J0_rpy),
         (_IDX_J2_xyz, _IDX_J2_rpy),
         (_IDX_J3_xyz, _IDX_J3_rpy),
         _IDX_TIP_xyz),
        # Middle
        ((_MID_J5_xyz, _MID_J5_rpy),
         (_MID_J4_xyz, _MID_J4_rpy),
         (_MID_J6_xyz, _MID_J6_rpy),
         (_MID_J7_xyz, _MID_J7_rpy),
         _MID_TIP_xyz),
        # Ring
        ((_RNG_J9_xyz,  _RNG_J9_rpy),
         (_RNG_J8_xyz,  _RNG_J8_rpy),
         (_RNG_J10_xyz, _RNG_J10_rpy),
         (_RNG_J11_xyz, _RNG_J11_rpy),
         _RNG_TIP_xyz),
        # Thumb
        ((_THB_J13_xyz, _THB_J13_rpy),
         (_THB_J12_xyz, _THB_J12_rpy),
         (_THB_J14_xyz, _THB_J14_rpy),
         (_THB_J15_xyz, _THB_J15_rpy),
         _THB_TIP_xyz),
    ]

    def __init__(self):
        self.limits = JOINT_LIMITS  # (16, 2)

    # ------------------------------------------------------------------
    # Forward Kinematics
    # ------------------------------------------------------------------

    def fk_finger(self, finger_idx: int, q4: np.ndarray):
        """
        FK for a single finger.

        Parameters
        ----------
        finger_idx : 0=Index, 1=Middle, 2=Ring, 3=Thumb
        q4 : (4,) joint angles [abduct/base, flex/mcp, pip, dip] in radians

        Returns
        -------
        tip_pos : (3,) fingertip xyz in palm frame (metres)
        transforms : list of 4 intermediate 4×4 transforms
        """
        p = self._FINGER_PARAMS[finger_idx]
        return _finger_fk(p[1], p[0], p[2], p[3], p[4], q4)

    def fk(self, q: np.ndarray):
        """
        Full-hand FK.

        Parameters
        ----------
        q : (16,) joint angles in radians (logical ordering)

        Returns
        -------
        tips : (4, 3) fingertip positions [index, middle, ring, thumb]
               in the palm frame (metres)
        """
        q = np.asarray(q, dtype=float)
        tips = np.zeros((4, 3))
        for i in range(4):
            q4 = q[i*4:(i+1)*4]
            tips[i], _ = self.fk_finger(i, q4)
        return tips

    def fk_degree(self, q_deg: np.ndarray):
        """FK with joint angles in degrees."""
        return self.fk(np.radians(q_deg))

    def fk_finger_degree(self, finger_idx: int, q4_deg: np.ndarray):
        """FK for a single finger with joint angles in degrees."""
        return self.fk_finger(finger_idx, np.radians(q4_deg))

    def fk_all_links(self, q: np.ndarray):
        """
        FK returning all intermediate link positions (not just fingertips).

        Returns
        -------
        dict with keys 'index', 'middle', 'ring', 'thumb', each containing
        a dict with keys 'mcp', 'pip', 'dip', 'tip' → (3,) positions.
        """
        q = np.asarray(q, dtype=float)
        result = {}
        for i, name in enumerate(self.FINGER_NAMES):
            q4 = q[i*4:(i+1)*4]
            tip_pos, transforms = self.fk_finger(i, q4)
            result[name] = {
                "mcp": transforms[0][:3, 3],
                "pip": transforms[1][:3, 3],
                "dip": transforms[2][:3, 3],
                "tip": tip_pos,
            }
        return result

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------

    def ik_finger(
        self,
        finger_idx: int,
        target_pos: np.ndarray,
        q0: np.ndarray = None,
        tol: float = 1e-4,
        max_iter: int = 200,
    ):
        """
        IK for a single finger using numerical optimisation (L-BFGS-B).

        Parameters
        ----------
        finger_idx : 0=Index, 1=Middle, 2=Ring, 3=Thumb
        target_pos : (3,) desired fingertip position in palm frame (metres)
        q0         : (4,) initial joint angles in radians (optional)
        tol        : position tolerance in metres
        max_iter   : maximum optimiser iterations

        Returns
        -------
        q_sol  : (4,) joint angles in radians
        info   : dict with keys 'success', 'error_m', 'message'
        """
        target_pos = np.asarray(target_pos, dtype=float)
        lo = self.limits[finger_idx*4:(finger_idx+1)*4, 0]
        hi = self.limits[finger_idx*4:(finger_idx+1)*4, 1]
        bounds = list(zip(lo, hi))

        if q0 is None:
            q0 = (lo + hi) / 2.0   # midpoint initialisation

        q0 = np.clip(np.asarray(q0, dtype=float), lo, hi)

        def cost(q):
            tip, _ = self.fk_finger(finger_idx, q)
            return float(np.sum((tip - target_pos) ** 2))

        result = minimize(
            cost,
            q0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )

        q_sol = result.x
        tip_final, _ = self.fk_finger(finger_idx, q_sol)
        error = float(np.linalg.norm(tip_final - target_pos))

        return q_sol, {
            "success": error < tol,
            "error_m": error,
            "message": result.message,
        }

    def ik_finger_multistart(
        self,
        finger_idx: int,
        target_pos: np.ndarray,
        n_starts: int = 8,
        tol: float = 1e-4,
        max_iter: int = 200,
        seed: int = 0,
    ):
        """
        IK with multiple random restarts for robustness.

        Returns the solution with the smallest position error.
        """
        rng = np.random.default_rng(seed)
        lo = self.limits[finger_idx*4:(finger_idx+1)*4, 0]
        hi = self.limits[finger_idx*4:(finger_idx+1)*4, 1]

        best_q, best_info = None, {"error_m": np.inf, "success": False}

        for _ in range(n_starts):
            q0 = rng.uniform(lo, hi)
            q_sol, info = self.ik_finger(finger_idx, target_pos, q0, tol, max_iter)
            if info["error_m"] < best_info["error_m"]:
                best_q, best_info = q_sol, info
            if best_info["success"]:
                break

        return best_q, best_info

    def ik(
        self,
        targets: np.ndarray,
        q0: np.ndarray = None,
        n_starts: int = 4,
        tol: float = 1e-4,
        max_iter: int = 200,
    ):
        """
        Full-hand IK: solve IK for all four fingers independently.

        Parameters
        ----------
        targets : (4, 3) desired fingertip positions [index, middle, ring, thumb]
                  in palm frame (metres)
        q0      : (16,) initial joint angles in radians (optional)
        n_starts: random restarts per finger
        tol     : position tolerance in metres
        max_iter: max optimiser iterations per finger

        Returns
        -------
        q_sol  : (16,) joint angles in radians
        infos  : list of 4 info dicts (one per finger)
        """
        targets = np.asarray(targets, dtype=float)
        q_sol = np.zeros(16)
        infos = []

        for i in range(4):
            q0_i = q0[i*4:(i+1)*4] if q0 is not None else None
            q_i, info = self.ik_finger_multistart(
                i, targets[i], n_starts=n_starts, tol=tol, max_iter=max_iter
            )
            q_sol[i*4:(i+1)*4] = q_i
            infos.append(info)

        return q_sol, infos

    def ik_degree(self, targets: np.ndarray, **kwargs):
        """Full-hand IK, returns joint angles in degrees."""
        q_sol, infos = self.ik(targets, **kwargs)
        return np.degrees(q_sol), infos

    def ik_finger_degree(self, finger_idx: int, target_pos: np.ndarray, **kwargs):
        """Single-finger IK, returns joint angles in degrees."""
        q_sol, info = self.ik_finger_multistart(finger_idx, target_pos, **kwargs)
        return np.degrees(q_sol), info

    # ------------------------------------------------------------------
    # Jacobian (numerical)
    # ------------------------------------------------------------------

    def jacobian_finger(self, finger_idx: int, q4: np.ndarray, eps: float = 1e-6):
        """
        Numerical Jacobian (3×4) for a single finger's fingertip position.

        J[:, i] = ∂tip_pos / ∂q_i
        """
        q4 = np.asarray(q4, dtype=float)
        tip0, _ = self.fk_finger(finger_idx, q4)
        J = np.zeros((3, 4))
        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            tip_plus, _ = self.fk_finger(finger_idx, q4 + dq)
            J[:, i] = (tip_plus - tip0) / eps
        return J

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clip_to_limits(self, q: np.ndarray):
        """Clip a 16-element joint vector to the URDF joint limits."""
        q = np.asarray(q, dtype=float)
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])

    def is_within_limits(self, q: np.ndarray):
        """Return True if all joints are within limits."""
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))

    def print_fk(self, q: np.ndarray):
        """Pretty-print FK results for all fingers."""
        tips = self.fk(q)
        print("Fingertip positions (palm frame, metres):")
        for i, name in enumerate(self.FINGER_NAMES):
            x, y, z = tips[i]
            print(f"  {name:6s}: x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")