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

# ------------------------------------------------------------------ #
# Joint limits (radians) — extracted directly from the LEAP Hand URDF #
# ------------------------------------------------------------------ #

JOINT_LIMITS = np.array([
    # Index finger
    [-1.047,  1.047],   # 0  MCP abduction/adduction
    [-0.314,  2.230],   # 1  MCP flexion/extension
    [-0.506,  1.885],   # 2  PIP flexion
    [-0.366,  2.042],   # 3  DIP flexion
    # Middle finger
    [-1.047,  1.047],   # 4  MCP abduction/adduction
    [-0.314,  2.230],   # 5  MCP flexion/extension
    [-0.506,  1.885],   # 6  PIP flexion
    [-0.366,  2.042],   # 7  DIP flexion
    # Ring finger
    [-1.047,  1.047],   # 8  MCP abduction/adduction
    [-0.314,  2.230],   # 9  MCP flexion/extension
    [-0.506,  1.885],   # 10 PIP flexion
    [-0.366,  2.042],   # 11 DIP flexion
    # Thumb
    [-2.443,  0.470],   # 12 base rotation (twist around palm normal)
    [-2.094,  0.349],   # 13 MCP (in/out of palm plane)
    [-1.200,  1.900],   # 14 PIP flexion
    [-1.340,  1.880],   # 15 DIP flexion
])

# ------------------------------------------------------------------ #
# Homogeneous transform helpers                                        #
# ------------------------------------------------------------------ #

def _rpy_to_rot(r, p, y):
    """
    Convert URDF roll-pitch-yaw angles to a 3×3 rotation matrix.

    URDF convention: R = Rz(yaw) · Ry(pitch) · Rx(roll)
    """
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0,  0 ], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0,  1,   0 ], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _tf(xyz, rpy):
    """
    Build a 4×4 homogeneous transform from a translation vector and RPY angles.

    Parameters
    ----------
    xyz : (3,) translation in metres
    rpy : (3,) roll, pitch, yaw in radians (URDF convention)
    """
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rot(*rpy)
    T[:3,  3] = xyz
    return T

def _rot_z(theta):
    """
    4×4 homogeneous rotation about the local Z axis.

    Used to apply each revolute joint angle along its local z-axis.
    All LEAP Hand joints rotate about z with axis direction z=-1 (see URDF),
    so callers negate θ before passing it here.
    """
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return T

# ------------------------------------------------------------------ #
# Per-finger URDF joint origins                                        #
# ------------------------------------------------------------------ #
# Each constant below is taken verbatim from the LEAP Hand URDF joint
# origins (xyz and rpy attributes).  They define the fixed transforms
# between successive link frames in the kinematic chain.
#
# Chain per finger (index / middle / ring):
#   palm → [joint_flex] → mcp_joint → [joint_abduct] → pip
#        → [joint_pip]  → dip       → [joint_dip]    → fingertip
#
# Logical joint ordering we expose: [abduct, flex, pip, dip]

# --- Index finger (logical joints 0-3) ---
_IDX_J1_xyz = np.array([-0.082574224938, -0.085740011501,  0.007800888584])
_IDX_J1_rpy = np.array([ 1.5707963268,    1.5707963268,    0.0])
_IDX_J0_xyz = np.array([-0.012200005979,  0.038100001921,  0.014499998512])
_IDX_J0_rpy = np.array([-1.5707963268,    0.0,             1.5707963268])
_IDX_J2_xyz = np.array([ 0.014999961281,  0.014299993272, -0.013000025424])
_IDX_J2_rpy = np.array([ 1.5707963268,   -1.5707963268,    0.0])
_IDX_J3_xyz = np.array([ 0.0,            -0.036100004210,  0.000200006124])
_IDX_J3_rpy = np.array([ 0.0,             0.0,             0.0])
_IDX_TIP_xyz = np.array([-0.00421, -0.0496,  0.014499994959])

# --- Middle finger (logical joints 4-7) ---
_MID_J5_xyz = np.array([-0.082574224939, -0.040290011501,  0.007800888584])
_MID_J5_rpy = np.array([ 1.5707963268,    1.5707963268,    0.0])
_MID_J4_xyz = np.array([-0.012200002509,  0.038099998804,  0.014499990010])
_MID_J4_rpy = np.array([-1.5707963268,    0.0,             1.5707963268])
_MID_J6_xyz = np.array([ 0.014999964750,  0.014299998633, -0.013000025424])
_MID_J6_rpy = np.array([ 1.5707963268,   -1.5707963268,    0.0])
_MID_J7_xyz = np.array([ 0.0,            -0.036099998752,  0.000200007512])
_MID_J7_rpy = np.array([ 0.0,             0.0,             0.0])
_MID_TIP_xyz = np.array([-0.00421, -0.0496,  0.014499994959])

# --- Ring finger (logical joints 8-11) ---
_RNG_J9_xyz  = np.array([-0.082574224938,  0.005159988499,  0.007800888584])
_RNG_J9_rpy  = np.array([ 1.5707963268,    1.5707963268,    0.0])
_RNG_J8_xyz  = np.array([-0.012200002509,  0.038100001921,  0.014500002470])
_RNG_J8_rpy  = np.array([-1.5707963268,    0.0,             1.5707963268])
_RNG_J10_xyz = np.array([ 0.014999968220,  0.014299998980, -0.013000025424])
_RNG_J10_rpy = np.array([ 1.5707963268,   -1.5707963268,    0.0])
_RNG_J11_xyz = np.array([ 0.0,            -0.036099998683,  0.000200007859])
_RNG_J11_rpy = np.array([ 0.0,             0.0,             0.0])
_RNG_TIP_xyz = np.array([-0.00421, -0.0496,  0.014499994959])

# --- Thumb (logical joints 12-15) ---
# The thumb chain differs: joint 13 (in/out of palm) is the parent,
# joint 12 (base twist) is applied second along the chain.
_THB_J13_xyz = np.array([-0.144874224938, -0.090040011501,  0.004900888584])
_THB_J13_rpy = np.array([ 0.0,             1.5707963268,    0.0])
_THB_J12_xyz = np.array([ 0.0,            -0.014100001263, -0.012999956035])
_THB_J12_rpy = np.array([-1.5707963268,   -1.5707963268,    0.0])
_THB_J14_xyz = np.array([ 0.0,             0.014499996778, -0.017000042855])
_THB_J14_rpy = np.array([-1.5707963268,    0.0,             0.0])
_THB_J15_xyz = np.array([ 0.0,             0.046599996833,  0.000200006545])
_THB_J15_rpy = np.array([ 0.0,             0.0,            -3.1415926536])
_THB_TIP_xyz = np.array([-0.005, -0.06206, -0.014])


# ------------------------------------------------------------------ #
# Generic 4-DOF finger FK                                             #
# ------------------------------------------------------------------ #

def _finger_fk(j0_tf, j1_tf, j2_tf, j3_tf, tip_xyz, q):
    """
    Compute forward kinematics for a single 4-DOF finger (index/middle/ring).

    All four joints rotate about their local z-axis with axis direction z=-1
    in the URDF, so each joint angle is negated before being passed to _rot_z.

    Parameters
    ----------
    j0_tf   : (xyz, rpy) — fixed transform from mcp_joint → pip (abduction joint)
    j1_tf   : (xyz, rpy) — fixed transform from palm → mcp_joint (flexion joint)
    j2_tf   : (xyz, rpy) — fixed transform from pip → dip (PIP joint)
    j3_tf   : (xyz, rpy) — fixed transform from dip → fingertip link (DIP joint)
    tip_xyz : (3,)       — fingertip offset in the fingertip link frame (metres)
    q       : (4,)       — [q_abduct, q_flex, q_pip, q_dip] in radians

    Returns
    -------
    tip_pos    : (3,) fingertip XYZ in the palm frame (metres)
    T_list     : [T_mcp, T_pip, T_dip, T_tip_frame] — intermediate 4×4 transforms
    """
    q_abduct, q_flex, q_pip, q_dip = q

    # Step 1: palm → mcp_joint frame (MCP flexion joint)
    T_mcp = _tf(*j1_tf) @ _rot_z(-q_flex)

    # Step 2: mcp_joint → pip frame (MCP abduction joint)
    T_pip = T_mcp @ _tf(*j0_tf) @ _rot_z(-q_abduct)

    # Step 3: pip → dip frame (PIP joint)
    T_dip = T_pip @ _tf(*j2_tf) @ _rot_z(-q_pip)

    # Step 4: dip → fingertip frame (DIP joint)
    T_tip_frame = T_dip @ _tf(*j3_tf) @ _rot_z(-q_dip)

    # Fingertip position: apply tip mesh offset in homogeneous coordinates
    tip_pos = (T_tip_frame @ np.array([*tip_xyz, 1.0]))[:3]

    return tip_pos, [T_mcp, T_pip, T_dip, T_tip_frame]


# ------------------------------------------------------------------ #
# Dedicated thumb FK                                                  #
# ------------------------------------------------------------------ #

def _thumb_fk(q):
    """
    Forward kinematics for the thumb, which has a different kinematic chain
    structure compared to the other three fingers.

    Chain:
      palm
       └─[joint 13: rpy=[0, π/2, 0]]──► thumb_left_temp_base  (q13: in/out of palm)
           └─[joint 12: rpy=[-π/2,-π/2, 0]]──► thumb_pip       (q12: axial twist)
               └─[joint 14: rpy=[-π/2, 0, 0]]──► thumb_dip     (q14: PIP flexion)
                   └─[joint 15: rpy=[0, 0, -π]]──► thumb_tip   (q15: DIP flexion)

    Parameters
    ----------
    q : (4,) — [q12, q13, q14, q15] in radians
        q[0] = q12 = axial twist
        q[1] = q13 = in/out of palm (parent joint, applied first)
        q[2] = q14 = PIP flexion
        q[3] = q15 = DIP flexion

    Returns
    -------
    tip_pos    : (3,) fingertip XYZ in palm frame (metres)
    transforms : [T_base, T_pip, T_dip, T_tip] — intermediate 4×4 transforms
    """
    q12, q13, q14, q15 = q

    # Step 1: palm → thumb_left_temp_base via joint 13 (in/out of palm)
    T_base = _tf(
        np.array([-0.144874224938, -0.090040011501, 0.004900888584]),
        np.array([ 0.0,             np.pi / 2,       0.0])
    ) @ _rot_z(-q13)

    # Step 2: thumb_left_temp_base → thumb_pip via joint 12 (axial twist)
    T_pip = T_base @ _tf(
        np.array([ 0.0, -0.014100001263, -0.012999956035]),
        np.array([-np.pi / 2, -np.pi / 2, 0.0])
    ) @ _rot_z(-q12)

    # Step 3: thumb_pip → thumb_dip via joint 14 (PIP)
    T_dip = T_pip @ _tf(
        np.array([ 0.0, 0.014499996778, -0.017000042855]),
        np.array([-np.pi / 2, 0.0, 0.0])
    ) @ _rot_z(-q14)

    # Step 4: thumb_dip → thumb_fingertip via joint 15 (DIP)
    T_tip_frame = T_dip @ _tf(
        np.array([ 0.0, 0.046599996833, 0.000200006545]),
        np.array([ 0.0, 0.0, -np.pi])
    ) @ _rot_z(-q15)

    # Apply anatomical fingertip offset within the tip link frame
    tip_pos = (T_tip_frame @ np.array([*_THB_TIP_xyz, 1.0]))[:3]

    return tip_pos, [T_base, T_pip, T_dip, T_tip_frame]


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

class LeapKinematics:
    """
    Forward and Inverse Kinematics engine for the LEAP Hand.

    All angles are in radians unless the method name ends in _degree.

    Finger indices:  0 = Index,  1 = Middle,  2 = Ring,  3 = Thumb

    Logical joint vector layout (16 elements):
      [0..3]   Index  : abduct, flex, pip, dip
      [4..7]   Middle : abduct, flex, pip, dip
      [8..11]  Ring   : abduct, flex, pip, dip
      [12..15] Thumb  : base_rot, mcp, pip, dip
    """

    FINGER_NAMES = ["index", "middle", "ring", "thumb"]

    # Lookup table of fixed URDF transforms for index/middle/ring.
    # Thumb uses _thumb_fk() directly and is not stored here.
    # Format per entry: (flex_tf, abduct_tf, pip_tf, dip_tf, tip_xyz)
    _FINGER_PARAMS = [
        # Index
        ((_IDX_J1_xyz, _IDX_J1_rpy), (_IDX_J0_xyz, _IDX_J0_rpy),
         (_IDX_J2_xyz, _IDX_J2_rpy), (_IDX_J3_xyz, _IDX_J3_rpy), _IDX_TIP_xyz),
        # Middle
        ((_MID_J5_xyz, _MID_J5_rpy), (_MID_J4_xyz, _MID_J4_rpy),
         (_MID_J6_xyz, _MID_J6_rpy), (_MID_J7_xyz, _MID_J7_rpy), _MID_TIP_xyz),
        # Ring
        ((_RNG_J9_xyz,  _RNG_J9_rpy),  (_RNG_J8_xyz,  _RNG_J8_rpy),
         (_RNG_J10_xyz, _RNG_J10_rpy), (_RNG_J11_xyz, _RNG_J11_rpy), _RNG_TIP_xyz),
        # Thumb — placeholder; fk_finger() routes to _thumb_fk() for finger_idx=3
        ((_THB_J13_xyz, _THB_J13_rpy), (_THB_J12_xyz, _THB_J12_rpy),
         (_THB_J14_xyz, _THB_J14_rpy), (_THB_J15_xyz, _THB_J15_rpy), _THB_TIP_xyz),
    ]

    def __init__(self):
        self.limits = JOINT_LIMITS  # (16, 2) — lower and upper bounds per joint

    # ------------------------------------------------------------------ #
    # Forward Kinematics                                                   #
    # ------------------------------------------------------------------ #

    def fk_finger(self, finger_idx: int, q4: np.ndarray):
        """
        FK for a single finger.

        Parameters
        ----------
        finger_idx : 0=Index, 1=Middle, 2=Ring, 3=Thumb
        q4         : (4,) joint angles in radians [abduct/base, flex/mcp, pip, dip]

        Returns
        -------
        tip_pos    : (3,) fingertip XYZ in the palm frame (metres)
        transforms : list of 4 intermediate 4×4 homogeneous transforms
        """
        if finger_idx == 3:
            return _thumb_fk(q4)

        p = self._FINGER_PARAMS[finger_idx]
        return _finger_fk(p[1], p[0], p[2], p[3], p[4], q4)

    def fk(self, q: np.ndarray):
        """
        Full-hand FK: compute all four fingertip positions at once.

        Parameters
        ----------
        q : (16,) joint angles in radians (logical ordering)

        Returns
        -------
        tips : (4, 3) fingertip XYZ array — [index, middle, ring, thumb]
               in the palm frame (metres)
        """
        q    = np.asarray(q, dtype=float)
        tips = np.zeros((4, 3))
        for i in range(4):
            tips[i], _ = self.fk_finger(i, q[i*4:(i+1)*4])
        return tips

    def fk_degree(self, q_deg: np.ndarray):
        """Full-hand FK with joint angles supplied in degrees."""
        return self.fk(np.radians(q_deg))

    def fk_finger_degree(self, finger_idx: int, q4_deg: np.ndarray):
        """Single-finger FK with joint angles supplied in degrees."""
        return self.fk_finger(finger_idx, np.radians(q4_deg))

    def fk_all_links(self, q: np.ndarray):
        """
        FK returning all intermediate link positions, not just fingertips.

        Useful for visualising the full skeleton or computing Jacobians
        at intermediate frames.

        Returns
        -------
        dict keyed by finger name ('index', 'middle', 'ring', 'thumb').
        Each value is a dict with keys 'mcp'/'base', 'pip', 'dip', 'tip'
        mapping to (3,) position arrays in the palm frame.
        """
        q      = np.asarray(q, dtype=float)
        result = {}
        for i, name in enumerate(self.FINGER_NAMES):
            q4 = q[i*4:(i+1)*4]
            tip_pos, transforms = self.fk_finger(i, q4)
            keys = ["base", "pip", "dip", "tip"] if i == 3 else ["mcp", "pip", "dip", "tip"]
            result[name] = {
                k: (transforms[j][:3, 3] if k != "tip" else tip_pos)
                for j, k in enumerate(keys)
            }
        return result

    # ------------------------------------------------------------------ #
    # Inverse Kinematics                                                   #
    # ------------------------------------------------------------------ #

    def ik_finger(
        self,
        finger_idx: int,
        target_pos: np.ndarray,
        q0: np.ndarray = None,
        tol: float = 1e-4,
        max_iter: int = 200,
    ):
        """
        Solve IK for a single finger using L-BFGS-B numerical optimisation.

        Minimises squared Euclidean distance between the FK fingertip and
        the target position, subject to URDF joint limits.

        Parameters
        ----------
        finger_idx : 0=Index, 1=Middle, 2=Ring, 3=Thumb
        target_pos : (3,) desired fingertip position in palm frame (metres)
        q0         : (4,) initial joint angles in radians.
                     If None, the first structured seed for the finger type is used.
        tol        : convergence threshold in metres
        max_iter   : maximum L-BFGS-B iterations

        Returns
        -------
        q_sol : (4,) solution joint angles in radians
        info  : dict with keys 'success' (bool), 'error_m' (float), 'message' (str)
        """
        target_pos = np.asarray(target_pos, dtype=float)
        lo     = self.limits[finger_idx * 4:(finger_idx + 1) * 4, 0]
        hi     = self.limits[finger_idx * 4:(finger_idx + 1) * 4, 1]
        bounds = list(zip(lo, hi))

        if q0 is None:
            q0 = self._structured_seeds(finger_idx)[0]

        q0 = np.clip(np.asarray(q0, dtype=float), lo, hi)

        def cost(q):
            tip, _ = self.fk_finger(finger_idx, q)
            return float(np.sum((tip - target_pos) ** 2))

        result = minimize(
            cost, q0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )

        q_sol = result.x.copy()
        if finger_idx == 3:
            # Swap q12/q13 back after optimiser to match logical ordering
            q_sol[0], q_sol[1] = q_sol[1], q_sol[0]

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
        IK with structured seeds followed by random restarts for robustness.

        Structured seeds (finger-type-aware heuristics) are tried first
        because they cover the typical workspace much more efficiently than
        random initialisation. Random restarts fill the remainder up to n_starts.
        The solution with the smallest position error is returned.

        Parameters
        ----------
        n_starts : total number of initial conditions to try (seeds + random)
        seed     : RNG seed for reproducible random restarts
        """
        rng    = np.random.default_rng(seed)
        lo     = self.limits[finger_idx * 4:(finger_idx + 1) * 4, 0]
        hi     = self.limits[finger_idx * 4:(finger_idx + 1) * 4, 1]

        structured = self._structured_seeds(finger_idx)
        n_random   = max(0, n_starts - len(structured))
        candidates = structured + [rng.uniform(lo, hi) for _ in range(n_random)]

        best_q    = None
        best_info = {"error_m": np.inf, "success": False}

        for q0 in candidates:
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
        Full-hand IK: solve all four fingers independently and return a
        16-element joint vector.

        Each finger is solved in isolation (the fingers do not mechanically
        interact), allowing independent multi-start optimisation per finger.

        Parameters
        ----------
        targets  : (4, 3) desired fingertip positions in palm frame (metres)
                   order: [index, middle, ring, thumb]
        q0       : (16,) warm-start joint angles in radians (optional).
                   Passing the previous frame's solution dramatically improves
                   tracking performance.
        n_starts : random restarts per finger
        tol      : per-finger position tolerance in metres
        max_iter : max optimiser iterations per finger

        Returns
        -------
        q_sol  : (16,) joint angles in radians
        infos  : list of 4 info dicts, one per finger
        """
        targets = np.asarray(targets, dtype=float)
        q_sol   = np.zeros(16)
        infos   = []

        for i in range(4):
            q0_i = q0[i*4:(i+1)*4] if q0 is not None else None
            q_i, info = self.ik_finger_multistart(
                i, targets[i], n_starts=n_starts, tol=tol, max_iter=max_iter
            )
            q_sol[i*4:(i+1)*4] = q_i
            infos.append(info)

        return q_sol, infos

    def _structured_seeds(self, finger_idx: int):
        """
        Return a list of hand-crafted seed joint configurations for each finger.

        Seeds are chosen to cover the finger's reachable workspace: neutral,
        partial curl, full curl, abducted, and (for the thumb) opposition poses.
        These outperform random initialisation for typical teleoperation targets.
        """
        if finger_idx == 3:
            return [
                np.array([ 0.0,  0.0,  0.0,  0.0]),   # neutral / fully open
                np.array([ 0.0, -1.0,  0.8,  0.8]),   # light pinch
                np.array([-1.0, -1.5,  0.5,  0.5]),   # opposition
                np.array([ 0.0, -2.0,  1.5,  1.0]),   # deep flexion
                np.array([ 0.3, -0.5,  1.0,  1.5]),   # lateral pinch
                np.array([-0.5, -1.0,  1.2,  1.2]),   # mid opposition
            ]
        else:
            return [
                np.array([0.0,  0.0,  0.0,  0.0]),    # neutral / fully open
                np.array([0.0,  1.0,  1.0,  1.0]),    # half curl
                np.array([0.0,  1.5,  1.5,  1.0]),    # three-quarter curl
                np.array([0.0,  2.0,  1.8,  1.5]),    # full fist
                np.array([0.5,  1.0,  1.0,  1.0]),    # abducted half curl
                np.array([-0.5, 1.0,  1.0,  1.0]),    # adducted half curl
            ]

    def ik_degree(self, targets: np.ndarray, **kwargs):
        """Full-hand IK; returns joint angles in degrees."""
        q_sol, infos = self.ik(targets, **kwargs)
        return np.degrees(q_sol), infos

    def ik_finger_degree(self, finger_idx: int, target_pos: np.ndarray, **kwargs):
        """Single-finger IK (multi-start); returns joint angles in degrees."""
        q_sol, info = self.ik_finger_multistart(finger_idx, target_pos, **kwargs)
        return np.degrees(q_sol), info

    # ------------------------------------------------------------------ #
    # Numerical Jacobian                                                   #
    # ------------------------------------------------------------------ #

    def jacobian_finger(self, finger_idx: int, q4: np.ndarray, eps: float = 1e-6):
        """
        Compute the numerical Jacobian J ∈ R^{3×4} for a single finger.

        J[:, i] = ∂tip_pos / ∂q_i  (finite-difference approximation).

        Useful for Jacobian-based velocity control or sensitivity analysis.
        """
        q4   = np.asarray(q4, dtype=float)
        tip0, _ = self.fk_finger(finger_idx, q4)
        J    = np.zeros((3, 4))
        for i in range(4):
            dq    = np.zeros(4)
            dq[i] = eps
            tip_plus, _ = self.fk_finger(finger_idx, q4 + dq)
            J[:, i] = (tip_plus - tip0) / eps
        return J

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    def clip_to_limits(self, q: np.ndarray):
        """Clip a 16-element joint vector to the URDF joint limits (in-place safe)."""
        return np.clip(np.asarray(q, dtype=float), self.limits[:, 0], self.limits[:, 1])

    def is_within_limits(self, q: np.ndarray):
        """Return True if every joint in q is within its URDF limit."""
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))

    def print_fk(self, q: np.ndarray):
        """Pretty-print all four fingertip positions in the palm frame."""
        tips = self.fk(q)
        print("Fingertip positions (palm frame, metres):")
        for i, name in enumerate(self.FINGER_NAMES):
            x, y, z = tips[i]
            print(f"  {name:6s}: x={x:+.4f}  y={y:+.4f}  z={z:+.4f}")