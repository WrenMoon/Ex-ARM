import numpy as np

class PlanarFinger:

    def __init__(
        self,
        link_lengths,
        q_min=None,
        q_max=None
    ):
        """
        link_lengths : list of link lengths [L1,L2,...]

        q_min/q_max : optional joint limits (degrees)
        """

        self.L = np.array(link_lengths, dtype=float)

        self.n = len(self.L)

        if q_min is None:
            q_min = -180 * np.ones(self.n)

        if q_max is None:
            q_max = 180 * np.ones(self.n)

        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)

    # ==========================================
    # Forward Kinematics
    # ==========================================

    def fk(self, q):

        q = np.radians(np.asarray(q))

        x = 0.0
        y = 0.0

        angle = 0.0

        for qi, Li in zip(q, self.L):

            angle += qi

            x += Li * np.cos(angle)
            y += Li * np.sin(angle)

        return np.array([x, y])

    # ==========================================
    # Joint Positions
    # ==========================================

    def joint_positions(self, q):

        q = np.radians(np.asarray(q))

        pts = [[0.0, 0.0]]

        x = 0.0
        y = 0.0

        angle = 0.0

        for qi, Li in zip(q, self.L):

            angle += qi

            x += Li * np.cos(angle)
            y += Li * np.sin(angle)

            pts.append([x, y])

        return np.array(pts)

    # ==========================================
    # Jacobian
    # ==========================================

    def jacobian(self, q):

        q = np.radians(np.asarray(q))

        J = np.zeros((2, self.n))

        cumulative = np.cumsum(q)

        for j in range(self.n):

            dx = 0.0
            dy = 0.0

            for k in range(j, self.n):

                angle = cumulative[k]

                dx -= self.L[k] * np.sin(angle)
                dy += self.L[k] * np.cos(angle)

            J[0, j] = dx
            J[1, j] = dy

        return J

    # ==========================================
    # IK
    # ==========================================

    def ik(
        self,
        target,
        q0=None,
        max_iter=100,
        tol=1e-4,
        damping=1e-3
    ):

        target = np.asarray(target)

        if q0 is None:
            q = np.ones(self.n) * 10.0
        else:
            q = np.asarray(q0, dtype=float).copy()

        for _ in range(max_iter):

            pos = self.fk(q)

            error = target - pos

            if np.linalg.norm(error) < tol:
                return q

            J = self.jacobian(q)

            JT = J.T

            dq_rad = JT @ np.linalg.inv(
                J @ JT + damping * np.eye(2)
            ) @ error

            dq_deg = np.degrees(dq_rad)

            q += 0.2* dq_deg

            q = np.clip(
                q,
                np.degrees(self.q_min),
                np.degrees(self.q_max)
            )

        q = (q + 180) % 360 - 180

        return q

    # ==========================================
    # Reachability
    # ==========================================

    def max_reach(self):
        return np.sum(self.L)