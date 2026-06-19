import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


class URDFFinger:

    def __init__(
        self,
        urdf_path,
        root_link,
        tip_link,
        q_min=None,
        q_max=None
    ):

        tree = ET.parse(urdf_path)
        robot = tree.getroot()

        all_joints = []

        for joint in robot.findall("joint"):

            if joint.attrib["type"] == "fixed":
                continue

            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]

            origin = joint.find("origin")
            axis = joint.find("axis")

            all_joints.append({
                "name": joint.attrib["name"],
                "parent": parent,
                "child": child,
                "xyz": np.fromstring(
                    origin.attrib["xyz"],
                    sep=" "
                ),
                "rpy": np.fromstring(
                    origin.attrib["rpy"],
                    sep=" "
                ),
                "axis": np.fromstring(
                    axis.attrib["xyz"],
                    sep=" "
                )
            })

        child_to_joint = {
            j["child"]: j
            for j in all_joints
        }

        self.chain = []

        current = tip_link

        while current != root_link:

            if current not in child_to_joint:

                raise RuntimeError(
                    f"Could not find joint leading to '{current}'"
                )

            joint = child_to_joint[current]

            self.chain.append(joint)

            current = joint["parent"]

        self.chain.reverse()

        print("\nFinger chain:")

        for joint in self.chain:

            print(
                joint["name"],
                ":",
                joint["parent"],
                "->",
                joint["child"]
            )

        self.n = len(self.chain)

        if q_min is None:
            q_min = np.full(
                self.n,
                -np.pi
            )

        if q_max is None:
            q_max = np.full(
                self.n,
                np.pi
            )

        self.q_min = np.asarray(q_min)
        self.q_max = np.asarray(q_max)

    # ==================================================
    # Transform Helpers
    # ==================================================

    @staticmethod
    def transform(xyz, rpy):

        T = np.eye(4)

        T[:3, :3] = (
            Rotation
            .from_euler(
                "xyz",
                rpy
            )
            .as_matrix()
        )

        T[:3, 3] = xyz

        return T

    @staticmethod
    def axis_rotation(axis, theta):

        axis = np.asarray(
            axis,
            dtype=float
        )

        axis /= np.linalg.norm(axis)

        R = Rotation.from_rotvec(
            axis * theta
        ).as_matrix()

        T = np.eye(4)

        T[:3, :3] = R

        return T

    # ==================================================
    # FK
    # ==================================================

    def fk_matrix(self, q):

        q = np.asarray(
            q,
            dtype=float
        )

        T = np.eye(4)

        for angle, joint in zip(
            q,
            self.chain
        ):

            T = (
                T
                @ self.transform(
                    joint["xyz"],
                    joint["rpy"]
                )
                @ self.axis_rotation(
                    joint["axis"],
                    angle
                )
            )

        return T

    def fk(self, q):
        """
        Returns fingertip xyz position.
        """

        return (
            self
            .fk_matrix(q)
        )[:3, 3]

    def pose(self, q):
        """
        Returns full 4x4 pose matrix.
        """

        return self.fk_matrix(q)

    # ==================================================
    # Joint Positions
    # ==================================================

    def joint_positions(self, q):

        q = np.asarray(q)

        pts = [np.zeros(3)]

        T = np.eye(4)

        for angle, joint in zip(
            q,
            self.chain
        ):

            T = (
                T
                @ self.transform(
                    joint["xyz"],
                    joint["rpy"]
                )
                @ self.axis_rotation(
                    joint["axis"],
                    angle
                )
            )

            pts.append(
                T[:3, 3].copy()
            )

        return np.array(pts)

    # ==================================================
    # Jacobian
    # ==================================================

    def jacobian(
        self,
        q,
        eps=1e-6
    ):

        q = np.asarray(q)

        p0 = self.fk(q)

        J = np.zeros(
            (3, self.n)
        )

        for i in range(self.n):

            q2 = q.copy()

            q2[i] += eps

            p1 = self.fk(q2)

            J[:, i] = (
                p1 - p0
            ) / eps

        return J

    # ==================================================
    # IK
    # ==================================================

    def ik(
        self,
        target,
        q0=None,
        max_iter=200,
        tol=1e-4,
        damping=1e-4
    ):

        target = np.asarray(
            target,
            dtype=float
        )

        if q0 is None:
            q = np.zeros(
                self.n
            )

        else:
            q = np.asarray(
                q0,
                dtype=float
            )

        for _ in range(max_iter):

            pos = self.fk(q)

            error = (
                target - pos
            )

            if (
                np.linalg.norm(error)
                < tol
            ):
                return q

            J = self.jacobian(q)

            dq = (
                J.T
                @ np.linalg.solve(
                    J @ J.T
                    + damping*np.eye(3),
                    error
                )
            )

            q += 0.5 * dq

            q = np.clip(
                q,
                self.q_min,
                self.q_max
            )

        return q

    # ==================================================
    # Utility
    # ==================================================

    @staticmethod
    def deg2rad(q):
        return np.radians(q)

    @staticmethod
    def rad2deg(q):
        return np.degrees(q)