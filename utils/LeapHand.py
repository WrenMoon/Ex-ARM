#!/usr/bin/env python3

from dynamixel_sdk import *
import numpy as np

# =========================
# Control Table Addresses
# =========================

ADDR_TORQUE_ENABLE = 64

ADDR_GOAL_POSITION = 116

ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

ADDR_PRESENT_POS_VEL_CUR = 126


# =========================
# Data Lengths
# =========================

LEN_GOAL_POSITION = 4

LEN_PRESENT_CURRENT = 2
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POSITION = 4

LEN_PRESENT_POS_VEL_CUR = 10


# =========================
# Conversion Constants
# =========================

DEGREE_TO_POSITION = 1 / 0.087891

LOGICAL_TO_PHYSICAL = np.array([
    8, 9, 10, 11,    # Index
    4, 5, 6, 7,      # Middle
    0, 1, 2, 3,      # Ring
    12,13,14,15      # Thumb
])

PHYSICAL_TO_LOGICAL = np.argsort(
    LOGICAL_TO_PHYSICAL
)


class LeapHand:

    def __init__(self, ids, port, baudrate, offsets):

        self.ids = list(ids)

        self.portHandler = PortHandler(port)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port {port}")

        if not self.portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")
    
        self.offsets = np.asarray(offsets)[LOGICAL_TO_PHYSICAL]
        # -------------------------
        # Sync Write (Goal Position)
        # -------------------------

        self.groupSyncWritePos = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )

        # -------------------------
        # Sync Read
        # Current + Velocity + Position
        # -------------------------

        self.groupSyncReadState = GroupSyncRead(
            self.portHandler,
            self.packetHandler,
            ADDR_PRESENT_POS_VEL_CUR,
            LEN_PRESENT_POS_VEL_CUR
        )

        for dxl_id in self.ids:
            if not self.groupSyncReadState.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add motor {dxl_id} to SyncRead"
                )

    # ==================================================
    # Utility
    # ==================================================

    @staticmethod
    def int32_to_bytes(value):

        value = int(value)

        return [
            value & 0xFF,
            (value >> 8) & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        ]

    # ==================================================
    # Port
    # ==================================================

    def close_port(self):
        self.portHandler.closePort()

    # ==================================================
    # Torque
    # ==================================================

    def set_torque_enabled(self, enable):

        sync = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            ADDR_TORQUE_ENABLE,
            1
        )

        value = [1 if enable else 0]

        for dxl_id in self.ids:
            sync.addParam(dxl_id, value)

        sync.txPacket()
        sync.clearParam()

    # ==================================================
    # Goal Position
    # ==================================================

    def set_goal_positions_degree(self, positions):

        positions = np.asarray(positions)

        positions = positions[
            LOGICAL_TO_PHYSICAL
]

        if len(positions) != len(self.ids):
            raise ValueError(
                f"Expected {len(self.ids)} positions, "
                f"got {len(positions)}"
            )

        self.groupSyncWritePos.clearParam()

        for dxl_id, position_deg in zip(self.ids, positions):

            raw_position = int((position_deg + self.offsets[self.ids.index(dxl_id)] + 180) * DEGREE_TO_POSITION)

            param = self.int32_to_bytes(raw_position)

            if not self.groupSyncWritePos.addParam(
                dxl_id,
                param
            ):
                raise RuntimeError(
                    f"Failed to add motor {dxl_id}"
                )

        result = self.groupSyncWritePos.txPacket()

        if result != COMM_SUCCESS:
            raise RuntimeError(
                self.packetHandler.getTxRxResult(result)
            )

    # ==================================================
    # Read State
    # ==================================================

    def get_state(self):

        result = self.groupSyncReadState.txRxPacket()

        if result != COMM_SUCCESS:
            return [0,0,0]

        positions = []
        velocities = []
        currents = []

        for dxl_id in self.ids:

            current = self.groupSyncReadState.getData(
                dxl_id,
                ADDR_PRESENT_CURRENT,
                LEN_PRESENT_CURRENT
            )

            velocity = self.groupSyncReadState.getData(
                dxl_id,
                ADDR_PRESENT_VELOCITY,
                LEN_PRESENT_VELOCITY
            )

            position = self.groupSyncReadState.getData(
                dxl_id,
                ADDR_PRESENT_POSITION,
                LEN_PRESENT_POSITION
            )

            positions.append(
                (position - self.offsets[list(self.ids).index(dxl_id)]) / DEGREE_TO_POSITION
            )

            velocities.append(velocity)
            currents.append(current)

        positions = np.array(positions)[
            PHYSICAL_TO_LOGICAL
        ]

        velocities = np.array(velocities)[
            PHYSICAL_TO_LOGICAL
        ]

        currents = np.array(currents)[
            PHYSICAL_TO_LOGICAL
        ]

        return (
            positions,
            velocities,
            currents
        )

    # ==================================================
    # Convenience Functions
    # ==================================================

    def get_present_positions_degree(self):
        positions, _, _ = self.get_state()
        return positions

    def get_present_velocities(self):
        _, velocities, _ = self.get_state()
        return velocities

    def get_present_currents(self):
        _, _, currents = self.get_state()
        return currents