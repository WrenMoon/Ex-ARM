#!/usr/bin/env python3
"""
LeapHand.py — Low-level Dynamixel driver for the LEAP Hand.

Wraps the Dynamixel SDK GroupSyncWrite / GroupSyncRead protocol to provide
a clean, high-level interface for sending joint angle goals and reading back
position / velocity / current state at high frequency.

Joint ordering throughout this module uses the *logical* convention:
  [index(4), middle(4), ring(4), thumb(4)]  — 16 joints total.

Internally, commands are remapped to the *physical* Dynamixel ID ordering
before any packet is transmitted.
"""

from dynamixel_sdk import *
import numpy as np

# ------------------------------------------------------------------ #
# Dynamixel control table addresses (Protocol 2.0)                   #
# ------------------------------------------------------------------ #

ADDR_TORQUE_ENABLE      = 64   # 1 byte  — enable/disable motor torque
ADDR_GOAL_POSITION      = 116  # 4 bytes — target position register
ADDR_PRESENT_CURRENT    = 126  # 2 bytes — measured motor current
ADDR_PRESENT_VELOCITY   = 128  # 4 bytes — measured joint velocity
ADDR_PRESENT_POSITION   = 132  # 4 bytes — measured joint position
ADDR_PRESENT_POS_VEL_CUR = 126 # burst-read start address (current→velocity→position)

# ------------------------------------------------------------------ #
# Register data lengths (bytes)                                       #
# ------------------------------------------------------------------ #

LEN_GOAL_POSITION       = 4
LEN_PRESENT_CURRENT     = 2
LEN_PRESENT_VELOCITY    = 4
LEN_PRESENT_POSITION    = 4
LEN_PRESENT_POS_VEL_CUR = 10   # current(2) + velocity(4) + position(4)

# ------------------------------------------------------------------ #
# Unit conversion                                                     #
# ------------------------------------------------------------------ #

# Dynamixel raw position units per degree (360° / 4096 ticks ≈ 0.0879°/tick)
DEGREE_TO_POSITION = 1 / 0.087891

# ------------------------------------------------------------------ #
# Joint index remapping                                               #
# ------------------------------------------------------------------ #

# Maps logical joint index → physical Dynamixel ID index.
# Logical order: index[0-3], middle[4-7], ring[8-11], thumb[12-15]
# Physical order on the hand wiring: ring, middle, index, thumb
LOGICAL_TO_PHYSICAL = np.array([
    8, 9, 10, 11,    # logical index   → physical slots 8-11
    4, 5, 6,  7,     # logical middle  → physical slots 4-7
    0, 1, 2,  3,     # logical ring    → physical slots 0-3
    12,13,14, 15     # logical thumb   → physical slots 12-15
])

# Inverse mapping: physical → logical (used when reading back state)
PHYSICAL_TO_LOGICAL = np.argsort(LOGICAL_TO_PHYSICAL)


class LeapHand:
    """
    Hardware driver for the LEAP Hand robotic finger assembly.

    Manages port initialisation, torque control, bulk goal-position writes,
    and synchronised state reads over a USB-to-TTL serial connection.
    """

    def __init__(self, ids, port, baudrate, offsets):
        """
        Open the serial port and initialise Sync-Write/Read handlers.

        Parameters
        ----------
        ids      : list of 16 Dynamixel servo IDs in *physical* order
        port     : serial port string (e.g. 'COM5' or '/dev/ttyUSB0')
        baudrate : communication baud rate (e.g. 4000000)
        offsets  : (16,) per-joint angle offsets in degrees, in *logical* order.
                   Applied as a bias when converting degrees ↔ raw ticks.
        """
        self.ids = list(ids)

        self.portHandler   = PortHandler(port)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            raise RuntimeError(f"Failed to open port {port}")

        if not self.portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to set baudrate {baudrate}")

        # Store offsets in physical order so they align with self.ids during writes/reads
        self.offsets = np.asarray(offsets)[LOGICAL_TO_PHYSICAL]

        # ---------------------------------------------------------- #
        # Sync Write handler — broadcasts goal positions to all motors
        # in a single USB packet to minimise latency.
        # ---------------------------------------------------------- #
        self.groupSyncWritePos = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION
        )

        # ---------------------------------------------------------- #
        # Sync Read handler — reads current + velocity + position
        # from all motors in one burst read (contiguous registers).
        # ---------------------------------------------------------- #
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

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def int32_to_bytes(value):
        """
        Convert a signed 32-bit integer to a 4-byte little-endian list,
        as required by the Dynamixel SDK addParam API.
        """
        value = int(value)
        return [
            value & 0xFF,
            (value >> 8)  & 0xFF,
            (value >> 16) & 0xFF,
            (value >> 24) & 0xFF,
        ]

    # ------------------------------------------------------------------ #
    # Port management                                                      #
    # ------------------------------------------------------------------ #

    def close_port(self):
        """Close the USB serial port gracefully."""
        self.portHandler.closePort()

    # ------------------------------------------------------------------ #
    # Torque control                                                       #
    # ------------------------------------------------------------------ #

    def set_torque_enabled(self, enable):
        """
        Enable or disable torque on all motors simultaneously.

        Parameters
        ----------
        enable : bool — True to energise, False to release.
        """
        sync  = GroupSyncWrite(
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

    # ------------------------------------------------------------------ #
    # Goal position                                                        #
    # ------------------------------------------------------------------ #

    def set_goal_positions_degree(self, positions):
        """
        Send target joint angles to all motors in a single Sync Write packet.

        Parameters
        ----------
        positions : (16,) array-like — joint angles in degrees, *logical* order.
                    The method remaps to physical order and applies stored offsets
                    before computing raw Dynamixel ticks.

        Raises
        ------
        ValueError    if the length of positions does not match the motor count.
        RuntimeError  if the Dynamixel TX packet fails.
        """
        positions = np.asarray(positions)

        # Remap from logical → physical ordering
        positions = positions[LOGICAL_TO_PHYSICAL]

        if len(positions) != len(self.ids):
            raise ValueError(
                f"Expected {len(self.ids)} positions, got {len(positions)}"
            )

        self.groupSyncWritePos.clearParam()

        for dxl_id, position_deg in zip(self.ids, positions):
            # Convert degree → raw tick, applying per-joint offset and 180° bias
            # (Dynamixel zero position corresponds to 180° in our convention)
            raw_position = int(
                (position_deg + self.offsets[self.ids.index(dxl_id)] + 180)
                * DEGREE_TO_POSITION
            )

            param = self.int32_to_bytes(raw_position)

            if not self.groupSyncWritePos.addParam(dxl_id, param):
                raise RuntimeError(f"Failed to add motor {dxl_id}")

        result = self.groupSyncWritePos.txPacket()

        if result != COMM_SUCCESS:
            raise RuntimeError(self.packetHandler.getTxRxResult(result))

    # ------------------------------------------------------------------ #
    # State reading                                                        #
    # ------------------------------------------------------------------ #

    def get_state(self):
        """
        Read position, velocity, and current from all motors in one burst.

        Returns
        -------
        positions  : (16,) float ndarray — joint angles in degrees, *logical* order.
        velocities : (16,) int   ndarray — raw velocity counts, *logical* order.
        currents   : (16,) int   ndarray — raw current counts,  *logical* order.

        Returns ([0,0,0]) on communication failure to avoid crashing callers.
        """
        result = self.groupSyncReadState.txRxPacket()

        if result != COMM_SUCCESS:
            return [0, 0, 0]

        positions  = []
        velocities = []
        currents   = []

        for dxl_id in self.ids:

            current = self.groupSyncReadState.getData(
                dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
            )
            velocity = self.groupSyncReadState.getData(
                dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
            )
            position = self.groupSyncReadState.getData(
                dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
            )

            # Convert raw ticks back to degrees, removing the per-joint offset
            positions.append(
                (position - self.offsets[list(self.ids).index(dxl_id)])
                / DEGREE_TO_POSITION
            )
            velocities.append(velocity)
            currents.append(current)

        # Reorder from physical → logical so callers always see logical ordering
        positions  = np.array(positions)[PHYSICAL_TO_LOGICAL]
        velocities = np.array(velocities)[PHYSICAL_TO_LOGICAL]
        currents   = np.array(currents)[PHYSICAL_TO_LOGICAL]

        return (positions, velocities, currents)

    # ------------------------------------------------------------------ #
    # Convenience accessors                                                #
    # ------------------------------------------------------------------ #

    def get_present_positions_degree(self):
        """Return current joint positions in degrees (logical order)."""
        positions, _, _ = self.get_state()
        return positions

    def get_present_velocities(self):
        """Return current joint velocities as raw Dynamixel counts (logical order)."""
        _, velocities, _ = self.get_state()
        return velocities

    def get_present_currents(self):
        """Return current motor currents as raw Dynamixel counts (logical order)."""
        _, _, currents = self.get_state()
        return currents