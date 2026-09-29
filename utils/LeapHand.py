#!/usr/bin/env python3
"""
LeapHand.py — Low-level Dynamixel driver for the LEAP Hand.

Wraps the Dynamixel SDK GroupSyncWrite / GroupBulkRead protocol to provide
a clean, high-level interface for sending joint angle goals and reading back
position / velocity / current state at high frequency.

Joint ordering throughout this module uses the *logical* convention:
  [index(4), middle(4), ring(4), thumb(4)]  — 16 joints total.

Internally, commands are remapped to the *physical* Dynamixel ID ordering
before any packet is transmitted.
"""

from dynamixel_sdk import *
import numpy as np
import threading

# ------------------------------------------------------------------ #
# Dynamixel control table addresses (Protocol 2.0)                   #
# ------------------------------------------------------------------ #

ADDR_TORQUE_ENABLE      = 64   # 1 byte  — enable/disable motor torque
ADDR_OPERATING_MODE     = 11   # 1 byte  — select motor control mode
ADDR_GOAL_PWM           = 100  # 2 bytes — PWM output goal
ADDR_GOAL_CURRENT       = 102  # 2 bytes — current goal / position-mode limit
ADDR_GOAL_VELOCITY      = 104  # 4 bytes — velocity goal
ADDR_GOAL_POSITION      = 116  # 4 bytes — target position register
ADDR_PRESENT_CURRENT    = 126  # 2 bytes — measured motor current
ADDR_PRESENT_VELOCITY   = 128  # 4 bytes — measured joint velocity
ADDR_PRESENT_POSITION   = 132  # 4 bytes — measured joint position
ADDR_PRESENT_POS_VEL_CUR = 126 # burst-read start address (current→velocity→position)

# ------------------------------------------------------------------ #
# Register data lengths (bytes)                                       #
# ------------------------------------------------------------------ #

LEN_GOAL_POSITION       = 4
LEN_GOAL_PWM            = 2
LEN_GOAL_CURRENT        = 2
LEN_GOAL_VELOCITY       = 4
LEN_PRESENT_CURRENT     = 2
LEN_PRESENT_VELOCITY    = 4
LEN_PRESENT_POSITION    = 4
LEN_PRESENT_POS_VEL_CUR = 10   # current(2) + velocity(4) + position(4)

# XL330 operating modes
MODE_CURRENT_CONTROL = 0
MODE_VELOCITY_CONTROL = 1
MODE_POSITION_CONTROL = 3
MODE_EXTENDED_POSITION_CONTROL = 4
MODE_CURRENT_BASED_POSITION_CONTROL = 5
MODE_PWM_CONTROL = 16
SUPPORTED_OPERATING_MODES = {
    MODE_CURRENT_CONTROL,
    MODE_VELOCITY_CONTROL,
    MODE_POSITION_CONTROL,
    MODE_EXTENDED_POSITION_CONTROL,
    MODE_CURRENT_BASED_POSITION_CONTROL,
    MODE_PWM_CONTROL,
}

VELOCITY_RPM_PER_COUNT = 0.229
PWM_PERCENT_PER_COUNT = 0.113

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
    0, 1, 2, 3,    # logical index   → physical slots 0-3
    4, 5, 6,  7,     # logical middle  → physical slots 4-7
    8, 9, 10, 11,    # logical ring    → physical slots 8-11
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
        self._bus_lock = threading.RLock()

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
        # Bulk Read handler — reads current + velocity + position
        # from the contiguous register range on each motor.
        # ---------------------------------------------------------- #
        self.groupBulkReadState = GroupBulkRead(
            self.portHandler,
            self.packetHandler,
        )

        for dxl_id in self.ids:
            if not self.groupBulkReadState.addParam(
                dxl_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR
            ):
                raise RuntimeError(
                    f"Failed to add motor {dxl_id} to BulkRead"
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

    @staticmethod
    def signed_int_to_bytes(value, data_length):
        """Encode a signed integer in little-endian two's-complement form."""
        value = int(value)
        bit_count = data_length * 8
        minimum = -(1 << (bit_count - 1))
        maximum = (1 << (bit_count - 1)) - 1
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{value} does not fit in a signed {bit_count}-bit register"
            )
        return [(value >> (8 * index)) & 0xFF for index in range(data_length)]

    def _logical_joint_values(self, values, name):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(self.ids),):
            raise ValueError(f"{name} must contain {len(self.ids)} values")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values[LOGICAL_TO_PHYSICAL]

    def _sync_write_register(self, address, data_length, physical_values):
        with self._bus_lock:
            sync = GroupSyncWrite(
                self.portHandler,
                self.packetHandler,
                address,
                data_length,
            )
            try:
                for dxl_id, value in zip(self.ids, physical_values):
                    param = self.signed_int_to_bytes(value, data_length)
                    if not sync.addParam(dxl_id, param):
                        raise RuntimeError(f"Failed to add motor {dxl_id}")

                result = sync.txPacket()
                if result != COMM_SUCCESS:
                    raise RuntimeError(self.packetHandler.getTxRxResult(result))
            finally:
                sync.clearParam()

    # ------------------------------------------------------------------ #
    # Port management                                                      #
    # ------------------------------------------------------------------ #

    def close_port(self):
        """Close the USB serial port gracefully."""
        with self._bus_lock:
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
        with self._bus_lock:
            sync  = GroupSyncWrite(
                self.portHandler,
                self.packetHandler,
                ADDR_TORQUE_ENABLE,
                1
            )
            value = [1 if enable else 0]

            try:
                for dxl_id in self.ids:
                    if not sync.addParam(dxl_id, value):
                        raise RuntimeError(f"Failed to add motor {dxl_id} for torque control")

                result = sync.txPacket()
                if result != COMM_SUCCESS:
                    raise RuntimeError(self.packetHandler.getTxRxResult(result))
            finally:
                sync.clearParam()

    def set_operating_mode(self, mode):
        """Set an XL330 operating mode; torque is disabled and left off."""
        try:
            mode = int(mode)
        except (TypeError, ValueError) as error:
            raise ValueError("mode must be a supported integer operating mode") from error
        if mode not in SUPPORTED_OPERATING_MODES:
            raise ValueError(
                f"Unsupported XL330 mode {mode}; expected one of "
                f"{sorted(SUPPORTED_OPERATING_MODES)}"
            )

        with self._bus_lock:
            self.set_torque_enabled(False)
            for dxl_id in self.ids:
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if result != COMM_SUCCESS:
                    raise RuntimeError(self.packetHandler.getTxRxResult(result))
                if error != 0:
                    raise RuntimeError(self.packetHandler.getRxPacketError(error))

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

        if len(positions) != len(self.ids):
                    raise ValueError(
                        f"Expected {len(self.ids)} positions, got {len(positions)}"
                    )

        # Remap from logical → physical ordering
        positions = positions[LOGICAL_TO_PHYSICAL]

        with self._bus_lock:
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

    def set_goal_currents(self, currents_ma):
        """Set current goals/limits in mA (Current or Current-based Position mode)."""
        currents = self._logical_joint_values(currents_ma, "currents_ma")
        self._sync_write_register(
            ADDR_GOAL_CURRENT,
            LEN_GOAL_CURRENT,
            np.rint(currents).astype(np.int64),
        )

    def set_goal_velocities_rpm(self, velocities_rpm):
        """Set signed velocity goals in RPM (Velocity Control mode)."""
        velocities = self._logical_joint_values(velocities_rpm, "velocities_rpm")
        raw_values = np.rint(velocities / VELOCITY_RPM_PER_COUNT).astype(np.int64)
        self._sync_write_register(
            ADDR_GOAL_VELOCITY, LEN_GOAL_VELOCITY, raw_values
        )

    def set_goal_pwm_percent(self, pwm_percent):
        """Set signed PWM goals as percentages from -100 to 100 (PWM mode)."""
        pwm = self._logical_joint_values(pwm_percent, "pwm_percent")
        if np.any(np.abs(pwm) > 100):
            raise ValueError("pwm_percent values must be between -100 and 100")
        raw_values = np.rint(pwm / PWM_PERCENT_PER_COUNT).astype(np.int64)
        self._sync_write_register(ADDR_GOAL_PWM, LEN_GOAL_PWM, raw_values)

    def set_goal_positions_current_based(self, positions_deg, current_limits_ma):
        """Set degree goals and current limits in Current-based Position mode."""
        self._logical_joint_values(positions_deg, "positions_deg")
        self._logical_joint_values(current_limits_ma, "current_limits_ma")
        self.set_goal_currents(current_limits_ma)
        self.set_goal_positions_degree(positions_deg)

    def set_goal_positions_pulses(self, positions_pulses):
        """Set signed raw position targets for extended/current-based position mode."""
        positions = self._logical_joint_values(positions_pulses, "positions_pulses")
        raw_values = np.rint(positions).astype(np.int64)
        if np.any(np.abs(raw_values) > 1_048_575):
            raise ValueError("position pulse targets must be within +/-1,048,575")
        self._sync_write_register(
            ADDR_GOAL_POSITION, LEN_GOAL_POSITION, raw_values
        )

    def set_goal_positions_current_based_pulses(
        self, positions_pulses, current_limits_ma
    ):
        """Set raw pulse targets and current limits in Current-based Position mode."""
        self._logical_joint_values(positions_pulses, "positions_pulses")
        self._logical_joint_values(current_limits_ma, "current_limits_ma")
        self.set_goal_currents(current_limits_ma)
        self.set_goal_positions_pulses(positions_pulses)

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
        currents   : (16,) int   ndarray — signed current register values in
                             *logical* order (mA on XL330).

        Returns [] on communication failure.
        """
        with self._bus_lock:
            result = self.groupBulkReadState.txRxPacket()

            if result != COMM_SUCCESS:
                return []

            if not all(
                self.groupBulkReadState.isAvailable(
                    dxl_id, ADDR_PRESENT_POS_VEL_CUR, LEN_PRESENT_POS_VEL_CUR
                )
                for dxl_id in self.ids
            ):
                return []

            positions  = []
            velocities = []
            currents   = []

            for physical_index, dxl_id in enumerate(self.ids):
                current = self.groupBulkReadState.getData(
                    dxl_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT
                )
                velocity = self.groupBulkReadState.getData(
                    dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                )
                position = self.groupBulkReadState.getData(
                    dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                )

                if current >= 1 << 15:
                    current -= 1 << 16
                if velocity >= 1 << 31:
                    velocity -= 1 << 32
                if position >= 1 << 31:
                    position -= 1 << 32

                # Invert the 180-degree bias and offset applied when commanding.
                positions.append(
                    position / DEGREE_TO_POSITION
                    - self.offsets[physical_index]
                    - 180
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