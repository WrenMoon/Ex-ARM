#!/usr/bin/env python3

from dynamixel_sdk import *
import numpy as np

class Dynamixel_servo:

    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_PRESENT_VELOCITY = 128
    ADDR_PRESENT_CURRENT = 126
    ADDR_PRESENT_POS_VEL_CUR = 126
    ADDR_PRESENT_POS_VEL = 128

    DEGREE_TO_POSITION = 1/0.087891

    def __init__(self, id, port, baudrate):
        self.id = id
        portHandler = PortHandler(port)  # your dxl port name
        packetHandler = PacketHandler(2.0)  # protocol version
        portHandler.openPort()
        portHandler.setBaudRate(baudrate)
        
    def open_port(self, port, baudrate):
        self.portHandler = PortHandler(port)  # your dxl port name
        self.packetHandler = PacketHandler(2.0)  # protocol version
        self.portHandler.openPort()
        self.portHandler.setBaudRate(baudrate)
    
    def close_port(self):
        self.portHandler.closePort
    
    def set_torque_enabled(self, enable):
        # Enable/Disable Torque
        if enable:
            packetHandler.write1ByteTxRx(portHandler, self.id, self.ADDR_TORQUE_ENABLE, 1)
        else:
            packetHandler.write1ByteTxRx(portHandler, self.id, self.ADDR_TORQUE_ENABLE, 0)
    
    def set_goal_position_degree(self, position):
        # Set Goal Position
        packetHandler.write4ByteTxRx(portHandler, self.id, self.ADDR_GOAL_POSITION, np.clip(position*self.DEGREE_TO_POSITION, 0, 4095))
    
    def get_present_position_degree(self):
        # Get Present Position
        return packetHandler.read4ByteTxRx(portHandler, self.id, self.ADDR_PRESENT_POSITION)[0]/self.DEGREE_TO_POSITION

    def get_present_velocity(self):
        # Get Preset Velocity
        return packetHandler.read4ByteRx(portHandler, self.id, self.ADDR_PRESENT_CURRENT)[0]

