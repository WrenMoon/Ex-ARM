import numpy as np
import time, math
from utils.ExARM import ExArm

def main(**kwargs):
    leap_hand = ExArm(
        mode="both",
        ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        port="COM5",
        baudrate=4000000,
        offsets=[0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        model_path="Data/mujoco_robot.urdf"
    )
    start_time = time.time()
    duration = 30

        
    while time.time() - start_time < duration:
        if time.time() - start_time < 5:
            leap_hand.set_goal_positions_degree(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
        elif time.time() - start_time < 7:
            leap_hand.set_goal_positions_degree(np.array([70,22,85,22,
                                            0,0,0,0,
                                            -70,22,85,22,
                                            -45,-100,30,42]))
        elif time.time() - start_time < 8:
            leap_hand.set_goal_positions_degree(np.array([70,22,85,45,
                                            0,0,0,0,
                                            -70,22,85,45,
                                            -65,-100,30,42]))
        elif time.time() - start_time < 9:
            leap_hand.set_goal_positions_degree(np.array([70,0,85,0,     
                                            0,0,0,0,
                                            0,22,85,0,
                                            -65,-45,37,50]))
        elif time.time() - start_time < 10:
            leap_hand.set_goal_positions_degree(np.array([0,0,0,0,
                                            -30,0,30,30,
                                            0,45,85,0,
                                            -65,-45,37,50]))
        elif time.time() - start_time < 12:
            leap_hand.set_goal_positions_degree(np.array([0,55,0,0,
                                            0,55,55,37,
                                            0,55,55,37,
                                            -120,-45,10,70]))
        elif time.time() - start_time < 12.5:
            leap_hand.set_goal_positions_degree(np.array([0,55,55,37,
                                            0,55,55,37,
                                            0,55,55,37,
                                            -120,-45,10,70]))

        time.sleep(0.03)
        leap_hand.set_torque_enabled(True)
        # positions, velocities, currents = leap_hand.get_state()
        # print("Positions:", positions)
    leap_hand.set_torque_enabled(False)



if __name__ == "__main__":
    main()