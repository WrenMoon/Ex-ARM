from utils.ExARM import ExArm
from utils.LeapKinematics import LeapKinematics
import time
import numpy as np
import pygame

# =========================
# INIT ROBOT
# =========================
leap_hand = ExArm(
    mode="sim",
    ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    port="COM5",
    baudrate=4000000,
    offsets=[0,-90,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    model_path="Data/mujoco_robot.urdf"
)
leap_kinematics = LeapKinematics()


# =========================
# INIT KEY CONTROL
# =========================
pygame.init()
pygame.display.set_mode((200, 200))  # required for key capture

q = np.zeros(4, dtype=float)
step = 5  # degrees

running = True

# =========================
# MAIN LOOP
# =========================
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_q]: q[0] += step
    if keys[pygame.K_a]: q[0] -= step
    if keys[pygame.K_w]: q[1] += step
    if keys[pygame.K_s]: q[1] -= step
    if keys[pygame.K_e]: q[2] += step
    if keys[pygame.K_d]: q[2] -= step
    if keys[pygame.K_r]: q[3] += step
    if keys[pygame.K_f]: q[3] -= step

    q_full = np.tile(q, 4)  # same 4 joints repeated for all 4 fingers

    target = leap_kinematics.fk_degree(q_full)        # (4, 3) fingertip positions
    # leap_kinematics.print_fk(np.radians(q_full))

    ik_sol, infos = leap_kinematics.ik_degree(target)  # unpack tuple
    ik_target = leap_kinematics.fk_degree(ik_sol)
    leap_hand.set_custom_markers(ik_target)

    leap_hand.set_goal_positions_degree(ik_sol)
    print(f"current Pos: {ik_target}", end="\r", flush=True)

    time.sleep(0.01)