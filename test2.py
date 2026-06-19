from utils.ExARM import ExArm
from utils.leap_kinematics import URDFFinger
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

# =========================
# FINGERS
# =========================
index = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_tip")
middle = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_2_tip")
ring = URDFFinger("Data/robot.urdf", "palm_lower_left", "fingertip_3_tip")
thumb = URDFFinger("Data/robot.urdf", "palm_lower_left", "thumb_fingertip_tip")

fingers = [index, middle, ring, thumb]

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

    # finger 1
    if keys[pygame.K_q]: q[0] += step
    if keys[pygame.K_a]: q[0] -= step

    # finger 2
    if keys[pygame.K_w]: q[1] += step
    if keys[pygame.K_s]: q[1] -= step

    # finger 3
    if keys[pygame.K_e]: q[2] += step
    if keys[pygame.K_d]: q[2] -= step

    # finger 4
    if keys[pygame.K_r]: q[3] += step
    if keys[pygame.K_f]: q[3] -= step

    fk_input = np.radians(q)

    positions = []  # reset every loop properly
    fk_sols = []

    for finger in fingers:
        fk_sol = finger.fk(fk_input)
        ik_sol = finger.ik(fk_sol)  

        fk_sols.append(fk_sol)

        positions.append(ik_sol)

    positions = np.concatenate(positions)

    leap_hand.set_goal_positions_degree(np.rad2deg(positions))
    print(f"current Pos: {fk_sols}")


    time.sleep(0.01)