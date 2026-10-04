import csv
import json
from pathlib import Path
import time

from utils.Constants import Connection, Proprioception


def grasp(leap_hand):
    grip = Proprioception.grip
    current_positions = grip["start_angles"].copy()
    stopped_fingers = [False] * 4
    log = []
    read_failures = 0

    leap_hand.set_torque_enabled(True)
    leap_hand.set_goal_positions_degree(current_positions)
    time.sleep(Proprioception.settle_time_s)

    for step in range(Proprioception.max_grasp_steps):
        state = leap_hand.get_state()
        real_state = state.get("real") if isinstance(state, dict) else state
        if not isinstance(real_state, (tuple, list)) or len(real_state) != 3:
            read_failures += 1
            if read_failures >= Proprioception.max_read_failures:
                raise RuntimeError("Unable to read the real hand state")
            time.sleep(Proprioception.step_time_s)
            continue

        read_failures = 0
        positions, velocities, currents = real_state
        if len(positions) != 16 or len(velocities) != 16 or len(currents) != 16:
            raise ValueError("Expected state for all 16 joints")

        for finger in range(4):
            if stopped_fingers[finger]:
                continue

            first_joint = finger * 4
            finger_joints = range(first_joint, first_joint + 4)
            for joint in finger_joints:
                current = currents[joint]
                limit = grip["max_currents"][joint]
                moving_positive = grip["step_sizes"][joint] > 0
                if (moving_positive and current > limit) or (not moving_positive and current < -limit):
                    current_positions[first_joint:first_joint + 4] = list(positions[first_joint:first_joint + 4])
                    stopped_fingers[finger] = True
                    print(f"Current limit exceeded on joint {joint}: {current}")
                    print(f"Stopped finger {finger}")
                    break

            if stopped_fingers[finger]:
                continue

            for joint in finger_joints:
                target = grip["max_angles"][joint]
                increment = grip["step_sizes"][joint]
                if increment > 0:
                    current_positions[joint] = min(current_positions[joint] + increment, target)
                else:
                    current_positions[joint] = max(current_positions[joint] + increment, target)

            stopped_fingers[finger] = all(
                abs(current_positions[joint] - grip["max_angles"][joint]) < 1
                for joint in finger_joints
            )

        leap_hand.set_goal_positions_degree(current_positions)
        log.append([time.time(), [float(value) for value in positions],
                    [float(value) for value in velocities],
                    [float(value) for value in currents]])

        if all(stopped_fingers):
            print("Grasp Completed")
            break
        time.sleep(Proprioception.step_time_s)
    else:
        raise RuntimeError("Grasp exceeded the maximum number of steps")

    time.sleep(Proprioception.step_time_s)
    final_state = leap_hand.get_state()
    final_state = final_state.get("real") if isinstance(final_state, dict) else final_state
    if not isinstance(final_state, (tuple, list)) or len(final_state) != 3:
        raise RuntimeError("Unable to read the final hand angles")
    final_positions = [float(value) for value in final_state[0]]
    log.append([time.time(), final_positions,
                [float(value) for value in final_state[1]],
                [float(value) for value in final_state[2]]])

    output_folder = Path(Proprioception.data_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(Proprioception.grasp_log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "positions", "velocities", "currents"])
        for timestamp, positions, velocities, currents in log:
            writer.writerow([timestamp, json.dumps(positions), json.dumps(velocities), json.dumps(currents)])

    with open(Proprioception.grasp_result_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([f"joint_{joint}" for joint in range(16)])
        writer.writerow(final_positions)

    return final_positions


def main():
    from utils.ExARM import ExArm

    leap_hand = ExArm(
        mode=Proprioception.hand_mode,
        ids=Connection.ids,
        port=Connection.Port,
        baudrate=Connection.baudrate,
        offsets=Connection.offsets,
        model_path=Connection.model_path,
    )
    try:
        grasp(leap_hand)
    finally:
        leap_hand.close()


if __name__ == "__main__":
    main()
