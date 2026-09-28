import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.LeapHand import LeapHand
from utils.LeapKinematics import LeapKinematics


@dataclass
class ProbeSettings:
    sample_hz: float = 30.0
    step_deg: float = 1.0
    settle_s: float = 0.08

    # Calibrate these from unloaded closing trials.
    contact_current_threshold: float = 80.0
    tracking_error_threshold_deg: float = 4.0
    required_contact_samples: int = 3

    # Conservative safety limits; calibrate for your hardware.
    max_current: float = 250.0
    max_steps: int = 50


def is_contact(
    commanded_deg: np.ndarray,
    measured_deg: np.ndarray,
    currents: np.ndarray,
    settings: ProbeSettings,
) -> np.ndarray:
    """Returns one Boolean contact estimate per joint."""
    tracking_error = np.abs(commanded_deg - measured_deg)

    return (
        (np.abs(currents) >= settings.contact_current_threshold)
        & (tracking_error >= settings.tracking_error_threshold_deg)
    )


def collect_probe_trial(
    hand: LeapHand,
    kin: LeapKinematics,
    label: str,
    open_pose_deg: np.ndarray,
    closing_delta_deg: np.ndarray,
    output_dir: str = "Data/proprioception_trials",
    settings: ProbeSettings = ProbeSettings(),
):
    """
    Execute one slow close-to-contact probe and save proprioceptive data.

    Assumptions:
      - all arrays use the project's logical 16-joint ordering;
      - get_state() returns positions in degrees, velocities, and currents;
      - closing_delta_deg contains the total intended motion for each joint.
    """
    open_pose_deg = np.asarray(open_pose_deg, dtype=float)
    closing_delta_deg = np.asarray(closing_delta_deg, dtype=float)

    if open_pose_deg.shape != (16,) or closing_delta_deg.shape != (16,):
        raise ValueError("open_pose_deg and closing_delta_deg must both contain 16 values.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hand.set_goal_positions_degree(open_pose_deg)
    time.sleep(1.0)

    contact_counts = np.zeros(16, dtype=int)
    contact_q_deg = np.full(16, np.nan)

    log_time = []
    log_commanded = []
    log_measured = []
    log_velocities = []
    log_currents = []
    log_contacts = []

    for step in range(settings.max_steps):
        fraction = min((step + 1) * settings.step_deg / 100.0, 1.0)
        commanded_deg = open_pose_deg + fraction * closing_delta_deg

        hand.set_goal_positions_degree(commanded_deg)
        time.sleep(settings.settle_s)

        measured_deg, velocities, currents = hand.get_state()

        measured_deg = np.asarray(measured_deg, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        currents = np.asarray(currents, dtype=float)

        if np.any(np.abs(currents) > settings.max_current):
            print("Safety stop: current limit exceeded.")
            break

        contact_now = is_contact(
            commanded_deg, measured_deg, currents, settings
        )

        contact_counts = np.where(contact_now, contact_counts + 1, 0)
        stable_contact = contact_counts >= settings.required_contact_samples

        newly_detected = stable_contact & np.isnan(contact_q_deg)
        contact_q_deg[newly_detected] = measured_deg[newly_detected]

        log_time.append(time.time())
        log_commanded.append(commanded_deg.copy())
        log_measured.append(measured_deg.copy())
        log_velocities.append(velocities.copy())
        log_currents.append(currents.copy())
        log_contacts.append(stable_contact.copy())

        # Stop once at least one joint on each of index, middle, ring and thumb
        # has a stable contact indication.
        per_finger_contact = np.array([
            stable_contact[0:4].any(),
            stable_contact[4:8].any(),
            stable_contact[8:12].any(),
            stable_contact[12:16].any(),
        ])

        if per_finger_contact.all():
            print("Four-finger contact pattern acquired.")
            break

    # Where no joint reached contact, use the final measured state so the trial
    # is still complete and can be filtered later.
    final_q_deg = np.asarray(log_measured[-1], dtype=float)
    q_at_contact_deg = np.where(np.isnan(contact_q_deg), final_q_deg, contact_q_deg)

    fingertip_positions_m = kin.fk_degree(q_at_contact_deg)

    timestamp = int(time.time() * 1000)
    filename = output_path / f"{label}_{timestamp}.npz"

    np.savez_compressed(
        filename,
        label=label,
        time=np.asarray(log_time),
        commanded_deg=np.asarray(log_commanded),
        measured_deg=np.asarray(log_measured),
        velocities=np.asarray(log_velocities),
        currents=np.asarray(log_currents),
        contact_flags=np.asarray(log_contacts),
        q_at_contact_deg=q_at_contact_deg,
        fingertip_positions_m=fingertip_positions_m,
    )

    print(f"Saved trial: {filename}")
    print("Fingertip positions at final/contact configuration (m):")
    print(fingertip_positions_m)

    return filename