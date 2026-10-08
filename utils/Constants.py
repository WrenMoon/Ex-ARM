class Connection:
    mode = "both"
    # Port = "/dev/cu.usbserial-FTBEQQVR"
    Port = "COM3"
    baudrate = 4000000
    ids=[0, 1, 2, 3, 4, 5, 6, 7,
             8, 9, 10, 11, 12, 13, 14, 15]
    offsets=[0, 0, 0, 0, 0, 0, 0, 0,
                 0,  0,  0, 0, 0, 0, 0, 0]
    model_path="Data/Leap_Model/mujoco_robot.urdf"


class Proprioception:
    hand_mode = "real"

    grip = dict(
        start_angles=[0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 0, 0, 0,
                      0, 180, 40, 25],
        max_angles=[-75, 90, 90, 60,
                    0, 80, 80, 65,
                    75, 90, 90, 60,
                    0, 100, 110, 80],
        step_sizes=[-1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, 1, 1, 1,
                    1, -1, 1, 1],
        max_currents=[22, 80, 30, 20,
                      22, 80, 30, 20,
                      22, 80, 30, 20,
                      27, 70, 30, 20],
    )

    settle_time_s = 2
    step_time_s = 0.2
    max_grasp_steps = 250
    max_read_failures = 10

    object_folder = "Data/Objects"
    objects = [
        dict(file="cube.step", class_name="cube", size_name="side_length",
             reference_size_m=0.05),
        dict(file="sphere.step", class_name="sphere", size_name="diameter",
             reference_size_m=0.05),
        dict(file="cylinder.step", class_name="cylinder", size_name="diameter",
             reference_size_m=0.05),
    ]
    mount_translation_m = [-0.06, -0.037, -0.035]
    mount_rotation_rpy_deg = [-180, 0, 0]
    sim_joint_offsets_deg = [0] * 16
    scale_start_percent = 150
    scale_stop_percent = 50
    scale_step_percent = -5

    measurement_noise_deg = 0.5
    noisy_samples_per_grasp = 50
    validation_stride = 5
    random_seed = 7

    network_hidden_size = 32
    network_epochs = 1000
    network_learning_rate = 0.02
    size_loss_weight = 1.0
    unknown_distance_multiplier = 1.5
    unknown_probability_min = 0.75
    unknown_class_margin_deg = 2.0

    data_folder = "Data/Proprioception"
    model_path = "Data/Proprioception/model.npz"
    simulation_path = "Data/Proprioception/simulation.csv"
    training_steps_path = "Data/Proprioception/training_steps.csv"
    training_samples_path = "Data/Proprioception/training_samples.csv"
    validation_path = "Data/Proprioception/validation.csv"
    training_report_path = "Data/Proprioception/training_report.json"
    grasp_log_path = "Data/Proprioception/log.csv"
    grasp_result_path = "Data/Proprioception/grasp_results.csv"
    recognition_result_path = "Data/Proprioception/recognition_result.json"

    viewer_refresh_s = 1 / 60


class PhysicalOnly:
    data_folder = "Data/Proprioception/Physical_Only"
    default_dataset_name = "initial_9"
    trials_per_class = 3
