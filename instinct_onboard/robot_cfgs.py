import math

import numpy as np

"""A general configuration file for the robots, shared between different scripts. """


class UnitreeWirelessButtons:
    R1 = 0b00000001  # 1
    L1 = 0b00000010  # 2
    start = 0b00000100  # 4
    select = 0b00001000  # 8
    R2 = 0b00010000  # 16
    L2 = 0b00100000  # 32
    F1 = 0b01000000  # 64
    F2 = 0b10000000  # 128
    A = 0b100000000  # 256
    B = 0b1000000000  # 512
    X = 0b10000000000  # 1024
    Y = 0b100000000000  # 2048
    up = 0b1000000000000  # 4096
    right = 0b10000000000000  # 8192
    down = 0b100000000000000  # 16384
    left = 0b1000000000000000  # 32768


class G1_29Dof_TorsoBase:
    NUM_JOINTS = 29
    joint_map = [
        15,
        22,  # shoulder pitch
        14,  # waist pitch
        16,
        23,  # shoulder roll
        13,  # waist roll
        17,
        24,  # shoulder yaw
        12,  # waist yaw
        18,
        25,  # elbow
        0,
        6,  # hip pitch
        19,
        26,  # wrist roll
        1,
        7,  # hip roll
        20,
        27,  # wrist pitch
        2,
        8,  # hip yaw
        21,
        28,  # wrist yaw
        3,
        9,  # knee
        4,
        10,  # ankle pitch
        5,
        11,  # ankle roll
    ]
    sim_joint_names = [  # NOTE: order matters. This list is the order in simulation.
        "left_shoulder_pitch_joint",  #
        "right_shoulder_pitch_joint",
        "waist_pitch_joint",
        "left_shoulder_roll_joint",  #
        "right_shoulder_roll_joint",
        "waist_roll_joint",
        "left_shoulder_yaw_joint",  #
        "right_shoulder_yaw_joint",
        "waist_yaw_joint",
        "left_elbow_joint",  #
        "right_elbow_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_hip_roll_joint",  #
        "right_hip_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_wrist_yaw_joint",  #
        "right_wrist_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",  #
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]
    real_joint_names = [  # NOTE: order matters. This list is the order in real robot.
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    joint_signs = np.array(
        [
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=np.float32,
    )
    joint_limits_high = np.array(
        [
            2.6704,
            2.6704,
            0.5200,
            2.2515,
            1.5882,
            0.5200,
            2.6180,
            2.6180,
            2.6180,
            2.0944,
            2.0944,
            2.8798,
            2.8798,
            1.9722,
            1.9722,
            2.9671,
            0.5236,
            1.6144,
            1.6144,
            2.7576,
            2.7576,
            1.6144,
            1.6144,
            2.8798,
            2.8798,
            0.5236,
            0.5236,
            0.2618,
            0.2618,
        ],
        dtype=np.float32,
    )
    joint_limits_low = np.array(
        [
            -3.0892,
            -3.0892,
            -0.5200,
            -1.5882,
            -2.2515,
            -0.5200,
            -2.6180,
            -2.6180,
            -2.6180,
            -1.0472,
            -1.0472,
            -2.5307,
            -2.5307,
            -1.9722,
            -1.9722,
            -0.5236,
            -2.9671,
            -1.6144,
            -1.6144,
            -2.7576,
            -2.7576,
            -1.6144,
            -1.6144,
            -0.0873,
            -0.0873,
            -0.8727,
            -0.8727,
            -0.2618,
            -0.2618,
        ],
        dtype=np.float32,
    )
    torque_limits = np.array(
        [  # from urdf and in simulation order
            25,
            25,
            50,
            25,
            25,
            50,
            25,
            25,
            88,
            25,
            25,
            88,
            88,
            25,
            25,
            88,
            88,
            5,
            5,
            88,
            88,
            5,
            5,
            139,
            139,
            50,
            50,
            50,
            50,
        ],
        dtype=np.float32,
    )
    turn_on_motor_mode = [0x01] * 29
    mode_pr = 0
    mode_machine = 5
    """ please check this value from
        https://support.unitree.com/home/zh/G1_developer/basic_services_interface
        https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description
    """
    realsense_depth_link_transform = {
        "translation": (
            0.04764571478 + 0.0039635 - 0.0042 * math.cos(math.radians(48)),
            0.015,
            0.46268178553 - 0.044 + 0.0042 * math.sin(math.radians(48)) + 0.016,
        ),
        "rotation": (
            math.cos(math.radians(0.5) / 2) * math.cos(math.radians(48) / 2),  # w
            math.sin(math.radians(0.5) / 2),  # x
            math.sin(math.radians(48) / 2),  # y
            0.0,  # z
        ),
        "parent_frame": "torso_link",
        "child_frame": "realsense_depth_link",
    }
