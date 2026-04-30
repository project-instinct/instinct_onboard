from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


VelocityEstimation = Literal["frontward", "backward", "frontbackward"]


@dataclass
class MotionData:
    framerate: float
    # Joint orders must match the robot joint names in simulation order.
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    total_num_frames: int


def estimate_velocity_np(
    positions: np.ndarray,
    dt: float,
    estimation_type: VelocityEstimation | None = "backward",
) -> np.ndarray:
    if estimation_type is None:
        return np.zeros_like(positions, dtype=np.float32)
    if estimation_type == "frontward":
        next_frame = np.roll(positions, -1, axis=0)
        next_frame[-1] = positions[-1]
        prev_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "backward":
        prev_frame = np.roll(positions, 1, axis=0)
        prev_frame[0] = positions[0]
        next_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "frontbackward":
        prev_frame = np.roll(positions, 1, axis=0)
        prev_frame[0] = positions[0]
        next_frame = np.roll(positions, -1, axis=0)
        next_frame[-1] = positions[-1]
        velocity = (next_frame - prev_frame) / (2.0 * dt)
    else:
        raise ValueError(f"Unknown velocity estimation type: {estimation_type}")
    return velocity.astype(np.float32)


def quat_slerp_batch_np(q1: np.ndarray, q2: np.ndarray, tau: np.ndarray) -> np.ndarray:
    assert q1.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert q2.shape[-1] == 4, "The quaternion must be in (w, x, y, z) format."
    assert tau.shape == q1.shape[:-1], "The batch size must be the same for all inputs."

    q1 = q1.astype(np.float32)
    q2 = q2.astype(np.float32)
    tau = tau.astype(np.float32)

    dot_product = np.sum(q1 * q2, axis=-1, keepdims=True).clip(-1.0, 1.0)
    q2 = np.where(dot_product < 0.0, -q2, q2)
    dot_product = np.where(dot_product < 0.0, -dot_product, dot_product)

    theta = np.arccos(dot_product)
    sin_theta = np.sin(theta)
    q_too_similar = (dot_product > (1.0 - 1e-9)) | (np.abs(theta) < 1e-9)
    sin_theta = np.where(np.abs(sin_theta) < 1e-9, np.ones_like(sin_theta), sin_theta)

    s1 = np.sin((1.0 - tau)[:, None] * theta) / sin_theta
    s2 = np.sin(tau[:, None] * theta) / sin_theta
    interpolated_quat = (s1 * q1 + s2 * q2) * (~q_too_similar) + q_too_similar * q1
    interpolated_quat = interpolated_quat / np.linalg.norm(interpolated_quat, axis=-1, keepdims=True).clip(min=1e-6)
    return interpolated_quat.astype(np.float32)


def load_motion_data(
    motion_file: str,
    robot_joint_names: list[str],
    target_framerate: float,
    velocity_estimation_method: VelocityEstimation | None = "backward",
) -> MotionData:
    motion_data = np.load(motion_file, allow_pickle=True)
    framerate = motion_data["framerate"].item()

    motion_joint_names_all = motion_data["joint_names"].tolist()
    motion_joint_to_robot_joint_ids = [motion_joint_names_all.index(j_name) for j_name in robot_joint_names]

    joint_pos = motion_data["joint_pos"][:, motion_joint_to_robot_joint_ids]
    joint_vel = estimate_velocity_np(joint_pos, 1.0 / framerate, velocity_estimation_method)
    base_pos = motion_data["base_pos_w"]
    base_quat = motion_data["base_quat_w"]
    total_num_frames = motion_data["joint_pos"].shape[0]

    motion = MotionData(
        framerate=framerate,
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        base_pos=base_pos.astype(np.float32),
        base_quat=base_quat.astype(np.float32),
        total_num_frames=total_num_frames,
    )

    return match_framerate(motion, target_framerate, velocity_estimation_method)


def match_framerate(
    motion_data: MotionData,
    target_framerate: float,
    velocity_estimation_method: VelocityEstimation | None = "backward",
) -> MotionData:
    if motion_data.framerate == target_framerate:
        return motion_data

    motion_length_s = motion_data.total_num_frames / motion_data.framerate
    new_total_num_frames = math.floor(motion_length_s * target_framerate)
    new_frame_idxs = np.linspace(0, motion_data.total_num_frames - 1, new_total_num_frames)
    floor = np.floor(new_frame_idxs).astype(int)
    ceil = np.ceil(new_frame_idxs).astype(int)
    frac = new_frame_idxs - floor

    new_joint_pos = (1.0 - frac)[:, None] * motion_data.joint_pos[floor] + frac[:, None] * motion_data.joint_pos[ceil]
    new_joint_vel = estimate_velocity_np(new_joint_pos, 1.0 / target_framerate, velocity_estimation_method)
    new_base_pos = (1.0 - frac)[:, None] * motion_data.base_pos[floor] + frac[:, None] * motion_data.base_pos[ceil]
    new_base_quat = quat_slerp_batch_np(motion_data.base_quat[floor], motion_data.base_quat[ceil], frac)

    return MotionData(
        framerate=target_framerate,
        joint_pos=new_joint_pos.astype(np.float32),
        joint_vel=new_joint_vel.astype(np.float32),
        base_pos=new_base_pos.astype(np.float32),
        base_quat=new_base_quat.astype(np.float32),
        total_num_frames=new_total_num_frames,
    )
