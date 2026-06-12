from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort
import quaternion
import ros2_numpy as rnp
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent, OnboardAgent
from instinct_onboard.normalizer import Normalizer
from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.utils import (
    inv_quat,
    quat_rotate_inverse,
    quat_slerp_batch,
    quat_to_tan_norm_batch,
    yaw_quat,
)


@dataclass
class MotionData:
    framerate: float
    # Joint orders must match the robot joint names in simulation order.
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    total_num_frames: int


def load_motion_data(motion_file: str, robot_joint_names: list[str], target_framerate: float) -> MotionData:
    motion_data = np.load(motion_file, allow_pickle=True)
    framerate = motion_data["framerate"].item()

    motion_joint_names_all = motion_data["joint_names"].tolist()
    motion_joint_to_robot_joint_ids = [motion_joint_names_all.index(j_name) for j_name in robot_joint_names]

    joint_pos = motion_data["joint_pos"][:, motion_joint_to_robot_joint_ids]
    joint_pos_ = np.concatenate([joint_pos[0:1], joint_pos])
    joint_vel = (joint_pos_[1:] - joint_pos_[:-1]) * framerate
    base_pos = motion_data["base_pos_w"]
    base_quat = motion_data["base_quat_w"]
    total_num_frames = motion_data["joint_pos"].shape[0]

    motion = MotionData(
        framerate=framerate,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pos=base_pos,
        base_quat=base_quat,
        total_num_frames=total_num_frames,
    )

    return match_framerate(motion, target_framerate)


def match_framerate(motion_data: MotionData, target_framerate: float) -> MotionData:
    if motion_data.framerate == target_framerate:
        return motion_data

    motion_length_s = motion_data.total_num_frames / motion_data.framerate
    new_total_num_frames = math.floor(motion_length_s * target_framerate)
    new_frame_idxs = np.linspace(0, motion_data.total_num_frames - 1, new_total_num_frames)
    floor = np.floor(new_frame_idxs).astype(int)
    ceil = np.ceil(new_frame_idxs).astype(int)
    frac = new_frame_idxs - floor

    new_joint_pos = (1 - frac)[:, None] * motion_data.joint_pos[floor] + frac[:, None] * motion_data.joint_pos[ceil]
    joint_pos_ = np.concatenate([new_joint_pos[0:1], new_joint_pos])
    new_joint_vel = (joint_pos_[1:] - joint_pos_[:-1]) * target_framerate
    new_base_pos = (1 - frac)[:, None] * motion_data.base_pos[floor] + frac[:, None] * motion_data.base_pos[ceil]
    new_base_quat = quat_slerp_batch(motion_data.base_quat[floor], motion_data.base_quat[ceil], frac)

    return MotionData(
        framerate=target_framerate,
        joint_pos=new_joint_pos.astype(np.float32),
        joint_vel=new_joint_vel.astype(np.float32),
        base_pos=new_base_pos.astype(np.float32),
        base_quat=new_base_quat.astype(np.float32),
        total_num_frames=new_total_num_frames,
    )


class TrackerAgent(OnboardAgent):
    """Different from ShadowingAgent, this agent reads the motion file directly and does not listen from the
    motion sequence topic. And we assume the network is just a MLP.
    """

    def __init__(
        self,
        logdir: str,
        motion_file_dir: str,  # retargetted motion file
        ros_node: RealNode,
        target_motion_framerate: float = 50.0,
    ):
        super().__init__(logdir, ros_node)
        self.target_motion_framerate = target_motion_framerate
        self.ort_sessions: dict[str, ort.InferenceSession] = dict()
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()
        self._load_all_motions(motion_file_dir)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")
        # load the normalizer
        normalizer_path = os.path.join(self.logdir, "exported", "policy_normalizer.npz")
        self.normalizer = Normalizer(load_path=normalizer_path)

    def _load_all_motions(self, motion_file_dir: str):
        """Load the motion file."""
        self.all_motion_datas: dict[str, MotionData] = dict()
        for motion_file in os.listdir(motion_file_dir):
            motion = load_motion_data(
                os.path.join(motion_file_dir, motion_file), self.ros_node.sim_joint_names, self.target_motion_framerate
            )
            self.all_motion_datas[motion_file] = motion
        self.motion_data = list(self.all_motion_datas.values())[
            0
        ]  # put the first motion in the dictionary as the default motion
        self.ros_node.get_logger().info(f"Loaded {len(self.all_motion_datas)} motions from {motion_file_dir}.")

        # prepare the frame indices (offset w.r.t current cursor)
        self.motion_num_frames = self.cfg["scene"]["motion_reference"][
            "num_frames"
        ]  # the num of frames to output as reference.
        self.motion_frame_indices_offset = np.arange(self.motion_num_frames).astype(float)
        if self.cfg["scene"]["motion_reference"]["data_start_from"] == "one_frame_interval":
            self.motion_frame_indices_offset += 1
        self.motion_frame_indices_offset *= (
            self.cfg["scene"]["motion_reference"]["frame_interval_s"] * self.target_motion_framerate
        )
        self.motion_frame_indices_offset = self.motion_frame_indices_offset.astype(int)

        self.motion_cursor_idx = 0

    def reset(self, motion_name: str = None):
        """Reset the agent state and the rosbag reader."""
        super().reset()
        if motion_name is None:
            motion_name = list(self.all_motion_datas.keys())[0]
        self.motion_data = self.all_motion_datas[motion_name]
        self.match_to_current_heading()
        self.ros_node.get_logger().info(f"Reference motion {motion_name} matched to current heading.")
        self.motion_cursor_idx = 0

    def get_done(self):
        """Get the done flag."""
        return (self.motion_cursor_idx + self.motion_frame_indices_offset[-1]) >= self.motion_data.total_num_frames - 1

    def step(self):
        """Perform a single step of the agent."""
        self.motion_cursor_idx += 1
        done = self.get_done()
        self.motion_cursor_idx = (
            self.motion_cursor_idx
            if self.motion_cursor_idx < self.motion_data.total_num_frames
            else self.motion_data.total_num_frames - 1
        )
        obs = self._get_observation()
        normalized_obs = self.normalizer.normalize(obs).astype(np.float32)[None, :]
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: normalized_obs})[0]
        action = action.reshape(-1)
        target_joint_state = self.pack_policy_action_to_target_joint_state(action)
        return target_joint_state, AgentStatus.Ended if done else AgentStatus.Working

    def match_to_current_heading(self):
        """Match the motion's 0-th frame to the current heading."""
        root_quat_w = quaternion.from_float_array(self.ros_node._get_quat_w_obs())  # (,) quaternion
        quat_w_ref = quaternion.from_float_array(self.motion_data.base_quat[0])  # (,) quaternion
        quat_err = root_quat_w * inv_quat(quat_w_ref)  # (,) quaternion
        heading_err_quat = yaw_quat(quat_err)  # (,) quaternion
        heading_err_quat_ = np.stack(
            [heading_err_quat for _ in range(len(self.motion_data.base_quat))], axis=0
        )  # (N, 4)

        # update the base_quat_w for each frame
        motion_quats = quaternion.from_float_array(self.motion_data.base_quat)  # (N,) quaternion
        updated_quats = heading_err_quat_ * motion_quats  # broadcasts to (N,)
        self.motion_data.base_quat = quaternion.as_float_array(updated_quats)  # (N, 4)

        # update the base_pos_w for each frame
        current_pos_w = self.motion_data.base_pos[0]  # (3,)
        rel_pos = self.motion_data.base_pos - self.motion_data.base_pos[0:1]  # (N, 3)
        rotated_rel_pos = quaternion.rotate_vectors(heading_err_quat, rel_pos)  # (N, 3)
        self.motion_data.base_pos = rotated_rel_pos + current_pos_w[None, :]  # (N, 3)

    """
    Agent specific observation functions for TrackerAgent.
    """

    def _get_joint_pos_ref_command_cmd_obs(self):
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_data.total_num_frames - 1)
        return self.motion_data.joint_pos[frame_indices] - self.default_joint_pos[None, :]

    def _get_joint_vel_ref_command_cmd_obs(self):
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_data.total_num_frames - 1)
        return self.motion_data.joint_vel[frame_indices]

    def _get_position_b_ref_command_cmd_obs(self):
        """Return the future position reference in current motion reference's base frame."""
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_data.total_num_frames - 1)
        current_motion_base_pos = self.motion_data.base_pos[self.motion_cursor_idx : self.motion_cursor_idx + 1]
        current_motion_base_quat = self.motion_data.base_quat[self.motion_cursor_idx]  # (4,)
        future_motion_base_pos = self.motion_data.base_pos[frame_indices]
        future_motion_base_pos_b = quat_rotate_inverse(
            quaternion.from_float_array(current_motion_base_quat), future_motion_base_pos - current_motion_base_pos
        )
        return future_motion_base_pos_b  # (num_frames, 3)

    def _get_rotation_ref_command_cmd_obs(self):
        """
        Return the future rotation reference in current robot's base frame.
        """
        frame_indices = self.motion_frame_indices_offset + self.motion_cursor_idx
        frame_indices = frame_indices.clip(max=self.motion_data.total_num_frames - 1)
        current_robot_base_quat = self.ros_node._get_quat_w_obs()[None, :]  # (1, 4)
        future_motion_base_quat = self.motion_data.base_quat[frame_indices]
        future_motion_base_quat_b = inv_quat(
            quaternion.from_float_array(current_robot_base_quat)
        ) * quaternion.from_float_array(future_motion_base_quat)
        return quat_to_tan_norm_batch(future_motion_base_quat_b)  # (num_frames, 6)

    def get_cold_start_agent(self, startup_step_size: float = 0.2, kpkd_factor: float = 1.0) -> ColdStartAgent:
        """Create a ColdStartAgent with joint_target_pos set to the 0-th frame of the motion."""
        joint_target_pos = self.motion_data.joint_pos[0].copy()
        return ColdStartAgent(
            startup_step_size=startup_step_size,
            ros_node=self.ros_node,
            joint_target_pos=joint_target_pos,
            # Share action layout with the tracker so last_action / cold-start stay consistent.
            action_terms=self.action_terms,
            p_gains=self.p_gains * kpkd_factor,
            d_gains=self.d_gains * kpkd_factor,
        )


class PerceptiveTrackerAgent(TrackerAgent):

    def __init__(self, *args, depth_vis: bool = False, pointcloud_vis: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_vis = depth_vis
        if self.depth_vis:
            self.debug_depth_publisher = self.ros_node.create_publisher(Image, "/debug/depth_image", 10)
        else:
            self.debug_depth_publisher = None
        self.pointcloud_vis = pointcloud_vis
        if self.pointcloud_vis:
            self.debug_pointcloud_publisher = self.ros_node.create_publisher(PointCloud2, "/debug/pointcloud", 10)
        else:
            self.debug_pointcloud_publisher = None

    def _load_models(self):
        super()._load_models()
        ort_execution_providers = ort.get_available_providers()
        depth_image_encoder_path = os.path.join(self.logdir, "exported", "0-depth_image.onnx")
        self.ort_sessions["depth_image_encoder"] = ort.InferenceSession(
            depth_image_encoder_path, providers=ort_execution_providers
        )

    def _parse_obs_config(self):
        super()._parse_obs_config()
        # add depth image cropping and normalization configs
        sim_resolution_before_crop = (
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        )
        sim_crop_region = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"][
            "crop_region"
        ]  # up, down, left, right
        real_resolution = self.ros_node.rs_resolution  # (width, height)
        real_crop_region = (
            int(sim_crop_region[0] * real_resolution[1] / sim_resolution_before_crop[1]),  # up
            int(sim_crop_region[1] * real_resolution[1] / sim_resolution_before_crop[1]),  # down
            int(sim_crop_region[2] * real_resolution[0] / sim_resolution_before_crop[0]),  # left
            int(sim_crop_region[3] * real_resolution[0] / sim_resolution_before_crop[0]),  # right
        )
        self.depth_image_crop_region = real_crop_region  # (up, down, left, right)
        self.depth_image_final_resolution = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"][
            "resize_shape"
        ]  # (height, width)
        self.depth_image_clip_range = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"][
            "depth_range"
        ]  # (min, max)
        self.depth_image_shall_normalize = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"]["normalize"]
        self.depth_image_normalized_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["normalize"][
            "output_range"
        ]
        if "gaussian_blur_noise" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.depth_image_gaussian_blur_kernel_size = self.cfg["scene"]["camera"]["noise_pipeline"][
                "gaussian_blur_noise"
            ]["kernel_size"]
            self.depth_image_gaussian_blur_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur_noise"][
                "sigma"
            ]
        else:
            self.depth_image_gaussian_blur_kernel_size = None
            self.depth_image_gaussian_blur_sigma = None

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if not hasattr(self, "depth_image_slice"):
            self.depth_image_slice = self._get_obs_slice("depth_image")

    def step(self):
        self.motion_cursor_idx += 1
        done = self.get_done()
        self.motion_cursor_idx = (
            self.motion_cursor_idx
            if self.motion_cursor_idx < self.motion_data.total_num_frames
            else self.motion_data.total_num_frames - 1
        )
        obs = self._get_observation()
        normalized_obs = self.normalizer.normalize(obs).astype(np.float32)[None, :]
        depth_image = normalized_obs[:, self.depth_image_slice].reshape(
            1, 1, *self.depth_image_final_resolution
        )  # (1, 1, height, width)
        depth_image_encoder_input_name = self.ort_sessions["depth_image_encoder"].get_inputs()[0].name
        depth_embedding = self.ort_sessions["depth_image_encoder"].run(
            None, {depth_image_encoder_input_name: depth_image}
        )[0]
        actor_input = np.concatenate(
            [
                normalized_obs[:, : self.depth_image_slice.start],
                normalized_obs[:, self.depth_image_slice.stop :],
                depth_embedding,
            ],
            axis=-1,
        )  # (1, dim)
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: actor_input})[0]
        action = action.reshape(-1)

        if self.debug_depth_publisher is not None:
            # NOTE: the +5.0 is a empirical value to ensure the normalized obs is not negative.
            # Not using normalizer's value is to prevent further visualization code bugs.
            depth_image_msg_data = np.asanyarray(
                obs[self.depth_image_slice].reshape(*self.depth_image_final_resolution) * 255 * 2,
                dtype=np.uint16,
            )
            depth_image_msg = rnp.msgify(Image, depth_image_msg_data, encoding="16UC1")
            depth_image_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            depth_image_msg.header.frame_id = "realsense_depth_link"
            self.debug_depth_publisher.publish(depth_image_msg)
        if self.debug_pointcloud_publisher is not None:
            pointcloud_msg = self.ros_node.depth_image_to_pointcloud_msg(
                obs[self.depth_image_slice].reshape(*self.depth_image_final_resolution) * self.depth_image_clip_range[1]
                + self.depth_image_clip_range[0]
            )
            self.debug_pointcloud_publisher.publish(pointcloud_msg)
        return action, done

    """
    Agent specific observation functions for PerceptiveTrackerAgent.
    """

    def _get_visualizable_image_obs(self):
        """Return the depth image."""
        self.ros_node.refresh_rs_data()
        depth_image: np.ndarray = self.ros_node.rs_depth_data
        # normalize based on given range
        depth_image = np.clip(depth_image, self.depth_image_clip_range[0], self.depth_image_clip_range[1])
        if self.depth_image_shall_normalize:
            depth_image = (depth_image - self.depth_image_clip_range[0]) / (
                self.depth_image_clip_range[1] - self.depth_image_clip_range[0]
            )
            if self.depth_image_normalized_output_range is not None:
                depth_image = (
                    depth_image
                    * (self.depth_image_normalized_output_range[1] - self.depth_image_normalized_output_range[0])
                    + self.depth_image_normalized_output_range[0]
                )
        # gaussian blur the depth image
        if self.depth_image_gaussian_blur_kernel_size is not None:
            depth_image = cv2.GaussianBlur(
                depth_image,
                # (self.depth_image_gaussian_blur_kernel_size, self.depth_image_gaussian_blur_kernel_size),
                # self.depth_image_gaussian_blur_sigma,
                (3, 3),
                0.5,
            )
        # crop the depth image
        depth_image = depth_image[
            self.depth_image_crop_region[0] : -self.depth_image_crop_region[1],
            self.depth_image_crop_region[2] : -self.depth_image_crop_region[3],
        ]
        # resize the depth image to the final resolution
        depth_image = cv2.resize(
            depth_image,
            (self.depth_image_final_resolution[1], self.depth_image_final_resolution[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        # inpaint the depth image
        depth_image = cv2.inpaint(depth_image, (depth_image < 0.2).astype(np.uint8), 3, cv2.INPAINT_NS)
        return depth_image
