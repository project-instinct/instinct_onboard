from __future__ import annotations

import math
import os
import time
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
import prettytable
import ros2_numpy as rnp
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

from instinct_onboard.agents.base import AgentStatus, OnboardAgent
from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.utils import CircularBuffer


class ParkourAgent(OnboardAgent):
    rs_resolution = (480, 270)
    rs_frequency = 60

    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
        depth_vis: bool = True,
        pointcloud_vis: bool = True,
        lin_vel_deadband=0.5,
        lin_vel_range=[0.5, 0.5],
        ang_vel_deadband=0.15,
        ang_vel_range=[0.0, 1.0],
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self.lin_vel_deadband = lin_vel_deadband
        self.ang_vel_deadband = ang_vel_deadband
        self.cmd_px_range = lin_vel_range
        self.cmd_nx_range = [0.0, 0.0]
        self.cmd_py_range = [0.0, 0.0]
        self.cmd_ny_range = [0.0, 0.0]
        self.cmd_pyaw_range = ang_vel_range
        self.cmd_nyaw_range = ang_vel_range
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()
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

    def _parse_obs_config(self):
        super()._parse_obs_config()
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if "depth" not in obs_name]
        print(f"ParkourAgent proprioception names: {self.proprio_obs_names}")
        self.depth_obs_names = [obs_name for obs_name in all_obs_names if "depth" in obs_name]
        assert len(self.depth_obs_names) == 1, "Only support one depth observation for now."
        print(f"ParkourAgent depth observation names: {self.depth_obs_names}")
        table = prettytable.PrettyTable()
        table.field_names = ["Observation Name", "Function"]
        for obs_name, func in self.obs_funcs.items():
            table.add_row([obs_name, func.__name__])
        print("Observation functions:")
        print(table)
        self._parse_depth_image_config()

    def _parse_depth_image_config(self):
        self.output_resolution = [
            self.cfg["scene"]["camera"]["pattern_cfg"]["width"],
            self.cfg["scene"]["camera"]["pattern_cfg"]["height"],
        ]

        self.depth_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["depth_range"]

        if self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"]["normalize"]:
            self.depth_output_range = self.cfg["scene"]["camera"]["noise_pipeline"]["depth_normalization"][
                "output_range"
            ]
        else:
            self.depth_output_range = self.depth_range

        if "crop_and_resize" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.crop_region = self.cfg["scene"]["camera"]["noise_pipeline"]["crop_and_resize"]["crop_region"]
        if "gaussian_blur" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.gaussian_kernel_size = (
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
                self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["kernel_size"],
            )
            self.gaussian_sigma = self.cfg["scene"]["camera"]["noise_pipeline"]["gaussian_blur"]["sigma"]
        if "blind_spot" in self.cfg["scene"]["camera"]["noise_pipeline"]:
            self.blind_spot_crop = self.cfg["scene"]["camera"]["noise_pipeline"]["blind_spot"]["crop_region"]
        self.depth_width = (
            self.output_resolution[0] - self.crop_region[2] - self.crop_region[3]
            if hasattr(self, "crop_region")
            else self.output_resolution[0]
        )
        self.depth_height = (
            self.output_resolution[1] - self.crop_region[0] - self.crop_region[1]
            if hasattr(self, "crop_region")
            else self.output_resolution[1]
        )
        # For sample resize
        square_size = int(self.rs_resolution[0] // self.output_resolution[0])
        rows, cols = self.rs_resolution[1], self.rs_resolution[0]
        center_y_coords = np.arange(self.output_resolution[1]) * square_size + square_size // 2
        center_x_coords = np.arange(self.output_resolution[0]) * square_size + square_size // 2
        y_grid, x_grid = np.meshgrid(center_y_coords, center_x_coords, indexing="ij")
        valid_mask = (y_grid < rows) & (x_grid < cols)
        self.y_valid = np.clip(y_grid, 0, rows - 1)
        self.x_valid = np.clip(x_grid, 0, cols - 1)
        # For downsample history
        if "history_skip_frames" in self.cfg["observations"]["policy"]["depth_image"]["params"]:
            downsample_factor = self.cfg["observations"]["policy"]["depth_image"]["params"]["history_skip_frames"]
        else:
            downsample_factor = self.cfg["observations"]["policy"]["depth_image"]["params"]["time_downsample_factor"]
        frames = int(
            (self.cfg["scene"]["camera"]["data_histories"]["distance_to_image_plane_noised"] - 1) / downsample_factor
            + 1
        )
        sim_frequency = int(1 / self.cfg["scene"]["camera"]["update_period"])
        real_downsample_factor = int(self.rs_frequency / sim_frequency * downsample_factor)
        self.depth_obs_indices = np.linspace(-1 - real_downsample_factor * (frames - 1), -1, frames).astype(int)
        print(f"Depth observation downsample indices: {self.depth_obs_indices}")
        self.depth_image_buffer = CircularBuffer(length=self.rs_frequency)

    def _parse_observation_function(self, obs_name, obs_config):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        if obs_func == "depth_image":
            obs_name = "depth_latent"
            if hasattr(self, f"_get_{obs_name}_obs"):
                self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_name}_obs")
                return
            else:
                raise ValueError(f"Unknown observation function for observation {obs_name}")
        self.xyyaw_command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return super()._parse_observation_function(obs_name, obs_config)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        depth_encoder_path = os.path.join(self.logdir, "exported", "0-depth_encoder.onnx")
        self.ort_sessions["depth_encoder"] = ort.InferenceSession(depth_encoder_path, providers=ort_execution_providers)
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        """Perform a single step of the agent."""
        # pack actor MLP input
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(np.reshape(obs_term_value, (1, -1)).astype(np.float32))
        proprio_obs = np.concatenate(proprio_obs, axis=-1)

        depth_obs = (
            self._get_single_obs_term(self.depth_obs_names[0])
            .reshape(1, -1, self.depth_height, self.depth_width)
            .astype(np.float32)
        )
        # if self.depth_vis:
        #     self._vis_depth_obs(depth_obs.reshape(-1, self.depth_height, self.depth_width))
        if self.debug_depth_publisher is not None:
            # NOTE: the +5.0 is a empirical value to ensure the normalized obs is not negative.
            # Not using normalizer's value is to prevent further visualization code bugs.
            depth_image_msg_data = np.asanyarray(
                depth_obs[0, -1].reshape(self.depth_height, self.depth_width) * 255 * 2,
                dtype=np.uint16,
            )
            depth_image_msg = rnp.msgify(Image, depth_image_msg_data, encoding="16UC1")
            depth_image_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            depth_image_msg.header.frame_id = "realsense_depth_link"
            self.debug_depth_publisher.publish(depth_image_msg)
        if self.debug_pointcloud_publisher is not None:
            pointcloud_msg = self.ros_node.depth_image_to_pointcloud_msg(
                depth_obs[0, -1].reshape(self.depth_height, self.depth_width) * self.depth_range[1]
                + self.depth_range[0]
            )
            self.debug_pointcloud_publisher.publish(pointcloud_msg)

        depth_image_output = self.ort_sessions["depth_encoder"].run(
            None, {self.ort_sessions["depth_encoder"].get_inputs()[0].name: depth_obs}
        )[0]
        # run actor MLP
        actor_input = np.concatenate([proprio_obs, depth_image_output], axis=1)
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: actor_input})[0]
        action = action.reshape(-1)
        target_joint_state = self.pack_policy_action_to_target_joint_state(action)
        return target_joint_state, AgentStatus.Working

    """
    Agent specific observation functions for Parkour Agent.
    """

    def _get_base_velocity_cmd_obs(self):
        """Return shape: (3,)"""
        # left-y for forward/backward
        ly = self.ros_node.joy_stick_data.ly
        if ly > self.lin_vel_deadband:
            vx = (ly - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (0, 1)
            vx = vx * (self.cmd_px_range[1] - self.cmd_px_range[0]) + self.cmd_px_range[0]
        elif ly < -self.lin_vel_deadband:
            vx = (ly + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)  # (-1, 0)
            vx = vx * (self.cmd_nx_range[1] - self.cmd_nx_range[0]) - self.cmd_nx_range[0]
        else:
            vx = 0
        # left-x for side moving left/right
        lx = -self.ros_node.joy_stick_data.lx
        if lx > self.lin_vel_deadband:
            vy = (lx - self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
            vy = vy * (self.cmd_py_range[1] - self.cmd_py_range[0]) + self.cmd_py_range[0]
        elif lx < -self.lin_vel_deadband:
            vy = (lx + self.lin_vel_deadband) / (1 - self.lin_vel_deadband)
            vy = vy * (self.cmd_ny_range[1] - self.cmd_ny_range[0]) - self.cmd_ny_range[0]
        else:
            vy = 0
        # right-x for turning left/right
        rx = -self.ros_node.joy_stick_data.rx
        if rx > self.ang_vel_deadband:
            yaw = (rx - self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
            yaw = yaw * (self.cmd_pyaw_range[1] - self.cmd_pyaw_range[0]) + self.cmd_pyaw_range[0]
        elif rx < -self.ang_vel_deadband:
            yaw = (rx + self.ang_vel_deadband) / (1 - self.ang_vel_deadband)
            yaw = yaw * (self.cmd_nyaw_range[1] - self.cmd_nyaw_range[0]) - self.cmd_nyaw_range[0]
        else:
            yaw = 0

        self.xyyaw_command = np.array([vx, vy, yaw], dtype=np.float32)
        return self.xyyaw_command

    def _get_joint_vel_rel_obs(self):
        """Return shape: (num_joints,)"""
        return self.ros_node.joint_vel_

    def refresh_depth_frame(self):
        """Return the depth image."""
        self.ros_node.refresh_rs_data()
        depth_image_np: np.ndarray = self.ros_node.rs_depth_data
        # normalize based on given range
        depth_image = cv2.resize(depth_image_np, self.output_resolution, interpolation=cv2.INTER_NEAREST)

        if hasattr(self, "crop_region"):
            shape = depth_image.shape
            x1, x2, y1, y2 = self.crop_region
            depth_image = depth_image[x1 : shape[0] - x2, y1 : shape[1] - y2]

        mask = (depth_image < 0.2).astype(np.uint8)
        depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)

        if hasattr(self, "blind_spot_crop"):
            shape = depth_image.shape
            x1, x2, y1, y2 = self.blind_spot_crop
            depth_image[:x1, :] = 0
            depth_image[shape[0] - x2 :, :] = 0
            depth_image[:, :y1] = 0
            depth_image[:, shape[1] - y2 :] = 0
        if hasattr(self, "gaussian_kernel_size"):
            depth_image = cv2.GaussianBlur(
                depth_image, self.gaussian_kernel_size, self.gaussian_sigma, self.gaussian_sigma
            )

        filt_m = np.clip(depth_image, self.depth_range[0], self.depth_range[1])
        filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])

        output_norm = filt_norm * (self.depth_output_range[1] - self.depth_output_range[0]) + self.depth_output_range[0]
        self.depth_image_buffer.append(output_norm)

    def _get_delayed_visualizable_image_obs(self):
        return self._get_depth_image_downsample_obs()

    def _get_depth_image_downsample_obs(self):
        self.refresh_depth_frame()
        return self.depth_image_buffer.buffer[self.depth_obs_indices, ...]

    def _vis_depth_obs(self, depth_obs: np.ndarray):
        depth_tiles = (np.clip(depth_obs, 0.0, 1.0) * 255).astype(np.uint8)
        rows, cols = 2, 4
        tile_h, tile_w = depth_tiles.shape[1], depth_tiles.shape[2]
        grid = np.zeros((rows * tile_h, cols * tile_w), dtype=np.uint8)
        for idx in range(depth_tiles.shape[0]):
            r, c = divmod(idx, cols)
            grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = depth_tiles[idx]
        cv2.imwrite("depth_obs_grid.png", grid)


class ParkourStandAgent(ParkourAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
    ):
        super().__init__(logdir, ros_node, depth_vis=False, pointcloud_vis=False)

    def _get_depth_image_downsample_obs(self):
        return np.zeros([len(self.depth_obs_indices), self.depth_height, self.depth_width])
