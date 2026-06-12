from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort
import prettytable
import quaternion
import yaml
from geometry_msgs.msg import PoseArray

from instinct_onboard.agents.action_term import (
    JointPositionAction,
    build_default_uncontrolled_joint_action,
    get_policy_action_dim,
)
from instinct_onboard.agents.base import AgentStatus, OnboardAgent
from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.utils import quat_to_tan_norm_batch
from motion_target_msgs.msg import MotionSequence


class ShadowingAgent(OnboardAgent):
    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
    ):
        super().__init__(logdir, ros_node)
        self.ort_sessions = dict()
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()

    def _parse_obs_config(self):
        super()._parse_obs_config()
        self.rotation_reference_in_base_frame = self.cfg["commands"]["rotation_ref_command"]["in_base_frame"]
        self.ros_node.get_logger().info(f"Rotation reference in base frame: {self.rotation_reference_in_base_frame}")
        with open(os.path.join(self.logdir, "params", "agent.yaml")) as f:
            self.agent_cfg = yaml.unsafe_load(f)
        self.motion_ref_obs_names = self.agent_cfg["policy"]["encoder_configs"]["motion_ref"]["component_names"]
        self.ros_node.get_logger().info(f"ShadowingAgent motion reference names: {self.motion_ref_obs_names}")
        all_obs_names = list(self.obs_funcs.keys())
        self.proprio_obs_names = [obs_name for obs_name in all_obs_names if obs_name not in self.motion_ref_obs_names]
        self.ros_node.get_logger().info(f"ShadowingAgent proprioception names: {self.proprio_obs_names}")
        table = prettytable.PrettyTable()
        table.field_names = ["Observation Name", "Function"]
        for obs_name, func in self.obs_funcs.items():
            table.add_row([obs_name, func.__name__])
        print("Observation functions:")
        print(table)

    def _parse_observation_function(self, obs_name, obs_config):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        if obs_func == "command_mask":
            command_name = obs_config["params"]["command_name"]
            if hasattr(self, f"_get_{command_name}_mask_obs"):
                self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_mask_obs")
                return
            else:
                raise ValueError(f"Unknown observation function for observation {obs_name}")
        return super()._parse_observation_function(obs_name, obs_config)

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        motion_ref_path = os.path.join(self.logdir, "exported", "0-motion_ref.onnx")
        self.ort_sessions["motion_ref"] = ort.InferenceSession(motion_ref_path, providers=ort_execution_providers)
        fk_path = os.path.join(self.logdir, "exported", "forward_kinematics.onnx")
        self.ort_sessions["fk"] = ort.InferenceSession(fk_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")

    def _update_links_poses(self):
        """Update the current link positions based on self.ros_node.joint_pos_."""
        # get the current joint positions
        joint_pos = self.ros_node.joint_pos_
        # run forward kinematics to get the link positions
        fk_input_name = self.ort_sessions["fk"].get_inputs()[0].name
        output = self.ort_sessions["fk"].run(None, {fk_input_name: joint_pos[None, :]})
        link_pos, link_quat = output[0][0], output[1][0]  # link_pos: (num_links, 3), link_quat: (num_links, 4)
        self.link_pos_ = link_pos
        self.link_quat_ = link_quat
        self.link_tannorm_ = quat_to_tan_norm_batch(link_quat)

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        super().reset()

    def step(self):
        """Perform a single step of the agent."""
        self.ros_node.refresh_time_to_target()
        self._update_links_poses()
        # due to the model which reads the motion sequence, and then concat at the end of the proioception vector, we get obs term one by one.

        # pack all motion sequence obs term
        motion_ref_obs = []
        for motion_ref_obs_name in self.motion_ref_obs_names:
            obs_term_value = self._get_single_obs_term(motion_ref_obs_name)
            time_dim = obs_term_value.shape[0]  # (time, batch_size, ...)
            motion_ref_obs.append(
                obs_term_value.reshape(1, time_dim, -1).astype(np.float32)
            )  # reshape to (batch_size, time, -1)
        motion_ref_obs = np.concatenate(
            motion_ref_obs, axis=-1
        )  # across time dimension. shape (batch_size, time, num_obs_terms)

        # run motion reference encoder
        motion_ref_input_name = self.ort_sessions["motion_ref"].get_inputs()[0].name
        motion_ref_output = self.ort_sessions["motion_ref"].run(None, {motion_ref_input_name: motion_ref_obs})[0]

        # pack actor MLP input
        proprio_obs = []
        for proprio_obs_name in self.proprio_obs_names:
            obs_term_value = self._get_single_obs_term(proprio_obs_name)
            proprio_obs.append(obs_term_value.reshape(1, -1).astype(np.float32))
        proprio_obs.append(motion_ref_output.reshape(1, -1).astype(np.float32))  # append motion reference output
        proprio_obs = np.concatenate(proprio_obs, axis=-1)

        # run actor MLP
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: proprio_obs})[0]
        action = action.reshape(-1)
        done = (
            self.ros_node.packed_motion_sequence_buffer["time_to_target"] < 0.0
        ).all()  # done if all time_to_target are negative

        target_joint_state = self.pack_policy_action_to_target_joint_state(action)
        return target_joint_state, AgentStatus.Ended if done else AgentStatus.Working

    """
    Agent specific observation functions for Shadowing Agent.
    """

    def _get_link_pos_b_obs(self):
        """Return shape: (num_links, 3)"""
        return self.link_pos_

    def _get_link_tannorm_b_obs(self):
        """Return shape: (num_links, 6) in tangent-normal form"""
        return self.link_tannorm_

    def _get_root_tannorm_w_obs(self):
        """Return the root link's tangent-normal form in world frame.
        Return shape: (6,)
        """
        quat_w = self.ros_node._get_quat_w_obs().reshape(1, 4)  # (1, 4)
        root_tannorm_w = quat_to_tan_norm_batch(quat_w).reshape(
            6,
        )  # (6,)
        return root_tannorm_w

    def _get_time_to_target_command_cmd_obs(self) -> np.ndarray:
        """Return shape: (num_frames, 1)"""
        return self.ros_node.packed_motion_sequence_buffer["time_to_target"].reshape(-1, 1)  # (num_frames, 1)

    def _get_time_from_reference_update_obs(self):
        """Return shape: (num_frames, 1)"""
        return np.array(
            [
                (self.ros_node.get_clock().now().nanoseconds - self.ros_node.motion_sequence_receive_time.nanoseconds)
                / 1e9
            ]
            * self.ros_node.packed_motion_sequence_buffer["time_to_target"].shape[0],
            dtype=np.float32,
        )[
            :, None
        ]  # (num_frames, 1)

    def _get_position_ref_command_cmd_obs(self):
        """Command, return shape: (num_frames, 3)"""
        return self.ros_node.packed_motion_sequence_buffer["root_pos_b"]

    def _get_rotation_ref_command_cmd_obs(self):
        """Command, return shape: (num_frames, 6)"""
        root_quat_w_ref_ = self.ros_node.packed_motion_sequence_buffer["root_quat_w"]  # (num_frames, 4)
        if self.rotation_reference_in_base_frame:
            root_quat_w_ = self.ros_node._get_quat_w_obs()[None, :]  # (1, 4)
            root_quat_w_ref = quaternion.from_float_array(root_quat_w_ref_)  # (num_frames, 4)
            root_quat_w = quaternion.from_float_array(root_quat_w_)  # (1, 4)
            root_quat_err = root_quat_w.conjugate() * root_quat_w_ref  # (num_frames, 4)
            root_tannorm_cmd = quat_to_tan_norm_batch(root_quat_err)  # (num_frames, 6)
        else:
            root_tannorm_cmd = quat_to_tan_norm_batch(root_quat_w_ref_)  # (num_frames, 6)
        return root_tannorm_cmd  # (num_frames, 6), in tangent-normal form

    def _get_position_ref_command_mask_cmd_obs(self):
        """Command, return shape: (num_frames, 2)"""
        return self.ros_node.packed_motion_sequence_buffer["pose_mask"][:, :2]

    def _get_rotation_ref_command_mask_cmd_obs(self):
        """Command, return shape: (num_frames, 2)"""
        return self.ros_node.packed_motion_sequence_buffer["pose_mask"][:, 2:]

    # def _get_pose_ref_mask_obs(self):
    #     return self.ros_node.packed_motion_sequence_buffer["pose_mask"]  # (num_frames, 4)

    def _get_joint_pos_ref_command_cmd_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return (
            self.ros_node.packed_motion_sequence_buffer["joint_pos"] - self.ros_node.default_joint_pos[None, :]
        )  # (num_frames, num_joints)

    def _get_joint_pos_err_ref_command_cmd_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return (
            self.ros_node.packed_motion_sequence_buffer["joint_pos"] - self.ros_node.joint_pos_[None, :]
        )  # (num_frames, num_joints)

    def _get_joint_pos_ref_command_mask_cmd_obs(self):
        """Command, return shape: (num_frames, num_joints)"""
        return self.ros_node.packed_motion_sequence_buffer["joint_pos_mask"]  # (num_frames, num_joints)

    def _get_link_pos_ref_command_cmd_obs(self):
        return self.ros_node.packed_motion_sequence_buffer["link_pos"]  # (num_frames, num_links, 3), in robot base link

    def _get_link_pos_err_ref_command_cmd_obs(self):
        return (
            self.ros_node.packed_motion_sequence_buffer["link_pos"] - self.link_pos_[None, :, :]
        )  # (num_frames, num_links, 3)

    def _get_link_pos_ref_command_mask_cmd_obs(self):
        return self.ros_node.packed_motion_sequence_buffer["link_pos_mask"]  # (num_frames, num_links)

    def _get_link_rot_ref_command_cmd_obs(self):
        return self.ros_node.packed_motion_sequence_buffer[
            "link_tannorm"
        ]  # (num_frames, num_links, 6), in robot base link

    def _get_link_rot_err_ref_command_cmd_obs(self):
        link_quat_ref_ = self.ros_node.packed_motion_sequence_buffer["link_quat"]  # (num_frames, num_links, 4)
        link_quat_ = self.link_quat_[None, :, :]  # (1, num_links, 4)
        link_quat_ref = quaternion.from_float_array(link_quat_ref_)  # (num_frames, num_links, quaternion(4))
        link_quat = quaternion.from_float_array(link_quat_)  # (1, num_links, quaternion(4))
        link_rot_err = link_quat.conjugate() * link_quat_ref  # (num_frames, num_links, quaternion(4))
        link_rot_err_ = link_rot_err.reshape(-1)  # (num_frames * num_links, quaternion(4))
        link_tannorm_err_ = quat_to_tan_norm_batch(link_rot_err_)
        link_tannorm_err = link_tannorm_err_.reshape(*link_rot_err.shape[:2], 6)
        return link_tannorm_err  # (num_frames, num_links, 6)

    def _get_link_rot_ref_command_mask_cmd_obs(self):
        return self.ros_node.packed_motion_sequence_buffer["link_quat_mask"]  # (num_frames, num_links)


class MotionAsActAgent(OnboardAgent):
    """An agent that only output joint_pos of motion reference as action.
    If the current joint_pos is far from the motion reference, it will output a scaled version of the difference.
    """

    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
        joint_diff_threshold: float = 0.2,
        joint_diff_scale: float = 0.2,
    ):
        super().__init__(logdir, ros_node)
        # NO parse_obs_config and _load_models because this agent use hand-coded logic.
        self.joint_diff_threshold = joint_diff_threshold
        self.joint_diff_scale = joint_diff_scale
        self._parse_action_config()

    def _parse_action_config(self):
        """Use identity full-joint position mapping; ignore ``cfg['actions']`` from the export."""
        self._load_robot_init_state_and_actuator_pd()
        self._action_terms = [
            JointPositionAction(
                name="motion_as_act_full_joint_position",
                action_cfg={
                    "joint_names": [".*"],
                    "scale": 1.0,
                    "offset": 0.0,
                    "use_default_offset": False,
                },
                ros_node=self.ros_node,
                default_joint_pos=self.default_joint_pos,
                p_gains=self._p_gains,
                d_gains=self._d_gains,
                action_cursor=0,
            )
        ]
        self._policy_action_dim = get_policy_action_dim(self._action_terms)
        self._default_uncontrolled_joint_action = build_default_uncontrolled_joint_action(
            ros_node=self.ros_node,
            default_joint_pos=self.default_joint_pos,
            p_gains=self._p_gains,
            d_gains=self._d_gains,
            action_terms=self._action_terms,
        )
        self._summarize_action_terms()

    def reset(self):
        """Reset the agent state and the rosbag reader."""
        pass

    def step(self):
        target_joint_pos = self.ros_node.packed_motion_sequence_buffer["joint_pos"][0]  # (num_joints,)
        current_joint_pos = self.ros_node.joint_pos_
        joint_diff = target_joint_pos - current_joint_pos
        command_joint_pos = target_joint_pos.copy()
        over_threshold_mask = np.abs(joint_diff) > self.joint_diff_threshold
        command_joint_pos[over_threshold_mask] = (
            current_joint_pos[over_threshold_mask] + np.sign(joint_diff[over_threshold_mask]) * self.joint_diff_scale
        )
        print(f"Joint Error max: {np.abs(joint_diff).max():.3f}", end="\r")
        action = command_joint_pos.astype(np.float32)
        reached = not np.any(over_threshold_mask)  # reached if all joint positions are within the threshold
        target_joint_state = self.pack_policy_action_to_target_joint_state(action)
        return target_joint_state, AgentStatus.Reached if reached else AgentStatus.Working
