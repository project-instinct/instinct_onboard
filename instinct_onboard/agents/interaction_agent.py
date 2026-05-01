"""Interaction deployment aligned with the sitting checkpoint truth.

`Instinct-Interaction-G1-v0` is deployed using the sitting checkpoint layout and
observation contract. The onboard implementation therefore reuses the mature
perceptive tracker pipeline:

raw policy obs -> policy normalizer -> depth slice -> depth encoder -> actor
"""

from __future__ import annotations

import os

from instinct_onboard.agents.tracking_agent import MotionData, PerceptiveTrackerAgent
from instinct_onboard.ros_nodes.base import RealNode


class InteractionAgent(PerceptiveTrackerAgent):
    """Thin interaction wrapper over the sitting-compatible perceptive tracker."""

    REQUIRED_FILES = (
        ("params", "env.yaml"),
        ("params", "agent.yaml"),
        ("exported", "actor.onnx"),
        ("exported", "0-depth_image.onnx"),
        ("exported", "policy_normalizer.npz"),
    )
    EXPECTED_POLICY_OBS = [
        "joint_pos_ref",
        "joint_vel_ref",
        "position_ref",
        "rotation_ref",
        "projected_gravity",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "last_action",
        "depth_image",
    ]
    EXPECTED_RAW_OBS_DIM = 1339
    EXPECTED_DEPTH_DIM = 18 * 32
    EXPECTED_DEPTH_IMAGE_RESOLUTION = (18, 32)
    EXPECTED_DEPTH_ENCODER_INPUT_SHAPE = [1, 1, 18, 32]
    EXPECTED_DEPTH_ENCODER_OUTPUT_DIM = 32
    EXPECTED_ACTOR_INPUT_DIM = 795
    POLICY_JOINT_NAMES = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]

    def __init__(
        self,
        logdir: str,
        motion_file_dir: str,
        ros_node: RealNode,
        depth_vis: bool = True,
        pointcloud_vis: bool = True,
        target_motion_framerate: float = 50.0,
    ):
        self._validate_checkpoint_layout(logdir)
        super().__init__(
            logdir=logdir,
            motion_file_dir=motion_file_dir,
            ros_node=ros_node,
            target_motion_framerate=target_motion_framerate,
            depth_vis=depth_vis,
            pointcloud_vis=pointcloud_vis,
        )
        self._validate_sitting_policy_contract()

    def _get_policy_joint_names(self) -> list[str]:
        return list(self.POLICY_JOINT_NAMES)

    @classmethod
    def _validate_checkpoint_layout(cls, logdir: str) -> None:
        missing = [
            os.path.join(logdir, *parts)
            for parts in cls.REQUIRED_FILES
            if not os.path.exists(os.path.join(logdir, *parts))
        ]
        if missing:
            raise FileNotFoundError(
                "Interaction sitting checkpoint is incomplete. Missing: " + ", ".join(missing)
            )

    def _validate_sitting_policy_contract(self) -> None:
        policy_obs_names = list(self.obs_funcs.keys())
        if policy_obs_names != self.EXPECTED_POLICY_OBS:
            raise ValueError(
                "Interaction policy observation order mismatch. "
                f"Expected {self.EXPECTED_POLICY_OBS}, got {policy_obs_names}."
            )

        if self.normalizer.mean.shape[0] != self.EXPECTED_RAW_OBS_DIM:
            raise ValueError(
                "Interaction raw observation dimension mismatch. "
                f"Expected {self.EXPECTED_RAW_OBS_DIM}, got {self.normalizer.mean.shape[0]}."
            )
        height, width = self.depth_image_final_resolution
        if (height, width) != self.EXPECTED_DEPTH_IMAGE_RESOLUTION:
            raise ValueError(
                "Interaction depth image resolution mismatch. "
                f"Expected {self.EXPECTED_DEPTH_IMAGE_RESOLUTION}, got {(height, width)}."
            )
        if height * width != self.EXPECTED_DEPTH_DIM:
            raise ValueError(
                "Interaction depth image size mismatch. "
                f"Expected {self.EXPECTED_DEPTH_DIM}, got {height * width}."
            )

        depth_encoder_input_shape = list(self.ort_sessions["depth_image_encoder"].get_inputs()[0].shape)
        if depth_encoder_input_shape != self.EXPECTED_DEPTH_ENCODER_INPUT_SHAPE:
            raise ValueError(
                "Interaction depth encoder input shape mismatch. "
                f"Expected {self.EXPECTED_DEPTH_ENCODER_INPUT_SHAPE}, got {depth_encoder_input_shape}."
            )

        depth_encoder_output_shape = self.ort_sessions["depth_image_encoder"].get_outputs()[0].shape
        depth_latent_dim = depth_encoder_output_shape[-1]
        if depth_latent_dim != self.EXPECTED_DEPTH_ENCODER_OUTPUT_DIM:
            raise ValueError(
                "Interaction depth encoder output dimension mismatch. "
                f"Expected {self.EXPECTED_DEPTH_ENCODER_OUTPUT_DIM}, got {depth_latent_dim}."
            )

        actor_input_dim = self.ort_sessions["actor"].get_inputs()[0].shape[-1]
        expected_actor_input_dim = (
            self.normalizer.mean.shape[0] - self.EXPECTED_DEPTH_DIM + self.EXPECTED_DEPTH_ENCODER_OUTPUT_DIM
        )
        if actor_input_dim != expected_actor_input_dim:
            raise ValueError(
                "Interaction actor input dimension mismatch. "
                f"Expected {expected_actor_input_dim}, got {actor_input_dim}."
            )
        if actor_input_dim != self.EXPECTED_ACTOR_INPUT_DIM:
            raise ValueError(
                "Interaction actor input dimension changed unexpectedly. "
                f"Expected {self.EXPECTED_ACTOR_INPUT_DIM}, got {actor_input_dim}."
            )


__all__ = ["InteractionAgent", "MotionData"]
