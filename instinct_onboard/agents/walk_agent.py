from __future__ import annotations

import os

import numpy as np
import onnxruntime as ort

from instinct_onboard.agents.base import AgentStatus, OnboardAgent
from instinct_onboard.normalizer import Normalizer
from instinct_onboard.ros_nodes.base import RealNode


class WalkAgent(OnboardAgent):
    """A simple walk agent that uses only actor.onnx for continuous walking."""

    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
    ):
        super().__init__(logdir, ros_node)
        if not hasattr(ros_node, "base_velocity_cmd"):
            raise AttributeError(
                "ros_node has no attribute 'base_velocity_cmd'. "
                "The entry script must set this attribute on the node before constructing the agent. "
                "Example: ros_node.base_velocity_cmd = np.zeros(3, dtype=np.float32) "
                "and update it each step from joystick, autonomous planner, etc."
            )
        self.ort_sessions = dict()
        self._parse_obs_config()
        self._parse_action_config()
        self._load_models()

    def _load_models(self):
        """Load the ONNX model for the agent."""
        # load ONNX models
        ort_execution_providers = ort.get_available_providers()
        actor_path = os.path.join(self.logdir, "exported", "actor.onnx")
        self.ort_sessions["actor"] = ort.InferenceSession(actor_path, providers=ort_execution_providers)
        print(f"Loaded ONNX models from {self.logdir}")
        # optionally load the normalizer if it exists
        normalizer_path = os.path.join(self.logdir, "exported", "policy_normalizer.npz")
        if os.path.exists(normalizer_path):
            self.normalizer = Normalizer(load_path=normalizer_path)
        else:
            self.normalizer = None

    def reset(self):
        """Reset the agent state."""
        super().reset()

    def step(self):
        """Perform a single step of the agent."""
        obs = self._get_observation()
        if self.normalizer is not None:
            normalized_obs = self.normalizer.normalize(obs).astype(np.float32)[None, :]
        else:
            normalized_obs = obs.astype(np.float32)[None, :]
        actor_input_name = self.ort_sessions["actor"].get_inputs()[0].name
        action = self.ort_sessions["actor"].run(None, {actor_input_name: normalized_obs})[0]
        action = action.reshape(-1)
        target_joint_state = self.pack_policy_action_to_target_joint_state(action)
        return target_joint_state, AgentStatus.Working

    """
    Agent specific observation functions for WalkAgent.
    """

    def _get_base_velocity_cmd_obs(self):
        """Return shape: (3,) — reads the generic velocity command buffer.

        The entry script is responsible for populating
        ``ros_node.base_velocity_cmd`` from the chosen source before each
        agent step.
        """
        return self.ros_node.base_velocity_cmd
