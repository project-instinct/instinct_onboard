import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import IntEnum
from typing import Callable, Tuple

import numpy as np
import yaml

from instinct_onboard.agents.action_term import (
    ActionTermBase,
    JointPositionAction,
    action_term_to_policy_action,
    build_default_uncontrolled_joint_action,
    get_policy_action_dim,
    pack_policy_action_to_target_joint_state,
    parse_action_cfgs,
    summarize_action_terms,
)
from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.target_joint_state import TargetJointState
from instinct_onboard.utils import CircularBuffer


class AgentStatus(IntEnum):
    Refused = -1  # The agent refused to run (e.g. due to invalid initial state).
    Working = 0  # The agent is currently working but no specific status to report.
    Reached = 1  # The agent has reached the designated state and is holding.
    Ended = 2  # The agent has completed its task and dropping control.
    Failed = 3  # The agent has failed to complete its task and dropping control.


class OnboardAgent(ABC):
    """Base class for onboard agents.

    This class is intended to be inherited by specific agents that handle
    interaction with the robot and the ONNX policy.

    Subclasses are expected to call :meth:`_parse_obs_config` and
    :meth:`_parse_action_config` from their own ``__init__`` after any
    prerequisite state has been set up. The attributes populated by those
    two parse phases are declared below as class-level annotations only:
    they do not exist on an instance until the corresponding parse method
    runs, so accessing them before initialization completes raises
    ``AttributeError`` instead of silently returning an empty default. The
    annotations are grouped by which parse phase owns them.
    """

    # ---- Set by _parse_action_config (via _load_robot_init_state_and_actuator_pd) ----
    # Default joint pose / velocity in sim joint order, sourced from
    # ``cfg["scene"]["robot"]["init_state"]``. Used as the offset for
    # JointPositionAction terms and by ``_get_joint_{pos,vel}_rel_obs``.
    default_joint_pos: np.ndarray
    default_joint_vel: np.ndarray
    # Per-joint PD gains in sim joint order, sourced from
    # ``cfg["scene"]["robot"]["actuators"]``. Exposed via the ``p_gains``
    # / ``d_gains`` properties and forwarded to every action term.
    _p_gains: np.ndarray
    _d_gains: np.ndarray
    # Ordered list of explicit action terms parsed from ``cfg["actions"]``.
    _action_terms: list[ActionTermBase]
    # Synthetic fallback term (built by build_default_uncontrolled_joint_action)
    # used to overlay hold-at-defaults commands onto joints not covered by any
    # explicit action term. None when either (a) every joint is already
    # controlled by an explicit term, or (b)
    # ``use_default_uncontrolled_joint_action == False`` and an external
    # process owns those joints.
    _default_uncontrolled_joint_action: ActionTermBase | None
    # Width of the raw policy action vector; sum of every term's action_dim.
    _policy_action_dim: int

    # ---- Set by _parse_obs_config ----
    # Per-term observation getter, in policy-config order. Concatenating
    # ``obs_funcs[name]()`` across keys reproduces the policy input vector.
    obs_funcs: "OrderedDict[str, Callable]"
    # Optional per-term symmetric clip / scalar scale applied in
    # ``_get_single_obs_term`` before history stacking.
    obs_clip: dict[str, float]
    obs_scales: dict[str, float]
    # Per-term ring buffer for terms whose config requests history stacking.
    obs_history_buffers: dict[str, CircularBuffer]

    # ---- Lazily built by _build_obs_shapes (guarded by hasattr) ----
    # Per-term shape of the flattened observation, used by ``_get_obs_slice``
    # to map an obs name to its slice in the concatenated vector.
    obs_shapes: "OrderedDict[str, tuple]"

    def __init__(
        self,
        logdir: str,
        ros_node: RealNode,
        use_default_uncontrolled_joint_action: bool = True,
    ):
        """Initialize the agent with the log directory and ROS node.

        Args:
            logdir: Directory where exported policy assets and env config are stored.
            ros_node: ROS node used to read state and send commands to the robot.
            use_default_uncontrolled_joint_action: If True (default), the
                agent owns every joint via a hold-at-defaults fill-in term;
                if False, joints outside the explicit action terms are left
                uncommanded for an external owner. See
                :func:`build_default_uncontrolled_joint_action` for the full
                contract.
        """
        self.logdir = logdir
        self.ros_node: RealNode = ros_node
        assert isinstance(self.ros_node, RealNode), "ros_node must be an instance of RealNode"
        env_yaml = os.path.join(self.logdir, "params", "env.yaml")
        with open(env_yaml) as f:
            self.cfg = yaml.unsafe_load(f)

        # Read by _parse_action_config; must exist before subclasses call the parse phase.
        self._use_default_uncontrolled_joint_action: bool = bool(use_default_uncontrolled_joint_action)

    def _load_robot_init_state_and_actuator_pd(self):
        """Load default joint state and actuator PD gains from env config."""

        # default joint positions
        self.default_joint_pos = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for joint_name_expr, joint_pos in self.cfg["scene"]["robot"]["init_state"]["joint_pos"].items():
            # Default joint pose from scene config (matches articulation.default_joint_pos in sim).
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if re.search(joint_name_expr, name):
                    self.default_joint_pos[i] = joint_pos

        # default joint velocities
        self.default_joint_vel = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for joint_name_expr, joint_vel in self.cfg["scene"]["robot"]["init_state"]["joint_vel"].items():
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                if re.search(joint_name_expr, name):
                    self.default_joint_vel[i] = joint_vel

        # stiffness and damping gains
        self._p_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32)
        for actuator_name, actuator_config in self.cfg["scene"]["robot"]["actuators"].items():
            print(f"Get {actuator_config['class_type']} in actuator {actuator_name}")
            for i in range(self.ros_node.NUM_JOINTS):
                name = self.ros_node.sim_joint_names[i]
                for joint_name_expr in actuator_config["joint_names_expr"]:
                    if re.search(joint_name_expr, name):
                        if isinstance(actuator_config["stiffness"], dict):
                            for key, value in actuator_config["stiffness"].items():
                                if re.search(key, name):
                                    self._p_gains[i] = value
                        else:
                            self._p_gains[i] = actuator_config["stiffness"]
                        if isinstance(actuator_config["damping"], dict):
                            for key, value in actuator_config["damping"].items():
                                if re.search(key, name):
                                    self._d_gains[i] = value
                        else:
                            self._d_gains[i] = actuator_config["damping"]

    def _parse_action_config(self):
        """Parse control-related configurations from the environment YAML file."""
        self._load_robot_init_state_and_actuator_pd()

        if not hasattr(self, "default_joint_pos"):
            raise RuntimeError(
                "default_joint_pos not set; did _load_robot_init_state_and_actuator_pd() run? "
                "If you override _load_robot_init_state_and_actuator_pd(), make sure to call "
                "super()._load_robot_init_state_and_actuator_pd() first."
            )

        # Manager-based action terms (Isaac Lab style): maps policy outputs to joint targets / gains.
        self._action_terms = parse_action_cfgs(
            action_cfgs=self.cfg["actions"],
            ros_node=self.ros_node,
            default_joint_pos=self.default_joint_pos,
            p_gains=self._p_gains,
            d_gains=self._d_gains,
            default_joint_vel=self.default_joint_vel,
        )
        self._policy_action_dim = get_policy_action_dim(self._action_terms)
        if self._use_default_uncontrolled_joint_action:
            self._default_uncontrolled_joint_action = build_default_uncontrolled_joint_action(
                ros_node=self.ros_node,
                default_joint_pos=self.default_joint_pos,
                p_gains=self._p_gains,
                d_gains=self._d_gains,
                action_terms=self._action_terms,
            )
        else:
            # External process owns joints outside this agent's action terms;
            # leave them uncommanded in the aggregate TargetJointState.
            self._default_uncontrolled_joint_action = None

        self._summarize_action_terms()

    def _summarize_action_terms(self):
        table_str = summarize_action_terms(self._action_terms, self.ros_node.sim_joint_names)
        self.ros_node.get_logger().info(
            f"Loaded {len(self._action_terms)} action terms with policy action dim {self._policy_action_dim}."
        )
        if self._use_default_uncontrolled_joint_action:
            self.ros_node.get_logger().info(
                "Default action for uncontrolled joints is ENABLED: joints not covered by "
                "explicit action terms will be held at default_joint_pos with configured PD gains."
            )
        else:
            self.ros_node.get_logger().info(
                "Default action for uncontrolled joints is DISABLED: joints not covered by "
                "explicit action terms will be left uncommanded (external controller expected)."
            )
        self.ros_node.get_logger().info(table_str)

    def _parse_obs_config(self):
        """Parse, set attributes from config dict, initialize buffers to speed up the computation"""
        self.obs_funcs: OrderedDict[str, Callable] = OrderedDict()
        self.obs_clip: dict[str, float] = dict()
        self.obs_scales: dict[str, float] = dict()
        self.obs_history_buffers: dict[str, CircularBuffer] = dict()
        for obs_name, obs_config in self.cfg["observations"]["policy"].items():
            if (
                obs_name == "concatenate_terms"
                or obs_name == "concatenate_dim"
                or obs_name == "enable_corruption"
                or obs_name == "history_length"
                or obs_name == "flatten_history_dim"
                or obs_config is None
            ):
                continue
            obs_func: str = obs_config["func"].split(":")[-1]  # get the function name from the config
            # self.obs_funcs will be update in these functions in the order of the config
            if "generated_commands" in obs_func:
                self._parse_generated_commands(obs_name, obs_config)
            else:
                self._parse_observation_function(obs_name, obs_config)
            if obs_config.get("clip", None) is not None:
                self.obs_clip[obs_name] = obs_config["clip"]
            if obs_config.get("scale", None) is not None:
                self.obs_scales[obs_name] = obs_config["scale"]
            if (
                obs_config.get("history_length", 0) != 0
                or self.cfg["observations"]["policy"].get("history_length", None) is not None
            ):
                # if obs_config.get("history_length", None) is not None, use it
                # otherwise, use the global history length
                self.obs_history_buffers[obs_name] = CircularBuffer(
                    obs_config.get("history_length", self.cfg["observations"]["policy"]["history_length"]),
                )

    def _parse_generated_commands(self, obs_name: str, obs_config: dict):
        """Parse the generated commands observation configuration.
        e.g. obs_name: "joint_command", obs_config: joint_command (class -> dict)
             obs_config["func"]:"generated_commands", obs_config["params"]["command_name"]: joint_pos_command
        """
        command_name = obs_config["params"]["command_name"]  # e.g. joint_pos_command
        if hasattr(self, f"_get_{command_name}_cmd_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_cmd_obs")
        elif hasattr(self, f"_get_{command_name}_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{command_name}_obs")
            print(
                "Warning: '_get_{command_name}_obs' for command {command_name} shall be deprecated. Please implementing your command-related observation function '_get_{command_name}_cmd_obs' instead."
            )
        else:
            raise ValueError(
                f"Generated command observation function '_get_{command_name}_obs' not found in the agent. "
                "Please check the configuration."
            )

    def _parse_observation_function(self, obs_name: str, obs_config: dict):
        obs_func = obs_config["func"].split(":")[-1]  # get the function name from the config
        """Parse the observation function from the config."""
        if hasattr(self, f"_get_{obs_func}_obs"):
            self.obs_funcs[obs_name] = getattr(self, f"_get_{obs_func}_obs")
        elif hasattr(self.ros_node, f"_get_{obs_func}_obs"):
            self.obs_funcs[obs_name] = getattr(self.ros_node, f"_get_{obs_func}_obs")
        else:
            raise ValueError(
                f"Observation function '_get_{obs_func}_obs' not found in the agent or ros_node. Please check the"
                " configuration."
            )

    def _get_single_obs_term(
        self,
        obs_name: str,
    ) -> np.ndarray:
        """Get a single observation term by its name. It only perform the post-processing operations
        when specified and available.
        """
        obs_value = self.obs_funcs[obs_name]()
        if obs_name in self.obs_clip:
            obs_value = np.clip(obs_value, -self.obs_clip[obs_name], self.obs_clip[obs_name])
        if obs_name in self.obs_scales:
            obs_value *= self.obs_scales[obs_name]
        if obs_name in self.obs_history_buffers:
            # NOTE: this function automatically handles the history buffer
            self.obs_history_buffers[obs_name].append(obs_value)
            obs_value = self.obs_history_buffers[obs_name].buffer
        return obs_value

    def _get_observation(self) -> np.ndarray:
        """Get all observations in the order of the config for the policy.
        Returns:
            np.ndarray: A single vector containing all observations with shape (dim,).
        """
        obs = []
        for obs_name in self.obs_funcs.keys():
            obs_value = self._get_single_obs_term(obs_name)
            obs.append(obs_value.flatten())  # Ensure obs is a 1D vector
        obs = np.concatenate(obs, axis=-1)  # Concatenate all observations into a single vector
        return obs

    def _build_obs_shapes(self) -> None:
        """Build the obs_shapes if not exists. Please make sure this functions is called before self.reset() procedures."""
        if not hasattr(self, "obs_shapes"):
            self.obs_shapes: OrderedDict[str, tuple] = OrderedDict()
            for obs_name in self.obs_funcs.keys():
                self.obs_shapes[obs_name] = self._get_single_obs_term(obs_name).shape  # (dim,)

    def _get_obs_slice(self, obs_name: str) -> slice:
        """Get the slice of the observation term by its name in the concatenated observation vector."""
        assert hasattr(self, "obs_shapes"), "obs_shapes must be built before calling this function"
        obs_term_names = list(self.obs_funcs.keys())
        target_obs_term_idx = obs_term_names.index(obs_name)
        start_idx = sum(np.prod(self.obs_shapes[obs_name]) for obs_name in obs_term_names[:target_obs_term_idx])
        end_idx = start_idx + np.prod(self.obs_shapes[obs_name])
        return slice(start_idx, end_idx)

    @abstractmethod
    def step(self) -> Tuple[TargetJointState, AgentStatus]:
        """Run one policy step; return the target joint state and the agent status.

        The subclass implementation should call :meth:`pack_policy_action_to_target_joint_state`
        to aggregate a full TargetJointState from the raw policy action vector and the agent's
        action terms.

        The returned :class:`AgentStatus` indicates the current status of the agent:
        ``Working`` for continuous operation, ``Reached`` when a target pose is attained,
        ``Ended`` when the task is complete, ``Failed`` on unrecoverable error.
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset agent-specific state and observation history buffers."""
        self._build_obs_shapes()
        for obs_history_buffer in self.obs_history_buffers.values():
            obs_history_buffer.reset()

    @property
    def action_terms(self) -> list[ActionTermBase]:
        return self._action_terms

    @property
    def default_uncontrolled_joint_action(self) -> ActionTermBase | None:
        return self._default_uncontrolled_joint_action

    @property
    def use_default_uncontrolled_joint_action(self) -> bool:
        """Whether this agent fills in uncontrolled joints with a default action term."""
        return self._use_default_uncontrolled_joint_action

    @property
    def policy_action_dim(self) -> int:
        return self._policy_action_dim

    @property
    def p_gains(self) -> np.ndarray:
        """Proportional gains for the default PD targets (sim joint order)."""
        return self._p_gains

    @property
    def d_gains(self) -> np.ndarray:
        """Derivative gains for the default PD targets (sim joint order)."""
        return self._d_gains

    def pack_policy_action_to_target_joint_state(self, policy_action: np.ndarray) -> TargetJointState:
        """Update each action term with its slice and aggregate into a full TargetJointState.

        When ``use_default_uncontrolled_joint_action`` was True at construction
        (IsaacLab-style), the aggregate also includes a hold-at-defaults
        fill-in for joints untouched by any explicit term. When False, those
        joints stay at zeros across all fields so their ``enable_mask``
        entries remain False and an external controller can own them.
        """
        return pack_policy_action_to_target_joint_state(  # module-level function of same name
            policy_action=policy_action,
            action_terms=self._action_terms,
            default_uncontrolled_joint_action=self._default_uncontrolled_joint_action,
        )

    def _get_last_action_obs(self) -> np.ndarray:
        """On-demand ``last_action`` observation in policy-action space.

        Inverts each ``ActionTermBase``'s affine mapping on the ROS node's
        cached ``last_sent_target_joint_state`` so the returned vector
        matches the raw policy output that produced the last successful
        command.
        """
        return action_term_to_policy_action(
            target_joint_state=self.ros_node.last_sent_target_joint_state,
            action_terms=self._action_terms,
        )

    """
    Some observation functions that depends on the agent's cfg
    """

    def _get_joint_pos_rel_obs(self) -> np.ndarray:
        """Get the joint position relative to the default_joint_pos."""
        return self.ros_node.joint_pos_ - self.default_joint_pos  # shape (NUM_JOINTS,)

    def _get_joint_vel_rel_obs(self) -> np.ndarray:
        """Get the joint velocity relative to the default_joint_pos."""
        return self.ros_node.joint_vel_ - self.default_joint_vel  # shape (NUM_JOINTS,)


class ColdStartAgent(OnboardAgent):
    def __init__(
        self,
        startup_step_size: float,
        ros_node: RealNode,
        joint_target_pos: np.ndarray,
        action_terms: list[ActionTermBase] = None,
        default_uncontrolled_joint_action: ActionTermBase | None = None,
        p_gains: np.ndarray = None,
        d_gains: np.ndarray = None,
    ):
        """Ramp joint positions toward a target using the same action-term layout as the main agent.

        Args:
            startup_step_size: Max change in joint position per step (rad).
            ros_node: ROS node for reads and commands.
            joint_target_pos: Final pose in sim joint order. Required — no default
                because defaulting to zeros is unsafe for humanoid robots.
            action_terms: Action terms from the runtime agent. When provided the
                cold-start path matches the same ``last_action`` layout.
            default_uncontrolled_joint_action: Paired fill-in term for joints
                not covered by ``action_terms``; if omitted a new one is built
                via :func:`build_default_uncontrolled_joint_action`.
            p_gains: Optional proportional gains for commands from this agent.
            d_gains: Optional derivative gains for commands from this agent.
        """
        self.ros_node = ros_node
        self.startup_step_size = startup_step_size
        self.joint_target_pos = np.asarray(joint_target_pos, dtype=np.float32).reshape(-1)
        if self.joint_target_pos.shape != (self.ros_node.NUM_JOINTS,):
            raise ValueError(
                f"joint_target_pos must have shape ({self.ros_node.NUM_JOINTS},), got {self.joint_target_pos.shape}"
            )

        self._p_gains = np.ones(self.ros_node.NUM_JOINTS, dtype=np.float32) * 10.0 if p_gains is None else p_gains
        self._d_gains = np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32) if d_gains is None else d_gains
        if self._p_gains.shape != (self.ros_node.NUM_JOINTS,):
            raise ValueError(f"p_gains must have shape ({self.ros_node.NUM_JOINTS},), got {self._p_gains.shape}")
        if self._d_gains.shape != (self.ros_node.NUM_JOINTS,):
            raise ValueError(f"d_gains must have shape ({self.ros_node.NUM_JOINTS},), got {self._d_gains.shape}")

        if action_terms is not None:
            self._action_terms = action_terms
        else:
            # Fallback: one identity full-joint position term covering every joint.
            self._action_terms = [
                JointPositionAction(
                    name="cold_start_position",
                    action_cfg={
                        "joint_names": [".*"],
                        "scale": 1.0,
                        "offset": 0.0,
                        "use_default_offset": False,
                    },
                    ros_node=self.ros_node,
                    default_joint_pos=self.joint_target_pos,
                    p_gains=self._p_gains,
                    d_gains=self._d_gains,
                    action_cursor=0,
                )
            ]
        self._policy_action_dim = get_policy_action_dim(self._action_terms)

        if default_uncontrolled_joint_action is not None:
            self._default_uncontrolled_joint_action = default_uncontrolled_joint_action
        else:
            self._default_uncontrolled_joint_action = build_default_uncontrolled_joint_action(
                ros_node=self.ros_node,
                default_joint_pos=self.joint_target_pos,
                p_gains=self._p_gains,
                d_gains=self._d_gains,
                action_terms=self._action_terms,
            )

    def step(self) -> Tuple[TargetJointState, AgentStatus]:
        """Step toward ``joint_target_pos`` and return the command bundle for this timestep."""
        dof_pos_err = self.joint_target_pos - self.ros_node._get_joint_pos_obs()
        err_large_mask = np.abs(dof_pos_err) > self.startup_step_size
        reached = not err_large_mask.any()
        if not reached:
            max_err_idx = np.argmax(np.abs(dof_pos_err))
            print(
                f"Current ColdStartAgent gets max error {np.round(np.max(np.abs(dof_pos_err)), decimals=3):.3f} "
                f"at sim joint {max_err_idx:2d}, should be {self.joint_target_pos[max_err_idx]:.3f} but currently is "
                f"{self.ros_node._get_joint_pos_obs()[max_err_idx]:.3f}",
                end="\r",
            )
        dof_pos_target = np.where(
            err_large_mask,
            self.ros_node._get_joint_pos_obs() + np.sign(dof_pos_err) * self.startup_step_size,
            self.joint_target_pos,
        )

        policy_action = self._build_policy_action_for_target_position(dof_pos_target.astype(np.float32))
        target_joint_state = self.pack_policy_action_to_target_joint_state(policy_action)
        return target_joint_state, AgentStatus.Reached if reached else AgentStatus.Working

    def _build_policy_action_for_target_position(self, target_joint_pos: np.ndarray) -> np.ndarray:
        """Back-solve the raw policy vector so the aggregate matches ``target_joint_pos`` for
        position-writing terms while producing zero processed output for velocity and effort terms.

        Delegates to :func:`action_term_to_policy_action` with a synthetic
        :class:`TargetJointState` whose velocity, effort, kp, and kd fields are
        zero — velocity and effort terms back-solve from this to produce zero
        processed output (``raw_action = -offset / scale``), avoiding the bug
        where a non-zero offset would leak through when raw_action is left at
        zero.
        """
        synthetic_tjs = TargetJointState(
            position=target_joint_pos.astype(np.float32),
            velocity=np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32),
            effort=np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32),
            kp=np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32),
            kd=np.zeros(self.ros_node.NUM_JOINTS, dtype=np.float32),
        )
        return action_term_to_policy_action(synthetic_tjs, self._action_terms)

    def reset(self):
        """Reset cold-start state.

        Cold-start is stateless beyond its immutable ``joint_target_pos`` —
        every ``step()`` reads the live joint position from ``ros_node`` and
        computes the next incremental command from scratch. The
        ``last_action`` observation source is
        ``ros_node.last_sent_target_joint_state`` (always the most recent
        command, regardless of which agent sent it), so there is no
        agent-local stale cache to clear.
        """
