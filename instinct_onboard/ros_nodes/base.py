from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import StaticTransformBroadcaster

from instinct_onboard import robot_cfgs
from instinct_onboard.target_joint_state import TargetJointState


@dataclass
class JoyStickData:
    # None for not available
    lx: Optional[float] = None  # + for stick right, - for stick left
    ly: Optional[float] = None  # + for stick up, - for stick down
    rx: Optional[float] = None  # + for stick right, - for stick left
    ry: Optional[float] = None  # + for stick up, - for stick down
    left_trigger: Optional[float] = None  # + for trigger pressed, - for trigger released, but could be ranging (0, 1)
    right_trigger: Optional[float] = None  # + for trigger pressed, - for trigger released, but could be ranging (0, 1)

    # True for pressed, False for released
    up: Optional[bool] = None
    down: Optional[bool] = None
    left: Optional[bool] = None
    right: Optional[bool] = None
    A: Optional[bool] = None
    B: Optional[bool] = None
    X: Optional[bool] = None
    Y: Optional[bool] = None
    start: Optional[bool] = None
    select: Optional[bool] = None
    L1: Optional[bool] = None
    L2: Optional[bool] = None
    R1: Optional[bool] = None
    R2: Optional[bool] = None


class RealNode(Node):
    """This is the basic implementation of handling ROS messages matching the design of IsaacLab.
    It is designed to be used in the script directly to run the ONNX function. But please handle the
    impl of combining observations in the agent implementation.

    This also defines the most common features for each OEM node, and the agents should use this interface to interact with the robot.
    """

    def __init__(
        self,
        node_name: str,
        computer_clip_torque: bool = True,  # if True, the action will be clipped by torque limits
        joint_pos_protect_ratio: float = 1.5,  # if the joint_pos is out of the range of this ratio, the process will shutdown.
        kp_factor: float = 1.0,  # the factor to multiply the p_gain and clip the value to be in [0, 500]
        kd_factor: float = 1.0,  # the factor to multiply the d_gain
        kp_clip: float = 500,  # the maximum limit to the kp factor to prevent rediculious behavior.
        kd_clip: float = 20,  # the maximum limit to the kd factor to prevent rediculious behavior.
        torque_limits_ratio: float = 1.0,  # the factor to multiply the torque limits
        robot_class_name: str = None,  # the robot class name, used to get the robot configuration
        dryrun: bool = True,  # if True, the robot will not send commands to the real robot
    ):
        super().__init__(node_name)
        if robot_class_name is None:
            raise ValueError("robot_class_name must be provided")

        self.NUM_JOINTS = getattr(robot_cfgs, robot_class_name).NUM_JOINTS
        self.NUM_ACTIONS = getattr(robot_cfgs, robot_class_name).NUM_ACTIONS
        self.computer_clip_torque = computer_clip_torque
        self.joint_pos_protect_ratio = joint_pos_protect_ratio
        self.kp_factor = kp_factor
        self.kd_factor = kd_factor
        self.kp_clip = kp_clip
        self.kd_clip = kd_clip
        self.torque_limits_ratio = torque_limits_ratio
        self.robot_class_name = robot_class_name
        self.dryrun = dryrun
        # This is a common joy stick data definition for multi-robot support.
        # Each OEM node should handle how to convert the raw joy stick data to this common definition.
        # Each agent should use this interface to acquire joy stick continuous values and button states.
        self._joy_stick_data = JoyStickData()

        self.parse_config()

    def parse_config(self):
        """Parse, set attributes from config dict, initialize buffers to speed up the computation"""

        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.gravity_vec = np.zeros(3)
        self.gravity_vec[self.up_axis_idx] = -1

        self.torque_limits = (
            np.array(getattr(robot_cfgs, self.robot_class_name).torque_limits) * self.torque_limits_ratio
        )
        self.get_logger().info(f"Torque limits are set by ratio of : {self.torque_limits_ratio}")

        # buffers for observation output (in simulation order)
        self.joint_pos_ = np.zeros(
            self.NUM_JOINTS, dtype=np.float32
        )  # in robot urdf coordinate, but in simulation order. no offset subtracted
        self.joint_vel_ = np.zeros(self.NUM_JOINTS, dtype=np.float32)

        # action buffer
        self.action = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        self.last_sent_target_joint_state: TargetJointState | None = None
        self._send_action_deprecated_warned = False

        # hardware related, in simulation order
        self.joint_signs = getattr(
            robot_cfgs, self.robot_class_name
        ).joint_signs  # in case of joint direction is different between sim and real
        self.sim_joint_names = getattr(robot_cfgs, self.robot_class_name).sim_joint_names
        self.joint_limits_high = np.array(getattr(robot_cfgs, self.robot_class_name).joint_limits_high)
        self.joint_limits_low = np.array(getattr(robot_cfgs, self.robot_class_name).joint_limits_low)
        joint_pos_mid = (self.joint_limits_high + self.joint_limits_low) / 2
        joint_pos_range = (self.joint_limits_high - self.joint_limits_low) / 2
        self.joint_pos_protect_high = joint_pos_mid + joint_pos_range * self.joint_pos_protect_ratio
        self.joint_pos_protect_low = joint_pos_mid - joint_pos_range * self.joint_pos_protect_ratio

    def start_ros_handlers(self):
        """Base method for initializing common ROS publishers.
        Derived classes should override this method to add their specific publishers/subscribers.
        """
        # Common publishers
        self.debug_msg_publisher = self.create_publisher(String, "/debug_msg", 10)
        self.action_publisher = self.create_publisher(Float32MultiArray, "/raw_actions", 10)

    @property
    def joy_stick_data(self) -> JoyStickData:
        return self._joy_stick_data

    def publish_auxiliary_static_transforms(self, transform_field_name: str):
        """Publish some additional static transforms that are not part of the robot model.
        Args:
            transform_field_name: The field name in the robot_cfg of the given robot class. The transform data should
                be a dictionary with the following keys:
                    - translation: (x, y, z)
                    - rotation: (w, x, y, z)
                    - parent_frame: the frame id of the parent frame
                    - child_frame: the frame id of the child frame
        """
        if not hasattr(self, "static_tf_broadcaster"):
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        robot_transform_data = getattr(getattr(robot_cfgs, self.robot_class_name), transform_field_name)
        t.header.frame_id = robot_transform_data["parent_frame"]
        t.child_frame_id = robot_transform_data["child_frame"]
        t.transform.translation.x = robot_transform_data["translation"][0]
        t.transform.translation.y = robot_transform_data["translation"][1]
        t.transform.translation.z = robot_transform_data["translation"][2]
        t.transform.rotation.w = robot_transform_data["rotation"][0]
        t.transform.rotation.x = robot_transform_data["rotation"][1]
        t.transform.rotation.y = robot_transform_data["rotation"][2]
        t.transform.rotation.z = robot_transform_data["rotation"][3]
        self.static_tf_broadcaster.sendTransform(t)

    """
    Get observation term from the corresponding buffers
    NOTE: everything will be NON-batchwise. There is NO batch dimension in the observation.
    """

    def _get_joint_pos_obs(self):
        return self.joint_pos_  # shape (NUM_JOINTS,)

    def _get_joint_vel_obs(self):
        return self.joint_vel_  # shape (NUM_JOINTS,)

    def _get_joint_vel_rel_obs(self):
        """Get the joint velocity relative to the default joint velocity
        TODO: Get the default joint velocity from the configuration and update it in parse_config
        """
        return self.joint_vel_ - np.zeros(self.NUM_JOINTS, dtype=np.float32)  # shape (NUM_JOINTS,)

    def _get_last_action_obs(self):
        return self.action  # shape (NUM_ACTIONS,)

    """
    Functions that actually publish the commands and take effect
    """

    def clip_by_torque_limit(
        self,
        target_joint_pos,
        p_gains: np.ndarray = 0.0,
        d_gains: np.ndarray = 0.0,
    ):
        """Different from simulation, we reverse the process and clip the target position directly,
        so that the PD controller runs in robot but not our script.
        """
        p_limits_low = (-self.torque_limits) + d_gains * self.joint_vel_
        p_limits_high = (self.torque_limits) + d_gains * self.joint_vel_
        action_low = (p_limits_low / p_gains) + self.joint_pos_
        action_high = (p_limits_high / p_gains) + self.joint_pos_

        return np.clip(target_joint_pos, action_low, action_high)

    def _expand_to_full_joint_array(self, values: np.ndarray | float, field_name: str) -> np.ndarray:
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if values_arr.shape == (1,):
            return np.full(self.NUM_JOINTS, values_arr[0].item(), dtype=np.float32)
        if values_arr.shape == (self.NUM_JOINTS,):
            return values_arr.copy()
        raise ValueError(f"{field_name} must be scalar or shape ({self.NUM_JOINTS},), got {values_arr.shape}")

    def send_target_joint_state(self, target_joint_state: TargetJointState) -> bool:
        """Publish a full-size TargetJointState to the hardware.

        The incoming ``target_joint_state`` is expected to be full-size (``num_joints ==
        NUM_JOINTS``); agents build it by aggregating per-joint action terms and a
        default-PD term, so no defaults need to be merged here. On success,
        ``last_sent_target_joint_state`` is updated so observations can replay the
        last command.
        """
        if len(target_joint_state) != self.NUM_JOINTS:
            self.get_logger().error(
                f"TargetJointState num_joints mismatch: expected {self.NUM_JOINTS}, got {target_joint_state.num_joints}"
            )
            return False

        if target_joint_state.isnan_any:
            self.get_logger().error("Target joint state contains NaN, skip sending command.")
            return False

        target_joint_state = target_joint_state.as_dtype(np.float32)

        p_gains = np.clip(target_joint_state.kp * self.kp_factor, 0.0, self.kp_clip)
        d_gains = np.clip(target_joint_state.kd * self.kd_factor, 0.0, self.kd_clip)
        target_joint_pos_send = target_joint_state.position.copy()
        if self.computer_clip_torque:
            target_joint_pos_send = self.clip_by_torque_limit(
                target_joint_pos_send,
                p_gains=p_gains,
                d_gains=d_gains,
            )
            if (target_joint_pos_send != target_joint_state.position).any():
                self.get_logger().info("Action clipped by torque limits.")

        send_ok = self._publish_motor_cmd(
            target_joint_pos=target_joint_pos_send,
            target_joint_vel=target_joint_state.velocity,
            target_joint_effort=target_joint_state.effort,
            p_gains=p_gains,
            d_gains=d_gains,
        )
        if not send_ok:
            return False

        self.last_sent_target_joint_state = TargetJointState(
            position=target_joint_state.position.copy(),
            velocity=target_joint_state.velocity.copy(),
            effort=target_joint_state.effort.copy(),
            kp=target_joint_state.kp.copy(),
            kd=target_joint_state.kd.copy(),
        )
        return True

    # ------------------------------------------------------------------
    # Deprecated legacy path — schedule for removal.
    #
    # Timeline:
    #   2026-06  All scripts migrated to send_target_joint_state directly.
    #   2026-09  (planned) Remove send_action, self.NUM_ACTIONS, and
    #            self.action from RealNode. Any remaining callers must
    #            convert to send_target_joint_state + TargetJointState.
    # ------------------------------------------------------------------
    def send_action(
        self,
        action: np.array,
        action_offset: np.array = 0.0,
        action_scale: np.ndarray = 1.0,
        p_gains: np.ndarray = 0.0,
        d_gains: np.ndarray = 0.0,
    ) -> bool:
        """Legacy full-joint position command path (compatibility only).

        This API assumes policy action layout is exactly one position scalar per
        joint (``NUM_ACTIONS == NUM_JOINTS``) and applies ``target_pos = action *
        action_scale + action_offset`` before delegating to
        ``send_target_joint_state``. New code should use ``send_target_joint_state``
        directly.
        """
        if self.NUM_ACTIONS != self.NUM_JOINTS:
            self.get_logger().error(
                "send_action only supports legacy full-joint position layout: "
                f"NUM_ACTIONS ({self.NUM_ACTIONS}) must equal NUM_JOINTS ({self.NUM_JOINTS}). "
                "Use send_target_joint_state for manager-based action terms."
            )
            return False

        if not self._send_action_deprecated_warned:
            self.get_logger().warn(
                "send_action is deprecated compatibility API. Prefer send_target_joint_state.",
                throttle_duration_sec=5.0,
            )
            self._send_action_deprecated_warned = True

        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape != (self.NUM_ACTIONS,):
            self.get_logger().error(
                f"Action shape mismatch in send_action: expected ({self.NUM_ACTIONS},), got {action_arr.shape}"
            )
            return False
        # NOTE: compatibility path keeps raw action history for legacy observations.
        self.action[:] = action_arr
        self.action_publisher.publish(Float32MultiArray(data=action_arr.tolist()))

        action_scale_arr = self._expand_to_full_joint_array(action_scale, "action_scale")
        action_offset_arr = self._expand_to_full_joint_array(action_offset, "action_offset")
        target_joint_pos = action_arr * action_scale_arr + action_offset_arr
        target_joint_vel = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        target_joint_effort = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        p_gains_arr = self._expand_to_full_joint_array(p_gains, "p_gains")
        d_gains_arr = self._expand_to_full_joint_array(d_gains, "d_gains")

        return self.send_target_joint_state(
            TargetJointState(
                position=target_joint_pos,
                velocity=target_joint_vel,
                effort=target_joint_effort,
                kp=p_gains_arr,
                kd=d_gains_arr,
            )
        )

    @abstractmethod
    def _publish_motor_cmd(
        self,
        target_joint_pos: np.array,
        target_joint_vel: np.array,
        target_joint_effort: np.array,
        p_gains: np.ndarray,
        d_gains: np.ndarray,
    ) -> bool:
        """Publish the joint commands to the robot motors in robot coordinates system.

        All arrays are in simulation order with shape (NUM_JOINTS,).
        Returns True on success, False on failure (e.g. NaN values).
        """
        pass

    @abstractmethod
    def _turn_off_motors(self):
        """Turn off the motors"""
        pass
