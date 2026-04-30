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

    def send_action(
        self,
        action: np.array,
        action_offset: np.array = 0.0,
        action_scale: np.ndarray = 1.0,
        p_gains: np.ndarray = 0.0,
        d_gains: np.ndarray = 0.0,
    ):
        """Send the action to the robot motors, which does the preprocessing
        just like env.step in simulation.
        However, since this process only controls one robot, the action is not batched.
        NOTE: when switching between agents, the last_action term should be shared between agents.
        Thus, the ros node has to update the action buffer
        """
        if np.isnan(action).any():
            self.get_logger().error("Actions contain NaN, Skip sending the action to the robot.")
            return
        self.action[:] = action
        self.action_publisher.publish(Float32MultiArray(data=action))
        action_scaled = action * action_scale
        target_joint_pos = action_scaled + action_offset
        p_gains = np.clip(p_gains * self.kp_factor, 0.0, self.kp_clip)
        d_gains = np.clip(d_gains * self.kd_factor, 0.0, self.kd_clip)
        if self.computer_clip_torque:
            target_joint_pos = self.clip_by_torque_limit(
                target_joint_pos,
                p_gains=p_gains,
                d_gains=d_gains,
            )
        self._publish_motor_cmd(target_joint_pos, p_gains=p_gains, d_gains=d_gains)

    def clear_action_buffer(self):
        """Clear last-action observations without sending any motor command."""
        self.action[:] = 0.0

    @abstractmethod
    def _publish_motor_cmd(self, target_joint_pos: np.array, p_gains: np.array, d_gains: np.array):
        """Publish the joint commands to the robot motors in robot coordinates system.
        robot_coordinates_action: shape (NUM_JOINTS,), in simulation order.
        """
        pass

    @abstractmethod
    def _turn_off_motors(self):
        """Turn off the motors"""
        pass
