import numpy as np
import rclpy
from crc_module import get_crc
from unitree_go.msg import WirelessController
from unitree_hg.msg import IMUState, LowCmd, LowState  # MotorState,; MotorCmd,

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard import utils
from instinct_onboard.ros_nodes.base import RealNode


class UnitreeNode(RealNode):
    """This is the implementation of the Unitree robot ROS interface."""

    def __init__(
        self,
        low_state_topic: str = "/lowstate",
        low_cmd_topic: str = "/lowcmd",
        imu_state_topic: str = "/secondary_imu",
        joy_stick_topic: str = "/wirelesscontroller",
        **kwargs,
    ):
        super().__init__(node_name="unitree_node", **kwargs)
        self.low_state_topic = low_state_topic
        self.imu_state_topic = imu_state_topic
        # Generate a unique cmd topic so that the low_cmd will not send to the robot's motor.
        self.low_cmd_topic = (
            low_cmd_topic if not self.dryrun else low_cmd_topic + "_dryrun_" + str(np.random.randint(0, 65535))
        )
        self.joy_stick_topic = joy_stick_topic

    def parse_config(self):
        super().parse_config()

        # load robot-specific configurations
        self.joint_map = getattr(robot_cfgs, self.robot_class_name).joint_map
        self.real_joint_names = getattr(robot_cfgs, self.robot_class_name).real_joint_names
        self.joint_signs = getattr(robot_cfgs, self.robot_class_name).joint_signs
        self.turn_on_motor_mode = getattr(robot_cfgs, self.robot_class_name).turn_on_motor_mode
        self.mode_pr = getattr(robot_cfgs, self.robot_class_name).mode_pr

    def start_ros_handlers(self):
        """After initializing the env and policy, register ros related callbacks and topics"""
        super().start_ros_handlers()
        self.low_cmd_publisher = self.create_publisher(LowCmd, self.low_cmd_topic, 10)
        self.low_cmd_buffer = LowCmd()
        self.low_cmd_buffer.mode_pr = self.mode_pr

        # ROS subscribers
        self.low_state_subscriber = self.create_subscription(
            LowState, self.low_state_topic, self._low_state_callback, 10
        )
        self.torso_imu_subscriber = self.create_subscription(
            IMUState, self.imu_state_topic, self._torso_imu_state_callback, 10
        )
        self.joy_stick_subscriber = self.create_subscription(
            WirelessController, self.joy_stick_topic, self._joy_stick_callback, 10
        )
        self.get_logger().info(
            "ROS handlers started, waiting to receive critical low state and wireless controller messages."
        )
        if not self.dryrun:
            self.get_logger().warn(
                f"You are running the code in no-dryrun mode and publishing to '{self.low_cmd_topic}', Please keep"
                " safe."
            )
        else:
            self.get_logger().warn(
                f"You are publishing low cmd to '{self.low_cmd_topic}' because of dryrun mode, Please check and be"
                " safe."
            )
        while rclpy.ok():
            rclpy.spin_once(self)
            if self.check_buffers_ready():
                break
        self.get_logger().info("All necessary buffers received, the robot is ready to go.")

    def check_buffers_ready(self):
        """Check if all the necessary buffers are ready to use. Only used at the the end of the start_ros_handlers."""
        buffer_ready = hasattr(self, "low_state_buffer") and self.joy_stick_data.lx is not None
        if self.imu_state_topic is not None:
            buffer_ready = buffer_ready and hasattr(self, "torso_imu_buffer")
        return buffer_ready

    """
    ROS callbacks and handlers that update the buffer
    """

    def _low_state_callback(self, msg):
        """store and handle proprioception data"""
        self.get_logger().info("Low state data received.", once=True)
        self.low_state_buffer = msg  # keep the latest low state
        self.low_cmd_buffer.mode_machine = msg.mode_machine

        # refresh joint_pos and joint_vel
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.joint_pos_[sim_idx] = self.low_state_buffer.motor_state[real_idx].q * self.joint_signs[sim_idx]
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.joint_vel_[sim_idx] = self.low_state_buffer.motor_state[real_idx].dq * self.joint_signs[sim_idx]
        # automatic safety check
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if (
                self.joint_pos_[sim_idx] > self.joint_pos_protect_high[sim_idx]
                or self.joint_pos_[sim_idx] < self.joint_pos_protect_low[sim_idx]
            ):
                self.get_logger().error(
                    f"Joint {sim_idx}(sim), {real_idx}(real) position out of range at"
                    f" {self.low_state_buffer.motor_state[real_idx].q}"
                )
                self.get_logger().error("The motors and this process shuts down.")
                self._turn_off_motors()
                raise SystemExit()

    def _torso_imu_state_callback(self, msg):
        """store and handle torso imu data"""
        self.get_logger().info("Torso IMU data received.", once=True)
        self.torso_imu_buffer = msg

    def _joy_stick_callback(self, msg):
        self.get_logger().info("Wireless controller data received.", once=True)
        # fill the joy stick data
        self.joy_stick_data.A = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.A)
        self.joy_stick_data.B = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.B)
        self.joy_stick_data.X = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.X)
        self.joy_stick_data.Y = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.Y)
        self.joy_stick_data.start = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.start)
        self.joy_stick_data.select = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.select)
        self.joy_stick_data.L1 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.L1)
        self.joy_stick_data.R1 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.R1)
        self.joy_stick_data.L2 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.L2)
        self.joy_stick_data.R2 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.R2)
        self.joy_stick_data.up = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.up)
        self.joy_stick_data.down = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.down)
        self.joy_stick_data.left = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.left)
        self.joy_stick_data.right = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.right)
        # fill the joy stick data
        self.joy_stick_data.lx = msg.lx
        self.joy_stick_data.ly = msg.ly
        self.joy_stick_data.rx = msg.rx
        self.joy_stick_data.ry = msg.ry
        # left/right trigger is not available in Unitree Wireless Controller

        # refer to Unitree Remote Control data structure, msg.keys is a bit mask
        # 00000000 00000001 means pressing the 0-th button (R1)
        # 00000000 00000010 means pressing the 1-th button (L1)
        # 10000000 00000000 means pressing the 15-th button (left)
        if (msg.keys & robot_cfgs.UnitreeWirelessButtons.R2) or (
            msg.keys & robot_cfgs.UnitreeWirelessButtons.L2
        ):  # R2 or L2 is pressed
            self.get_logger().warn("R2 or L2 is pressed, the motors and this process shuts down.")
            self._turn_off_motors()
            raise SystemExit()

    """
    Refresh observation buffer and corresponding sub-functions
    NOTE: everything will be NON-batchwise. There is NO batch dimension in the observation.
    """

    def _get_quat_w_obs(self):
        """Get the quaternion in wxyz format from the torso IMU or low state buffer."""
        if hasattr(self, "torso_imu_buffer"):
            return np.array(self.torso_imu_buffer.quaternion, dtype=np.float32)
        else:
            return np.array(self.low_state_buffer.imu_state.quaternion, dtype=np.float32)

    def _get_base_ang_vel_obs(self):
        if hasattr(self, "torso_imu_buffer"):
            return np.array(self.torso_imu_buffer.gyroscope, dtype=np.float32)
        else:
            return np.array(self.low_state_buffer.imu_state.gyroscope, dtype=np.float32)

    def _get_projected_gravity_obs(self):
        if hasattr(self, "torso_imu_buffer"):
            quat_wxyz = np.quaternion(
                self.torso_imu_buffer.quaternion[0],
                self.torso_imu_buffer.quaternion[1],
                self.torso_imu_buffer.quaternion[2],
                self.torso_imu_buffer.quaternion[3],
            )
        else:
            quat_wxyz = np.quaternion(
                self.low_state_buffer.imu_state.quaternion[0],
                self.low_state_buffer.imu_state.quaternion[1],
                self.low_state_buffer.imu_state.quaternion[2],
                self.low_state_buffer.imu_state.quaternion[3],
            )
        return utils.quat_rotate_inverse(
            quat_wxyz,
            self.gravity_vec,
        ).astype(np.float32)

    """
    Control related functions
    """

    """
    Functions that actually publish the commands and take effect
    """

    def _publish_motor_cmd(
        self,
        target_joint_pos: np.array,  # shape (NUM_JOINTS,), in simulation order
        target_joint_vel: np.array,  # shape (NUM_JOINTS,), in simulation order
        target_joint_effort: np.array,  # shape (NUM_JOINTS,), in simulation order
        p_gains: np.ndarray,  # In the order of simulation joints, not real joints
        d_gains: np.ndarray,  # In the order of simulation joints, not real joints
    ) -> bool:
        """Publish the joint commands to the robot motors in robot coordinates system.

        All arrays are in simulation order with shape (NUM_JOINTS,).
        Returns True on success, False on failure.
        """
        if np.isnan(target_joint_pos).any() or np.isnan(p_gains).any() or np.isnan(d_gains).any():
            self.get_logger().error("Motor command arrays contain NaN, skip sending command")
            return False

        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            if not self.dryrun:
                self.low_cmd_buffer.motor_cmd[real_idx].mode = self.turn_on_motor_mode[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].q = (target_joint_pos[sim_idx] * self.joint_signs[sim_idx]).item()
            self.low_cmd_buffer.motor_cmd[real_idx].dq = (target_joint_vel[sim_idx] * self.joint_signs[sim_idx]).item()
            self.low_cmd_buffer.motor_cmd[real_idx].tau = (
                target_joint_effort[sim_idx] * self.joint_signs[sim_idx]
            ).item()
            self.low_cmd_buffer.motor_cmd[real_idx].kp = p_gains[sim_idx].item()
            self.low_cmd_buffer.motor_cmd[real_idx].kd = d_gains[sim_idx].item()

        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_publisher.publish(self.low_cmd_buffer)
        return True

    def _turn_off_motors(self):
        """Turn off the motors"""
        for sim_idx in range(self.NUM_JOINTS):
            real_idx = self.joint_map[sim_idx]
            self.low_cmd_buffer.motor_cmd[real_idx].mode = 0x00
            self.low_cmd_buffer.motor_cmd[real_idx].q = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].dq = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].tau = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kp = 0.0
            self.low_cmd_buffer.motor_cmd[real_idx].kd = 0.0
        self.low_cmd_buffer.crc = get_crc(self.low_cmd_buffer)
        self.low_cmd_publisher.publish(self.low_cmd_buffer)
