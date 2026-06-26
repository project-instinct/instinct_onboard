import queue
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.parkour_agent import (
    ParkourAgent,
    ParkourStandAgent,
)
from instinct_onboard.joystick import UnitreeJoyStick
from instinct_onboard.ros_nodes.realsense import RealsenseMPCamera
from instinct_onboard.ros_nodes.unitree import UnitreeNode

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500

"""
G1 Parkour Node

A ROS2 node for controlling Unitree G1 robot using parkour agent with depth camera perception.
This script enables autonomous parkour behaviors including standing, walking, and obstacle
navigation using depth perception from RealSense camera.

Features:
    - Depth perception using RealSense D435 camera
    - ParkourAgent for autonomous parkour behaviors
    - ParkourStandAgent for standing and balancing
    - Velocity control via joystick/wireless controller
    - Real-time obstacle avoidance and navigation

Command-Line Arguments:
    Required:
        --logdir PATH          Directory containing the trained parkour agent model
                              (must contain exported/actor.onnx and depth image encoder)
        --standdir PATH        Directory containing the stand agent model
                              (must contain exported/actor.onnx)

    Optional:
        --startup_step_size FLOAT
                              Startup step size for cold start agent (default: 0.2)
        --nodryrun            Disable dry run mode (default: False, runs in dry run mode)
        --kpkd_factor FLOAT   KP/KD gain multiplier for cold start agent (default: 2.0)
        --depth_vis            Enable depth image visualization (publishes to /realsense/depth_image)
        --pointcloud_vis       Enable pointcloud visualization (publishes to /realsense/pointcloud)
        --lin_vel_deadband FLOAT
                              Deadband for linear velocity control (default: 0.5)
        --lin_vel_range LIST   Range of linear velocity [min, max] (default: [0.5, 0.5])
        --ang_vel_deadband FLOAT
                              Deadband for angular velocity control (default: 0.5)
        --ang_vel_range LIST   Range of angular velocity [min, max] (default: [0.0, 1.0])
        --debug                Enable debug mode with debugpy (listens on 0.0.0.0:6789)

Agent Workflow:
    1. Cold Start Agent (initial state)
       - Automatically starts when node launches
       - Transitions robot to initial pose
       - Press 'R1' to switch to stand agent (if available)
       - Press any direction button to switch to parkour agent

    2. Stand Agent (requires --standdir)
       - Activated by pressing 'R1' after cold start completes
       - Provides standing and balancing behavior
       - Press 'L1' to switch to parkour agent

    3. Parkour Agent
       - Executes autonomous parkour behaviors with depth perception
       - Responds to velocity commands from joystick/wireless controller
       - Press 'R1' to switch back to stand agent

Joystick Controls:
    R1 Button:   Switch to stand agent (from cold start or parkour)
    L1 Button:   Switch to parkour agent (from stand)
    Direction Buttons: Switch to parkour agent (from cold start)

Velocity Control (Wireless Controller):
    Linear velocity is controlled based on forward/backward input with deadband
    Angular velocity is controlled based on left/right rotation input with deadband
    Velocity ranges can be adjusted via command-line arguments

Example Usage:
    Basic usage with required arguments:
        python g1_parkour.py \\
            --logdir /path/to/parkour/model \\
            --standdir /path/to/stand/model

    With visualization options:
        python g1_parkour.py \\
            --logdir /path/to/parkour/model \\
            --standdir /path/to/stand/model \\
            --depth_vis --pointcloud_vis

    Real robot control (disable dry run):
        python g1_parkour.py \\
            --logdir /path/to/parkour/model \\
            --standdir /path/to/stand/model \\
            --nodryrun

    With custom velocity parameters:
        python g1_parkour.py \\
            --logdir /path/to/parkour/model \\
            --standdir /path/to/stand/model \\
            --lin_vel_deadband 0.3 \\
            --lin_vel_range 0.3 0.7 \\
            --ang_vel_deadband 0.2 \\
            --ang_vel_range 0.0 1.5

    With custom startup parameters:
        python g1_parkour.py \\
            --logdir /path/to/parkour/model \\
            --standdir /path/to/stand/model \\
            --startup_step_size 0.3 \\
            --kpkd_factor 1.5

Notes:
    - The script runs at 50Hz main loop frequency (20ms period)
    - RealSense camera is configured at 480x270 resolution, 60 FPS
    - Robot configuration: G1_29Dof_TorsoBase (29 degrees of freedom)
    - Joint position protection ratio: 2.0
    - Camera runs in a separate process for better performance
    - Velocity control parameters affect joystick/wireless controller responsiveness
"""


class G1ParkourNode(RealsenseMPCamera, UnitreeNode):
    def __init__(
        self,
        *args,
        lin_vel_deadband: float = 0.5,
        ang_vel_deadband: float = 0.5,
        cmd_px_range: tuple = (0.5, 0.5),
        cmd_nx_range: tuple = (0.0, 0.0),
        cmd_py_range: tuple = (0.0, 0.0),
        cmd_ny_range: tuple = (0.0, 0.0),
        cmd_pyaw_range: tuple = (0.0, 1.0),
        cmd_nyaw_range: tuple = (0.0, 1.0),
        joystick_topic: str = "/wirelesscontroller",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None

        # Velocity-control parameters — the script writer decides how velocity
        # is computed from joystick input.
        self._lin_vel_deadband = lin_vel_deadband
        self._ang_vel_deadband = ang_vel_deadband
        self._cmd_px_range = cmd_px_range
        self._cmd_nx_range = cmd_nx_range
        self._cmd_py_range = cmd_py_range
        self._cmd_ny_range = cmd_ny_range
        self._cmd_pyaw_range = cmd_pyaw_range
        self._cmd_nyaw_range = cmd_nyaw_range

        # Wire up the wireless controller (joystick).  The default
        # safety-shutdown (turn off motors + SystemExit) is sufficient.
        self._joystick = UnitreeJoyStick(self, joy_stick_topic=joystick_topic)

        # Generic velocity command buffer — populated each main-loop tick
        # from the joystick.  Agents read it via _get_base_velocity_cmd_obs.
        self.base_velocity_cmd = np.zeros(3, dtype=np.float32)

    def register_agent(self, name: str, agent):
        self.available_agents[name] = agent

    def check_buffers_ready(self):
        """Also wait for the first wireless-controller message before the main loop."""
        if not super().check_buffers_ready():
            return False
        return self._joystick.data.ly is not None

    def start_ros_handlers(self):
        super().start_ros_handlers()
        # build the joint state publisher and base_link tf publisher
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        # start the main loop with 20ms duration
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_timer_counter: int = 0  # counter for the main loop timer to assess the actual frequency
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)

    def main_loop_callback(self):
        main_loop_callback_start_time = time.time()

        # Compute base velocity command from joystick (or any other source the
        # script writer chooses) and write it to the generic ros_node buffer so
        # the agent can read it via _get_base_velocity_cmd_obs.
        self.base_velocity_cmd = self._compute_velocity_from_joystick()

        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        elif self.current_agent_name == "cold_start":
            tjs, status = self.available_agents[self.current_agent_name].step()
            if status != AgentStatus.Working:
                if "stand" in self.available_agents.keys():
                    self.get_logger().info(
                        "ColdStartAgent done, press 'R1' to switch to stand agent.", throttle_duration_sec=10.0
                    )
                else:
                    self.get_logger().info(
                        "ColdStartAgent done, press any direction button to switch to parkour agent.",
                        throttle_duration_sec=10.0,
                    )
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working and (self._joystick.data.R1):
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "stand":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.refresh_camera_data()
            self.send_target_joint_state(tjs)
            if self._joystick.data.L1:
                self.get_logger().info("L1 button pressed, switching to parkour agent.")
                self.current_agent_name = "parkour"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "parkour":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if self._joystick.data.R1:
                self.get_logger().info("R1 button pressed, switching to stand agent.")
                self.current_agent_name = "stand"
                self.available_agents[self.current_agent_name].reset()

        # count the main loop timer counter and log the actual frequency every 500 counts
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_callback_time_consumptions.put(time.time() - main_loop_callback_start_time)
            self.main_loop_timer_counter += 1
            if self.main_loop_timer_counter % MAIN_LOOP_FREQUENCY_CHECK_INTERVAL == 0:
                time_consumptions = [
                    self.main_loop_callback_time_consumptions.get() for _ in range(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)
                ]
                self.get_logger().info(
                    f"Actual main loop frequency: {(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL / (time.time() - self.main_loop_timer_counter_time)):.2f} Hz. Mean time consumption: {np.mean(time_consumptions):.4f} s."
                )
                self.main_loop_timer_counter = 0
                self.main_loop_timer_counter_time = time.time()

    def _compute_velocity_from_joystick(self) -> np.ndarray:
        """Compute [vx, vy, yaw] from the wireless controller axes.

        This is the script writer's decision — other scripts may use a
        different mapping or source entirely.  The result is written to
        ``self.base_velocity_cmd`` so the agent picks it up.
        """
        jy = self._joystick.data
        # left-y for forward/backward
        ly = jy.ly
        if ly > self._lin_vel_deadband:
            vx = (ly - self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vx = vx * (self._cmd_px_range[1] - self._cmd_px_range[0]) + self._cmd_px_range[0]
        elif ly < -self._lin_vel_deadband:
            vx = (ly + self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vx = vx * (self._cmd_nx_range[1] - self._cmd_nx_range[0]) - self._cmd_nx_range[0]
        else:
            vx = 0.0
        # left-x for side moving left/right
        lx = -jy.lx
        if lx > self._lin_vel_deadband:
            vy = (lx - self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vy = vy * (self._cmd_py_range[1] - self._cmd_py_range[0]) + self._cmd_py_range[0]
        elif lx < -self._lin_vel_deadband:
            vy = (lx + self._lin_vel_deadband) / (1 - self._lin_vel_deadband)
            vy = vy * (self._cmd_ny_range[1] - self._cmd_ny_range[0]) - self._cmd_ny_range[0]
        else:
            vy = 0.0
        # right-x for turning left/right
        rx = -jy.rx
        if rx > self._ang_vel_deadband:
            yaw = (rx - self._ang_vel_deadband) / (1 - self._ang_vel_deadband)
            yaw = yaw * (self._cmd_pyaw_range[1] - self._cmd_pyaw_range[0]) + self._cmd_pyaw_range[0]
        elif rx < -self._ang_vel_deadband:
            yaw = (rx + self._ang_vel_deadband) / (1 - self._ang_vel_deadband)
            yaw = yaw * (self._cmd_nyaw_range[1] - self._cmd_nyaw_range[0]) - self._cmd_nyaw_range[0]
        else:
            yaw = 0.0
        return np.array([vx, vy, yaw], dtype=np.float32)


def main(args):
    rclpy.init()

    node = G1ParkourNode(
        depth_resolution=(480, 270),  # (width, height)
        depth_fps=60,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        dryrun=not args.nodryrun,
        lin_vel_deadband=args.lin_vel_deadband,
        ang_vel_deadband=args.ang_vel_deadband,
        cmd_px_range=args.lin_vel_range,
        cmd_pyaw_range=args.ang_vel_range,
        cmd_nyaw_range=args.ang_vel_range,
    )

    stand_agent = ParkourStandAgent(
        logdir=args.standdir,
        ros_node=node,
    )
    node.register_agent("stand", stand_agent)

    parkour_agent = ParkourAgent(
        logdir=args.logdir,
        ros_node=node,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
    )
    node.register_agent("parkour", parkour_agent)

    cold_start_agent = ColdStartAgent(
        startup_step_size=args.startup_step_size,
        ros_node=node,
        joint_target_pos=parkour_agent.default_joint_pos,
        action_terms=parkour_agent.action_terms,
        p_gains=parkour_agent.p_gains * args.kpkd_factor,
        d_gains=parkour_agent.d_gains * args.kpkd_factor,
    )
    node.register_agent("cold_start", cold_start_agent)

    if args.depth_vis or args.pointcloud_vis:
        node.publish_auxiliary_static_transforms("camera_depth_link_transform")

    node.start_ros_handlers()
    node.get_logger().info("G1ParkourNode is ready to run.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Node shutdown complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G1 Parkour Node")
    parser.add_argument(
        "--standdir",
        type=str,
        help="Directory to load the stand agent from",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the parkour agent from",
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the cold start agent (default: 0.2)",
    )
    parser.add_argument(
        "--kpkd_factor",
        type=float,
        default=2.0,
        help="KPKD factor for the cold start agent (default: 2.0)",
    )
    parser.add_argument(
        "--depth_vis",
        action="store_true",
        default=False,
        help="Visualize the depth image (default: False)",
    )
    parser.add_argument(
        "--pointcloud_vis",
        action="store_true",
        default=False,
        help="Visualize the pointcloud (default: False)",
    )
    parser.add_argument(
        "--lin_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for linear velocity (default: 0.5)",
    )
    parser.add_argument(
        "--lin_vel_range",
        type=list,
        default=[0.5, 0.5],
        help="Range of linear velocity, only forward (default: [0.5 0.5])",
    )
    parser.add_argument(
        "--ang_vel_deadband",
        type=float,
        default=0.5,
        help="Deadband of wireless control for angular velocity (default: 0.5)",
    )
    parser.add_argument(
        "--ang_vel_range",
        type=list,
        default=[0.0, 1.0],
        help="Range of linear velocity, both turn left and turn right (default: [0.0 1.0])",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()
    if args.debug:
        # import typing; typing.TYPE_CHECKING = True
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
