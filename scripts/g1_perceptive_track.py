import os
import queue
import sys
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.tracking_agent import PerceptiveTrackerAgent, TrackerAgent
from instinct_onboard.agents.walk_agent import WalkAgent
from instinct_onboard.joystick import UnitreeJoyStick
from instinct_onboard.ros_nodes.realsense import UnitreeRsCameraNode

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500

"""
G1 Perceptive Tracking Node

A ROS2 node for controlling Unitree G1 robot using perceptive tracking agent with depth camera
perception. This script integrates RealSense camera for depth-based motion tracking and supports
multiple agent modes for different behaviors.

Features:
    - Depth perception using RealSense D435 camera
    - PerceptiveTrackerAgent with depth image encoding
    - Multiple agent modes: cold start, walk (optional), and tracking
    - Real-time motion tracking with joystick control
    - Visualization options for debugging

Command-Line Arguments:
    Required:
        --logdir PATH          Directory containing the trained perceptive tracking agent model
                              (must contain exported/actor.onnx and exported/0-depth_image.onnx)
        --motion_dir PATH      Directory containing retargeted motion files (.npz format)

    Optional:
        --walk_logdir PATH     Directory containing the walk agent model (enables walk agent mode)
        --startup_step_size FLOAT
                              Startup step size for cold start agent (default: 0.2)
        --nodryrun            Disable dry run mode (default: False, runs in dry run mode)
        --kpkd_factor FLOAT    KP/KD gain multiplier for cold start agent (default: 2.0)
        --motion_vis           Enable motion visualization by publishing joint states and TF
                              (requires robot_state_publisher for visualization)
        --depth_vis            Enable depth image visualization (publishes to /realsense/depth_image)
        --pointcloud_vis       Enable pointcloud visualization (publishes to /realsense/pointcloud)
        --debug                Enable debug mode with debugpy (listens on 0.0.0.0:6789)

Agent Workflow:
    1. Cold Start Agent (initial state)
       - Automatically starts when node launches
       - Transitions robot to initial pose
       - Press 'L1' to switch to walk agent (if available)
       - Press any direction button to switch to tracking agent

    2. Walk Agent (optional, requires --walk_logdir)
       - Activated by pressing 'L1' after cold start completes
       - Provides basic walking behavior
       - Press direction buttons to switch to tracking agent with specific motions:
         * UP:    diveroll4-ziwen-0-retargeted.npz
         * DOWN:  kneelClimbStep1-x-0.1-ziwen-retargeted.npz
         * LEFT:  rollVault11-ziwen-retargeted.npz
         * RIGHT: jumpsit2-ziwen-retargeted.npz
         * X:     superheroLanding-retargeted.npz
       - Press 'L1' from tracking agent to return to walk agent

    3. Tracking Agent (perceptive tracking)
       - Executes motion sequences with depth perception
       - Press 'A' button to match motion to current robot heading
       - Automatically switches to walk agent when motion completes (if available)
       - Otherwise turns off motors and exits

Joystick Controls:
    A Button:     Match tracking motion to current robot heading
    L1 Button:   Switch between walk and tracking agents
    UP:           Load and execute diveroll4 motion sequence
    DOWN:         Load and execute kneelClimbStep1 motion sequence
    LEFT:         Load and execute rollVault11 motion sequence
    RIGHT:        Load and execute jumpsit2 motion sequence
    X Button:     Load and execute superheroLanding motion sequence

Example Usage:
    Basic usage with required arguments:
        python g1_perceptive_track.py --logdir /path/to/tracking/model --motion_dir /path/to/motions

    With walk agent:
        python g1_perceptive_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --walk_logdir /path/to/walk/model

    With visualization options:
        python g1_perceptive_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --depth_vis --pointcloud_vis --motion_vis

    Dry run mode (default, no actual robot control):
        python g1_perceptive_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions

    Real robot control (disable dry run):
        python g1_perceptive_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --nodryrun

    With custom startup parameters:
        python g1_perceptive_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --startup_step_size 0.3 \\
            --kpkd_factor 1.5

Notes:
    - The script runs at 50Hz main loop frequency (20ms period)
    - RealSense camera is configured at 480x270 resolution, 60 FPS
    - Robot configuration: G1_29Dof_TorsoBase (29 degrees of freedom)
    - Joint position protection ratio: 2.0
    - Camera runs in a separate process for better performance
"""


class G1TrackingNode(UnitreeRsCameraNode):
    def __init__(
        self,
        *args,
        motion_vis: bool = False,
        x_vel_scale: float = 0.5,
        y_vel_scale: float = 0.5,
        yaw_vel_scale: float = 1.0,
        joystick_topic: str = "/wirelesscontroller",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None
        self.motion_vis = motion_vis

        # Velocity scales for the walk agent — script writer's choice.
        self._x_vel_scale = x_vel_scale
        self._y_vel_scale = y_vel_scale
        self._yaw_vel_scale = yaw_vel_scale

        # Wire up the wireless controller (joystick) for agent switching and
        # velocity commands.  Default safety-shutdown is sufficient.
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
        # start the visualization timer with 100ms duration
        vis_duration = 0.1
        if self.motion_vis:
            self.vis_timer = self.create_timer(vis_duration, self.vis_callback)

    def main_loop_callback(self):
        main_loop_callback_start_time = time.time()

        # Compute base velocity command from joystick for the walk agent.
        self._update_base_velocity_cmd_from_joystick()

        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.get_logger().info("Press 'A' button to match motion to current heading.", throttle_duration_sec=2.0)
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        if self._joystick.data.A:
            self.get_logger().info("A button pressed, matching motion to current heading.", throttle_duration_sec=2.0)
            self.available_agents["tracking"].match_to_current_heading()

        elif self.current_agent_name == "cold_start":
            tjs, status = self.available_agents[self.current_agent_name].step()
            if status != AgentStatus.Working and ("walk" in self.available_agents.keys()):
                self.get_logger().info(
                    "ColdStartAgent done, press 'L1' to switch to walk agent.", throttle_duration_sec=10.0
                )
            elif status != AgentStatus.Working:
                self.get_logger().info(
                    "ColdStartAgent done, press any direction button to switch to tracking agent.",
                    throttle_duration_sec=10.0,
                )
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working and (self._joystick.data.L1):
                self.get_logger().info("L1 button pressed, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            if status != AgentStatus.Working and (self._joystick.data.up):
                if "walk" in self.available_agents.keys():
                    self.get_logger().warn("up button pressed, but there is a walk agent registered. ignored")

        elif self.current_agent_name == "walk":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if self._joystick.data.up:
                self.get_logger().info("up button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("diveroll4-ziwen-0-retargeted.npz")
            elif self._joystick.data.down:
                self.get_logger().info("down button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("kneelClimbStep1-x-0.1-ziwen-retargeted.npz")
            elif self._joystick.data.left:
                self.get_logger().info("left button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("rollVault11-ziwen-retargeted.npz")
            elif self._joystick.data.right:
                self.get_logger().info("right button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("jumpsit2-ziwen-retargeted.npz")
            elif self._joystick.data.X:
                self.get_logger().info("right button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset("superheroLanding-retargeted.npz")

        elif self.current_agent_name == "tracking":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if self._joystick.data.L1:
                self.get_logger().info(
                    "L1 button pressed, switching to walk agent (no matter whether the tracking agent is done)."
                )
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            if status == AgentStatus.Ended and ("walk" in self.available_agents.keys()):
                # switch to walk agent
                self.get_logger().info("TrackingAgent done, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            elif status == AgentStatus.Ended:
                self.get_logger().info("TrackingAgent done, turning off motors.")
                self._turn_off_motors()
                sys.exit(0)

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

    def _update_base_velocity_cmd_from_joystick(self):
        """Write the joystick-derived velocity command to the generic buffer.

        The walk agent reads ``base_velocity_cmd`` via
        ``_get_base_velocity_cmd_obs``.  Other agents are free to ignore it.
        """
        jy = self._joystick.data
        x_vel = jy.ly * self._x_vel_scale if jy.ly is not None else 0.0
        y_vel = -jy.lx * self._y_vel_scale if jy.lx is not None else 0.0
        yaw_vel = -jy.rx * self._yaw_vel_scale if jy.rx is not None else 0.0
        self.base_velocity_cmd = np.array([x_vel, y_vel, yaw_vel], dtype=np.float32)

    def vis_callback(self):
        agent: PerceptiveTrackerAgent = self.available_agents["tracking"]
        cursor = agent.motion_cursor_idx
        # Publish JointState for target joints
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.sim_joint_names
        joint_pos = agent.motion_data.joint_pos[cursor]
        joint_vel = agent.motion_data.joint_vel[cursor]
        js.position = joint_pos.tolist()
        js.velocity = joint_vel.tolist()
        js.effort = [0.0] * len(joint_pos)
        self.joint_state_publisher.publish(js)
        # Broadcast TF for target base
        pos = agent.motion_data.base_pos[cursor]
        quat = agent.motion_data.base_quat[cursor]
        t = TransformStamped()
        t.header.stamp = js.header.stamp
        t.header.frame_id = "world"
        t.child_frame_id = "torso_link"
        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])
        t.transform.rotation.w = float(quat[0])
        t.transform.rotation.x = float(quat[1])
        t.transform.rotation.y = float(quat[2])
        t.transform.rotation.z = float(quat[3])
        self.tf_broadcaster.sendTransform(t)


def main(args):
    rclpy.init()

    node = G1TrackingNode(
        rs_resolution=(480, 270),  # (width, height)
        rs_fps=60,
        camera_individual_process=True,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        motion_vis=args.motion_vis,
        dryrun=not args.nodryrun,
    )

    tracking_agent = PerceptiveTrackerAgent(
        logdir=args.logdir,
        motion_file_dir=args.motion_dir,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
        ros_node=node,
    )
    if args.walk_logdir is not None:
        walk_agent = WalkAgent(
            logdir=args.walk_logdir,
            ros_node=node,
        )
        cold_start_agent = ColdStartAgent(
            startup_step_size=args.startup_step_size,
            ros_node=node,
            joint_target_pos=walk_agent.default_joint_pos,
            action_terms=walk_agent.action_terms,
            p_gains=walk_agent.p_gains * args.kpkd_factor,
            d_gains=walk_agent.d_gains * args.kpkd_factor,
        )
        node.register_agent("walk", walk_agent)
    else:
        cold_start_agent = tracking_agent.get_cold_start_agent(args.startup_step_size, args.kpkd_factor)

    if args.depth_vis or args.pointcloud_vis:
        node.publish_auxiliary_static_transforms("realsense_depth_link_transform")

    node.register_agent("cold_start", cold_start_agent)
    node.register_agent("tracking", tracking_agent)

    node.start_ros_handlers()
    node.get_logger().info("G1TrackingNode is ready to run.")
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

    parser = argparse.ArgumentParser(description="G1 Tracking Node")
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the agent from",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        help="Directory to the motion files",
    )
    parser.add_argument(
        "--walk_logdir",
        type=str,
        help="Directory to load the walk agent from",
        default=None,
    )
    parser.add_argument(
        "--startup_step_size",
        type=float,
        default=0.2,
        help="Startup step size for the cold start agent (default: 0.2)",
    )
    parser.add_argument(
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
    )
    parser.add_argument(
        "--kpkd_factor",
        type=float,
        default=2.0,
        help="KPKD factor for the cold start agent (default: 2.0)",
    )
    parser.add_argument(
        "--motion_vis",
        action="store_true",
        default=False,
        help="Visualize the motion sequence by publishing motion sequence as joint state, need robot state publisher to visuzlize the robot model (default: False)",
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
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()

    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
