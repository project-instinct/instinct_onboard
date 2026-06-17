import os
import sys

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.agents.tracking_agent import TrackerAgent
from instinct_onboard.joystick import UnitreeJoyStick
from instinct_onboard.ros_nodes.unitree import UnitreeNode

"""
G1 Tracking Node

A ROS2 node for controlling Unitree G1 robot using basic tracking agent without depth perception.
This script executes pre-recorded motion sequences by tracking joint positions and velocities
from motion files.

Features:
    - TrackerAgent for motion sequence playback
    - Cold start agent for safe initialization
    - Motion visualization via joint states and TF
    - Simple two-agent workflow

Command-Line Arguments:
    Required:
        --logdir PATH          Directory containing the trained tracking agent model
                              (must contain exported/actor.onnx)
        --motion_dir PATH      Directory containing retargeted motion files (.npz format)

    Optional:
        --startup_step_size FLOAT
                              Startup step size for cold start agent (default: 0.2)
        --nodryrun            Disable dry run mode (default: False, runs in dry run mode)
        --kpkd_factor FLOAT   KP/KD gain multiplier for cold start agent (default: 1.0)
        --debug                Enable debug mode with debugpy (listens on 0.0.0.0:6789)

Agent Workflow:
    1. Cold Start Agent (initial state)
       - Automatically starts when node launches
       - Transitions robot to initial pose from motion file
       - Press 'L1' button to switch to tracking agent after completion

    2. Tracking Agent
       - Executes motion sequences from motion files
       - Press 'A' button to match motion to current robot heading
       - Automatically turns off motors and exits when motion completes

Joystick Controls:
    A Button:     Match tracking motion to current robot heading
    L1 Button:   Switch from cold start to tracking agent

Example Usage:
    Basic usage with required arguments:
        python g1_track.py --logdir /path/to/tracking/model --motion_dir /path/to/motions

    Real robot control (disable dry run):
        python g1_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --nodryrun

    With custom startup parameters:
        python g1_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --startup_step_size 0.3 \\
            --kpkd_factor 1.5

    Debug mode:
        python g1_track.py \\
            --logdir /path/to/tracking/model \\
            --motion_dir /path/to/motions \\
            --debug

Notes:
    - The script runs at 50Hz main loop frequency (20ms period)
    - Robot configuration: G1_29Dof_TorsoBase (29 degrees of freedom)
    - This script does NOT use depth camera (unlike g1_perceptive_track.py)
    - Motion visualization publishes joint states and TF transforms for RViz
"""


class G1TrackingNode(UnitreeNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None

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
        # start the visualization timer with 100ms duration
        vis_duration = 0.1
        self.vis_timer = self.create_timer(vis_duration, self.vis_callback)

    def main_loop_callback(self):
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
            if status != AgentStatus.Working:
                self.get_logger().info(
                    "ColdStartAgent done, press 'L1' to switch to tracking agent.", throttle_duration_sec=10.0
                )
            self.send_target_joint_state(tjs)
            if status != AgentStatus.Working and (self._joystick.data.L1):
                self.get_logger().info("L1 button pressed, switching to tracking agent.")
                self.current_agent_name = "tracking"
                self.available_agents[self.current_agent_name].reset()

        elif self.current_agent_name == "tracking":
            tjs, status = self.available_agents[self.current_agent_name].step()
            self.send_target_joint_state(tjs)
            if status == AgentStatus.Ended:
                self.get_logger().info("TrackingAgent done, turning off motors.")
                self._turn_off_motors()
                sys.exit(0)

    def vis_callback(self):
        agent: TrackerAgent = self.available_agents["tracking"]
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
        robot_class_name="G1_29Dof_TorsoBase",
        dryrun=not args.nodryrun,
    )

    # Wire up the wireless controller (joystick) for agent switching.
    joystick = UnitreeJoyStick(node)
    node._joystick = joystick

    tracking_agent = TrackerAgent(
        logdir=args.logdir,
        motion_file_dir=args.motion_dir,
        ros_node=node,
    )
    cold_start_agent = tracking_agent.get_cold_start_agent(args.startup_step_size, args.kpkd_factor)

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
        help="Path to the motion file",
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
        default=1.0,
        help="KPKD factor for the cold start agent (default: 1.0)",
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
