#!/usr/bin/env python3
"""
G1 Interaction Node aligned with the sitting interaction checkpoint.

This script treats `interaction` as the sitting-updated version of the same task.
The onboard policy therefore follows the exported sitting deployment contract:

    raw policy obs -> policy normalizer -> depth slice -> 0-depth_image.onnx -> actor.onnx

Features:
    - Depth perception using RealSense D435 camera
    - InteractionAgent backed by the sitting truth checkpoint
    - Motion reference tracking from NPZ files
    - Multiple agent modes: cold start, walk, and interaction
    - Visualization options for debugging

Command-Line Arguments:
    Required:
        --logdir PATH          Directory containing the sitting-aligned interaction export
                              (must contain params/env.yaml, params/agent.yaml,
                              exported/actor.onnx, exported/0-depth_image.onnx,
                              exported/policy_normalizer.npz)
        --walk_logdir PATH    Directory containing the walk agent model
        --motion_dir PATH     Directory containing retargeted motion files (.npz format)

    Optional:
        --startup_step_size FLOAT
                              Startup step size for cold start agent (default: 0.2)
        --nodryrun           Disable dry run mode (default: False, runs in dry run mode)
        --kpkd_factor FLOAT  KP/KD gain multiplier for cold start agent (default: 2.0)
        --depth_vis          Enable depth image visualization (publishes to /debug/depth_image)
        --pointcloud_vis     Enable pointcloud visualization (publishes to /debug/pointcloud)
        --motion_vis         Enable motion visualization by publishing joint states and TF
        --debug              Enable debug mode with debugpy (listens on 0.0.0.0:6789)

Agent Workflow:
    1. Cold Start Agent (initial state)
       - Automatically starts when node launches
       - Transitions robot to initial pose using walk agent's default configuration
       - Press 'L1' to switch to walk agent (if available)
       - Press any direction button to switch to interaction agent

    2. Walk Agent
       - Activated by pressing 'L1' after cold start completes
       - Provides basic walking behavior
       - Press direction buttons to switch to interaction agent with specific motions:
         * UP:    First motion file in directory
         * DOWN:  Second motion file (if available)
         * LEFT/RIGHT: Alternative motions (if available)
       - Press 'L1' from interaction agent to return to walk agent

    3. Interaction Agent
       - Executes interaction motions with sitting-truth observations
       - Press 'A' button to match motion to current robot heading
       - Automatically switches to walk agent when motion completes (if available)
       - Otherwise turns off motors and exits

Joystick Controls:
    A Button:     Match interaction motion to current robot heading
    L1 Button:    Switch between walk and interaction agents
    Direction Buttons: Switch to interaction agent with specific motion files

Example Usage:
    Basic usage with required arguments:
        python g1_interaction.py \
            --logdir /path/to/interaction/model \
            --walk_logdir /path/to/walk/model \
            --motion_dir /path/to/motions

    With visualization options:
        python g1_interaction.py \
            --logdir /path/to/interaction/model \
            --walk_logdir /path/to/walk/model \
            --motion_dir /path/to/motions \
            --depth_vis --pointcloud_vis --motion_vis

    Dry run mode (default, no actual robot control):
        python g1_interaction.py \
            --logdir /path/to/interaction/model \
            --walk_logdir /path/to/walk/model \
            --motion_dir /path/to/motions

    Real robot control (disable dry run):
        python g1_interaction.py \
            --logdir /path/to/interaction/model \
            --walk_logdir /path/to/walk/model \
            --motion_dir /path/to/motions \
            --nodryrun

    With custom startup parameters:
        python g1_interaction.py \
            --logdir /path/to/interaction/model \
            --walk_logdir /path/to/walk/model \
            --motion_dir /path/to/motions \
            --startup_step_size 0.3 \
            --kpkd_factor 1.5

Notes:
    - The script runs at 50Hz main loop frequency (20ms period)
    - RealSense camera is configured at 480x270 resolution, 60 FPS
    - Robot configuration: G1_29Dof_TorsoBase (29 degrees of freedom)
    - Joint position protection ratio: 2.0
    - Camera runs in a separate process for better performance
    - Interaction deployment does not use object-state policy inputs anymore
    - The default interaction logdir points to the sitting export
"""

import os
import queue
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from instinct_onboard.agents.base import ColdStartAgent
from instinct_onboard.agents.interaction_agent import InteractionAgent
from instinct_onboard.agents.walk_agent import WalkAgent
from instinct_onboard.ros_nodes.realsense import UnitreeRsCameraNode

MAIN_LOOP_FREQUENCY_CHECK_INTERVAL = 500
SITTING_INTERACTION_LOGDIR = "/home/fan/dev3/project-instinct/InstinctLab/logs/instinct_rl/g1_interaction/sitting"
INTERACTION_REQUIRED_FILES = (
    "params/env.yaml",
    "params/agent.yaml",
    "exported/actor.onnx",
    "exported/0-depth_image.onnx",
    "exported/policy_normalizer.npz",
)


def validate_interaction_logdir(logdir: str) -> None:
    """Fail fast if the interaction sitting export is incomplete."""
    missing = [os.path.join(logdir, relpath) for relpath in INTERACTION_REQUIRED_FILES if not os.path.exists(os.path.join(logdir, relpath))]
    if missing:
        raise FileNotFoundError("Interaction logdir is missing required sitting artifacts: " + ", ".join(missing))


class G1InteractionNode(UnitreeRsCameraNode):
    """ROS2 node for G1 interaction task."""

    def __init__(
        self,
        *args,
        motion_vis: bool = False,
        interaction_startup_step_size: float = 0.2,
        interaction_kpkd_factor: float = 1.0,
        **kwargs,
    ):
        """Initialize the G1 interaction node.
        
        Args:
            *args: Positional arguments passed to parent UnitreeRsCameraNode
            motion_vis: Whether to visualize motion sequences
            **kwargs: Keyword arguments passed to parent UnitreeRsCameraNode
        """
        super().__init__(*args, **kwargs)
        self.available_agents = dict()
        self.current_agent_name: str | None = None
        self.motion_vis = motion_vis
        self._motion_file_list = []  # List of available motion files
        self.interaction_startup_step_size = interaction_startup_step_size
        self.interaction_kpkd_factor = interaction_kpkd_factor
        self.interaction_cold_start_agent = None

    def register_agent(self, name: str, agent):
        """Register an agent with the node.
        
        Args:
            name: Agent name identifier
            agent: Agent instance
        """
        self.available_agents[name] = agent
        
        # Build motion file list for interaction agent
        if name == "interaction" and hasattr(agent, "all_motion_datas"):
            self._motion_file_list = list(agent.all_motion_datas.keys())
            self.get_logger().info(f"Available interaction motions: {self._motion_file_list}")

    def start_ros_handlers(self):
        """Start ROS handlers including publishers and timers."""
        super().start_ros_handlers()
        
        # Create joint state publisher for motion visualization
        self.joint_state_publisher = self.create_publisher(JointState, "joint_states", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Start main loop timer at 50Hz (20ms period)
        main_loop_duration = 0.02
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.main_loop_timer = self.create_timer(main_loop_duration, self.main_loop_callback)
        
        # Performance monitoring
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_timer_counter: int = 0
            self.main_loop_timer_counter_time = time.time()
            self.main_loop_callback_time_consumptions = queue.Queue(
                maxsize=MAIN_LOOP_FREQUENCY_CHECK_INTERVAL
            )
        
        # Motion visualization timer at 10Hz (100ms period)
        if self.motion_vis:
            self.vis_timer = self.create_timer(0.1, self.vis_callback)

    def main_loop_callback(self):
        """Main control loop executed at 50Hz.
        
        This callback implements the state machine for agent switching:
        - cold_start: Initial pose, transitions to walk/interaction on button press
        - walk: Basic walking, transitions to interaction on direction button
        - interaction: Motion execution, transitions to walk on L1 or on completion
        """
        main_loop_callback_start_time = time.time()
        
        # Auto-start cold start agent if no agent is active
        if self.current_agent_name is None:
            self.get_logger().info("Starting cold start agent automatically.")
            self.get_logger().info(
                "Press 'L1' to switch to walk agent, or direction buttons for interaction.",
                throttle_duration_sec=2.0,
            )
            self.current_agent_name = "cold_start"
            self.available_agents[self.current_agent_name].reset()
            return

        # Handle cold start agent
        elif self.current_agent_name == "cold_start":
            action, done = self.available_agents[self.current_agent_name].step()
            
            if done:
                if "walk" in self.available_agents:
                    self.get_logger().info(
                        "ColdStartAgent done, press 'L1' to switch to walk agent.",
                        throttle_duration_sec=10.0,
                    )
                else:
                    self.get_logger().info(
                        "ColdStartAgent done, press direction buttons to switch to interaction agent.",
                        throttle_duration_sec=10.0,
                    )
            
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            
            # Switch to walk on L1 button
            if done and self.joy_stick_data.L1 and "walk" in self.available_agents:
                self.get_logger().info("L1 button pressed, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            
            # Switch to interaction on direction buttons (if no walk agent)
            if done and not self.joy_stick_data.L1:
                if self.joy_stick_data.up and "interaction" in self.available_agents:
                    self._switch_to_interaction_motion(0)  # First motion
                elif self.joy_stick_data.down and "interaction" in self.available_agents:
                    self._switch_to_interaction_motion(1)  # Second motion
                elif self.joy_stick_data.left and "interaction" in self.available_agents:
                    self._switch_to_interaction_motion(2)  # Third motion
                elif self.joy_stick_data.right and "interaction" in self.available_agents:
                    self._switch_to_interaction_motion(3)  # Fourth motion

        # Handle walk agent
        elif self.current_agent_name == "walk":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            
            # Switch to interaction on direction buttons
            if self.joy_stick_data.up and "interaction" in self.available_agents:
                self.get_logger().info("UP button pressed, switching to interaction agent.")
                self._switch_to_interaction_motion(0)
            elif self.joy_stick_data.down and "interaction" in self.available_agents:
                self.get_logger().info("DOWN button pressed, switching to interaction agent.")
                self._switch_to_interaction_motion(1)
            elif self.joy_stick_data.left and "interaction" in self.available_agents:
                self.get_logger().info("LEFT button pressed, switching to interaction agent.")
                self._switch_to_interaction_motion(2)
            elif self.joy_stick_data.right and "interaction" in self.available_agents:
                self.get_logger().info("RIGHT button pressed, switching to interaction agent.")
                self._switch_to_interaction_motion(3)

        # Handle interaction cold start agent
        elif self.current_agent_name == "interaction_cold_start":
            if self.joy_stick_data.L1 and "walk" in self.available_agents:
                self.get_logger().info("L1 button pressed, aborting interaction cold start and switching to walk agent.")
                self.interaction_cold_start_agent = None
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            else:
                action, done = self.interaction_cold_start_agent.step()
                self.send_action(
                    action,
                    self.interaction_cold_start_agent.action_offset,
                    self.interaction_cold_start_agent.action_scale,
                    self.interaction_cold_start_agent.p_gains,
                    self.interaction_cold_start_agent.d_gains,
                )
                if done:
                    self.get_logger().info("Interaction cold start done, switching to interaction policy.")
                    self.clear_action_buffer()
                    self.interaction_cold_start_agent = None
                    self.current_agent_name = "interaction"

        # Handle interaction agent
        elif self.current_agent_name == "interaction":
            action, done = self.available_agents[self.current_agent_name].step()
            self.send_action(
                action,
                self.available_agents[self.current_agent_name].action_offset,
                self.available_agents[self.current_agent_name].action_scale,
                self.available_agents[self.current_agent_name].p_gains,
                self.available_agents[self.current_agent_name].d_gains,
            )
            
            # Match motion to current heading on A button
            if self.joy_stick_data.A:
                self.get_logger().info(
                    "A button pressed, matching motion to current heading.",
                    throttle_duration_sec=2.0,
                )
                self.available_agents["interaction"].match_to_current_heading()
            
            # Switch to walk on L1 button
            if self.joy_stick_data.L1:
                self.get_logger().info("L1 button pressed, switching to walk agent.")
                self.current_agent_name = "walk"
                self.available_agents[self.current_agent_name].reset()
            
            # Handle motion completion
            if done:
                if "walk" in self.available_agents:
                    self.get_logger().info("Interaction motion done, switching to walk agent.")
                    self.current_agent_name = "walk"
                    self.available_agents[self.current_agent_name].reset()
                else:
                    self.get_logger().info("Interaction motion done, turning off motors.")
                    self._turn_off_motors()
                    sys.exit(0)

        # Log actual frequency every 500 cycles
        if MAIN_LOOP_FREQUENCY_CHECK_INTERVAL > 1:
            self.main_loop_callback_time_consumptions.put(
                time.time() - main_loop_callback_start_time
            )
            self.main_loop_timer_counter += 1
            if self.main_loop_timer_counter % MAIN_LOOP_FREQUENCY_CHECK_INTERVAL == 0:
                time_consumptions = [
                    self.main_loop_callback_time_consumptions.get()
                    for _ in range(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL)
                ]
                self.get_logger().info(
                    f"Actual main loop frequency: "
                    f"{(MAIN_LOOP_FREQUENCY_CHECK_INTERVAL / (time.time() - self.main_loop_timer_counter_time)):.2f} Hz. "
                    f"Mean time consumption: {np.mean(time_consumptions):.4f} s."
                )
                self.main_loop_timer_counter = 0
                self.main_loop_timer_counter_time = time.time()

    def _switch_to_interaction_motion(self, motion_index: int):
        """Switch to interaction agent with specified motion.
        
        Args:
            motion_index: Index into the available motion file list
        """
        if motion_index < len(self._motion_file_list):
            motion_name = self._motion_file_list[motion_index]
            self.get_logger().info(f"Switching to interaction with motion: {motion_name}")
        else:
            motion_name = None
            self.get_logger().warn(
                f"Motion index {motion_index} out of range, using default motion"
            )
        
        interaction_agent = self.available_agents["interaction"]
        interaction_agent.reset(motion_name)
        self.interaction_cold_start_agent = interaction_agent.get_cold_start_agent(
            startup_step_size=self.interaction_startup_step_size,
            kpkd_factor=self.interaction_kpkd_factor,
        )
        self.current_agent_name = "interaction_cold_start"

    def vis_callback(self):
        """Visualization callback for publishing motion sequence as JointState.
        
        This is called at 10Hz to publish the target motion's joint states
        for visualization in rviz or similar tools.
        """
        if "interaction" not in self.available_agents:
            return
        
        agent: InteractionAgent = self.available_agents["interaction"]
        
        if agent.motion_data is None:
            return
        
        cursor = min(agent.motion_cursor_idx, agent.motion_data.total_num_frames - 1)
        
        # Publish JointState for target joints
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = getattr(agent, "policy_joint_names", self.sim_joint_names)
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
    """Main entry point for the G1 interaction node."""
    validate_interaction_logdir(args.logdir)
    rclpy.init()

    node = G1InteractionNode(
        rs_resolution=(480, 270),  # (width, height)
        rs_fps=60,
        camera_individual_process=True,
        joint_pos_protect_ratio=2.0,
        robot_class_name="G1_29Dof_TorsoBase",
        motion_vis=args.motion_vis,
        interaction_startup_step_size=args.startup_step_size,
        interaction_kpkd_factor=args.kpkd_factor,
        dryrun=not args.nodryrun,
    )

    # Create walk agent
    walk_agent = WalkAgent(
        logdir=args.walk_logdir,
        ros_node=node,
    )
    node.register_agent("walk", walk_agent)

    # Create interaction agent
    interaction_agent = InteractionAgent(
        logdir=args.logdir,
        motion_file_dir=args.motion_dir,
        ros_node=node,
        depth_vis=args.depth_vis,
        pointcloud_vis=args.pointcloud_vis,
    )
    node.register_agent("interaction", interaction_agent)

    # Create cold start agent using walk agent's configuration
    cold_start_agent = ColdStartAgent(
        startup_step_size=args.startup_step_size,
        ros_node=node,
        joint_target_pos=walk_agent.default_joint_pos,
        action_scale=walk_agent.action_scale,
        action_offset=walk_agent.action_offset,
        p_gains=walk_agent.p_gains * args.kpkd_factor,
        d_gains=walk_agent.d_gains * args.kpkd_factor,
    )
    node.register_agent("cold_start", cold_start_agent)

    # Publish static transforms for visualization
    if args.depth_vis or args.pointcloud_vis:
        node.publish_auxiliary_static_transforms("realsense_depth_link_transform")

    node.start_ros_handlers()
    node.get_logger().info("G1InteractionNode is ready to run.")
    
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

    parser = argparse.ArgumentParser(
        description="G1 Interaction Node aligned with the sitting interaction export"
    )
    
    # Required arguments
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to load the sitting-aligned interaction agent from",
        default=SITTING_INTERACTION_LOGDIR,
    )
    parser.add_argument(
        "--walk_logdir",
        type=str,
        help="Directory to load the walk agent from (must contain exported/actor.onnx)",
        default="/home/fan/dev3/project-instinct/InstinctLab/logs/instinct_rl/g1_locomotion_flat/20260322_134921_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.40_jointDeviationKnee0.20",
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        help="Directory containing motion files (.npz format)",
        default="/home/fan/dev3/project-instinct/InstinctLab/assets_datasets/interaction/output_npz_29dof_with_object/chair",
    )
    
    # Optional arguments
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
        "--nodryrun",
        action="store_true",
        default=False,
        help="Run the node without dry run mode (default: False)",
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
        "--motion_vis",
        action="store_true",
        default=False,
        help="Visualize the motion sequence by publishing joint states and TF (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with debugpy (default: False)",
    )

    args = parser.parse_args()

    # Enable debugpy if requested
    if args.debug:
        import debugpy

        ip_address = ("0.0.0.0", 6789)
        print("Process: " + " ".join(sys.argv[:]))
        print(f"Is waiting for attach at address: {ip_address[0]}:{ip_address[1]}", flush=True)
        debugpy.listen(ip_address)
        debugpy.wait_for_client()
        debugpy.breakpoint()

    main(args)
