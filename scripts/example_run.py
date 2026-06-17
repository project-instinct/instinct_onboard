import os

import numpy as np
import rclpy
import yaml

from instinct_onboard.agents.base import AgentStatus, ColdStartAgent
from instinct_onboard.ros_nodes.unitree import UnitreeNode


class ExampleNode(UnitreeNode):
    """An example ROS node that patches Node and Agent to run the system."""

    def start_ros_handlers(self):
        """Start the ROS handlers for the node."""
        super().start_ros_handlers()
        main_loop_duration = self.get_cfg_main_loop_duration()
        self.get_logger().info(f"Starting main loop with duration: {main_loop_duration} seconds.")
        self.create_timer(main_loop_duration, self.main_loop_callback)

    def main_loop_callback(self):
        """Main loop callback for the ROS node."""
        # This is where you would implement the main loop logic for your ROS node.
        # For example, you could publish messages, process incoming data, etc.
        self.get_logger().info("ExampleNode main loop is running.", throttle_duration_sec=5.0)
        # You can also call agent methods here if needed.
        tjs, status = self.agent.step()
        self.send_target_joint_state(tjs)


def main(args):
    """Main function to run the example ROS node."""
    rclpy.init()

    logdir = os.path.expanduser(args.logdir)
    with open(os.path.join(logdir, "params", "env.yaml")) as f:
        cfg = yaml.unsafe_load(f)
    node = ExampleNode(cfg=cfg, dryrun=not args.nodryrun)

    agent = ColdStartAgent(
        startup_step_size=0.2,
        ros_node=node,
        joint_target_pos=np.zeros(node.NUM_JOINTS, dtype=np.float32),
    )
    node.agent = agent

    node.start_ros_handlers()
    node.get_logger().info("ExampleNode is ready to run.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run the example ROS node.")
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.expanduser("~/instinct_onboard/logs/example"),
        help="The directory to store logs and models.",
    )
    parser.add_argument("--nodryrun", action="store_true", help="If set, the node will run without dry run mode.")
    args = parser.parse_args()
    main(args=args)
#     # This is a simple example of how to run a ROS node with an agent.
#     # The agent is a ColdStartAgent that will be initialized with the log directory.
#     # The node will start the agent and handle ROS messages.
#     # The log directory can be specified as a command line argument.
#     # The node will run until it is interrupted by a keyboard signal.
#     # The node will shut down gracefully when it is done.
