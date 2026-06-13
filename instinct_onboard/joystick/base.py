from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


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


class JoyStickBase(ABC):
    """Abstract base for OEM-specific joystick providers.

    Handles the shared lifecycle: creates a ROS subscription on construction,
    routes every incoming message through :meth:`_parse_message`, and exposes
    the latest parsed state via :attr:`data`.

    Subclasses must implement:

    * :meth:`_get_message_type` — return the ROS message class.
    * :meth:`_parse_message` — populate ``self._data`` from a message.

    Optionally override :meth:`_check_safety_shutdown` to trigger a shutdown
    on a button combination (e.g. R2 or L2 on Unitree controllers).
    """

    def __init__(
        self,
        ros_node,
        joy_stick_topic: str,
        safety_shutdown_callback: Optional[Callable[[], None]] = None,
    ):
        """Create the joystick provider and subscribe to *joy_stick_topic*.

        Args:
            ros_node: An ``rclpy.node.Node`` used to create the subscription.
                Must expose a ``_turn_off_motors()`` method (as ``RealNode``
                does) — it is **always** called before the optional callback
                during a safety shutdown.
            joy_stick_topic: ROS topic name for the controller messages.
            safety_shutdown_callback: Optional extra hook invoked **after**
                ``_turn_off_motors()`` when :meth:`_check_safety_shutdown`
                returns True.  The default behaviour (motor-off + SystemExit)
                is sufficient for most scripts, so this can usually be left
                as *None*.
        """
        self._ros_node = ros_node
        self._data = JoyStickData()
        self._safety_shutdown_callback = safety_shutdown_callback

        msg_type = self._get_message_type()
        self._subscription = ros_node.create_subscription(msg_type, joy_stick_topic, self._joy_stick_callback, 10)
        ros_node.get_logger().info(f"{type(self).__name__} subscribed to '{joy_stick_topic}'.")

    # ------------------------------------------------------------------
    # Public read-only access
    # ------------------------------------------------------------------

    @property
    def data(self) -> JoyStickData:
        """The latest parsed joystick data (read-only)."""
        return self._data

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_message_type(self):
        """Return the ROS message **class** this joystick subscribes to.

        Example: ``return WirelessController``.
        """

    @abstractmethod
    def _parse_message(self, msg) -> None:
        """Parse *msg* and write its fields into ``self._data``.

        Called from the ROS subscription callback on every incoming message.
        """

    def _check_safety_shutdown(self, _msg) -> bool:
        """Return True when *_msg* signals an emergency stop.

        The default implementation never triggers a shutdown — override in a
        subclass to match the OEM controller's stop combination.
        """
        return False

    # ------------------------------------------------------------------
    # Internal plumbing
    # ------------------------------------------------------------------

    def _handle_safety_shutdown(self):
        """Turn off motors, run optional callback, then raise ``SystemExit``."""
        self._ros_node.get_logger().warn("Safety shutdown triggered via joystick.")
        self._ros_node._turn_off_motors()
        if self._safety_shutdown_callback is not None:
            self._safety_shutdown_callback()
        raise SystemExit("Safety shutdown triggered via joystick.")

    def _joy_stick_callback(self, msg):
        self._ros_node.get_logger().info("Joystick data received.", once=True)
        self._parse_message(msg)
        if self._check_safety_shutdown(msg):
            self._handle_safety_shutdown()
