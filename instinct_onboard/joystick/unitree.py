from unitree_go.msg import WirelessController

import instinct_onboard.robot_cfgs as robot_cfgs
from instinct_onboard.joystick.base import JoyStickBase


class UnitreeJoyStick(JoyStickBase):
    """Unitree Wireless Controller joystick provider.

    Subscribes to ``/wirelesscontroller`` (or a custom topic), parses
    ``WirelessController`` messages into :attr:`data`, and triggers a safety
    shutdown when R2 or L2 is pressed.

    The entry script should store the joystick instance and read its
    ``data`` directly for button checks and velocity computation.  Use
    :attr:`RealNode.base_velocity_cmd` to pass the computed velocity to
    agents (via ``_get_base_velocity_cmd_obs``).

    Example::

        joystick = UnitreeJoyStick(node)
        node._joystick = joystick

        # In the main loop, before calling agent.step():
        jy = node._joystick.data
        node.base_velocity_cmd = np.array([jy.ly * 0.5, -jy.lx * 0.5, -jy.rx], dtype=np.float32)
    """

    def __init__(
        self,
        ros_node,
        joy_stick_topic: str = "/wirelesscontroller",
        safety_shutdown_callback=None,
    ):
        super().__init__(ros_node, joy_stick_topic, safety_shutdown_callback)

    # ------------------------------------------------------------------
    # JoyStickBase contract
    # ------------------------------------------------------------------

    def _get_message_type(self):
        return WirelessController

    def _parse_message(self, msg: WirelessController) -> None:
        # -- buttons -------------------------------------------------------
        self._data.A = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.A)
        self._data.B = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.B)
        self._data.X = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.X)
        self._data.Y = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.Y)
        self._data.start = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.start)
        self._data.select = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.select)
        self._data.L1 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.L1)
        self._data.R1 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.R1)
        self._data.L2 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.L2)
        self._data.R2 = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.R2)
        self._data.up = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.up)
        self._data.down = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.down)
        self._data.left = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.left)
        self._data.right = bool(msg.keys & robot_cfgs.UnitreeWirelessButtons.right)

        # -- axes ----------------------------------------------------------
        self._data.lx = msg.lx
        self._data.ly = msg.ly
        self._data.rx = msg.rx
        self._data.ry = msg.ry
        # left_trigger / right_trigger are not available on Unitree controllers

    def _check_safety_shutdown(self, msg: WirelessController) -> bool:
        """Shut down when R2 or L2 is pressed."""
        return bool(
            (msg.keys & robot_cfgs.UnitreeWirelessButtons.R2) or (msg.keys & robot_cfgs.UnitreeWirelessButtons.L2)
        )
