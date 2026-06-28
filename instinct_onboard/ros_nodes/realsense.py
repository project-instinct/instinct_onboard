from __future__ import annotations

import numpy as np

from instinct_onboard.ros_nodes.camera_base import CameraBase, CameraProcessSpawner


class RealsenseCamera(CameraBase):
    """Single-process RealSense camera implementation.

    Use this class when you want the camera to run in the same process as
    the ROS node (blocking I/O during :meth:`refresh_camera_data`).

    Inherit from this class alongside a ROS node base for single-process
    camera operation::

        class MyNode(RealsenseCamera, UnitreeNode):
            ...
    """

    def __init__(
        self,
        *args,
        depth_fps: int = 60,
        color_fps: int = 30,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth_fps = depth_fps
        self.color_fps = color_fps

        try:
            import pyrealsense2 as _rs
        except ImportError:
            raise ModuleNotFoundError(
                "pyrealsense2 is required for RealSense camera support. " "Install it via: `pip install pyrealsense2`"
            )

        self.pipeline = _rs.pipeline()
        self.config = _rs.config()

        if self.camera_serial is not None:
            self.config.enable_device(self.camera_serial)

        if self.depth_resolution is not None:
            self.config.enable_stream(
                _rs.stream.depth,
                self.depth_resolution[0],
                self.depth_resolution[1],
                _rs.format.z16,
                depth_fps,
            )

        if self.color_resolution is not None:
            self.config.enable_stream(
                _rs.stream.color,
                self.color_resolution[0],
                self.color_resolution[1],
                _rs.format.rgb8,
                color_fps,
            )

        self.profile = self.pipeline.start(self.config)
        self.align = _rs.align(_rs.stream.depth)
        if self.depth_resolution is not None:
            self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

        # Warm up the pipeline
        _ = self.pipeline.wait_for_frames(1000)

    # ------------------------------------------------------------------
    def refresh_camera_data(self) -> bool:
        """Read one frame set from the RealSense pipeline.

        Returns:
            True if frames were acquired, False otherwise.
        """
        # Compute timeout from the *slowest* enabled stream so that colour
        # frames are not silently dropped when color_fps < depth_fps, and
        # the timeout is still correct when depth is disabled entirely.
        fps_values = []
        if self.depth_resolution is not None:
            fps_values.append(self.depth_fps)
        if self.color_resolution is not None:
            fps_values.append(self.color_fps)
        effective_fps = min(fps_values) if fps_values else 30
        timeout_ms = int(1000 / effective_fps)
        frames = self.pipeline.wait_for_frames(timeout_ms * 2)

        got_data = False

        if self.depth_resolution is not None:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                self._depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * self.depth_scale
                got_data = True

        if self.color_resolution is not None:
            color_frame = frames.get_color_frame()
            if color_frame is not None:
                self._color_data = np.asanyarray(color_frame.get_data())
                got_data = True

        return got_data

    # ------------------------------------------------------------------
    def destroy_node(self):
        """Stop the RealSense pipeline, then chain up."""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        super().destroy_node()


class RealsenseMPCamera(CameraProcessSpawner):
    """Two-process RealSense camera (convenience class).

    Pre-wires :class:`CameraProcessSpawner` with ``camera_cls=RealsenseCamera``
    so that entry scripts only need a single inheritance::

        class G1ParkourNode(RealsenseMPCamera, UnitreeNode):
            ...
    """

    def __init__(
        self,
        *args,
        depth_fps: int = 60,
        color_fps: int = 30,
        **kwargs,
    ):
        super().__init__(
            *args,
            camera_cls=RealsenseCamera,
            camera_kwargs={
                "depth_fps": depth_fps,
                "color_fps": color_fps,
            },
            depth_fps=depth_fps,
            **kwargs,
        )
