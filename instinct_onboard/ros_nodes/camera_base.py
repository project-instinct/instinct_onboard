from __future__ import annotations

import ctypes
import multiprocessing as mp
import multiprocessing.shared_memory as mp_shm
import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np
from rclpy.node import Node

from instinct_onboard.utils import _depth_to_ros_pointcloud_msg


class MpSharedHeader(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_double),  # bytes: 8
        ("writer_status", ctypes.c_uint32),  # bytes: 4, 0: idle, 1: writing
        ("writer_termination_signal", ctypes.c_uint32),  # bytes: 4, 0: alive, 1: should terminate
        ("frame_counter", ctypes.c_uint32),  # bytes: 4, monotonically increasing per write
        ("_pad", ctypes.c_uint32 * 3),  # bytes: 12, pad to 32 bytes
    ]


SIZE_OF_MP_SHARED_HEADER = ctypes.sizeof(MpSharedHeader)  # bytes: 32
assert SIZE_OF_MP_SHARED_HEADER == 32

CAMERA_PROCESS_FREQUENCY_CHECK_INTERVAL = 500


# ------------------------------------------------------------------
# Cross-process memory barrier
# ------------------------------------------------------------------
# On ARM (Jetson, Raspberry Pi) weakly-ordered memory can reorder stores
# so that ``writer_status = 0`` becomes visible before ``depth_buffer[:]``
# has propagated.  ``__sync_synchronize()`` emits a full memory fence
# (``DMB SY`` on ARM, ``MFENCE`` on x86) that prevents this.
#
# The symbol lives in libgcc_s on Linux; on platforms where it is unavailable
# we degrade gracefully — the frame_counter check still catches the vast
# majority of reordering windows.

try:
    _libgcc = ctypes.CDLL("libgcc_s.so.1")
    _sync_synchronize = _libgcc.__sync_synchronize
except (OSError, AttributeError):
    try:
        # macOS fallback: libgcc is typically statically linked into libSystem
        _libgcc = ctypes.CDLL(None)
        _sync_synchronize = _libgcc.__sync_synchronize
    except (OSError, AttributeError):
        _sync_synchronize = None  # no barrier available — accept the risk


def _release_barrier() -> None:
    """Memory fence that ensures all prior writes are visible before a subsequent store."""
    if _sync_synchronize is not None:
        _sync_synchronize()


def _acquire_barrier() -> None:
    """Memory fence that ensures all subsequent reads see stores that happened before the fence."""
    if _sync_synchronize is not None:
        _sync_synchronize()


class CameraBase(ABC):
    """Abstract base class defining the common interface for all camera models.

    This is designed as a MRO-compatible mixin that can be inherited alongside
    ROS2 Node classes.  It detects whether it is running inside a ROS node
    (via ``isinstance(self, Node)``) and uses ROS logging when available;
    otherwise it stays silent.
    """

    def __init__(
        self,
        *args,
        depth_resolution: tuple[int, int] | None = (480, 270),  # (width, height), None to disable
        color_resolution: tuple[int, int] | None = None,  # (width, height), None to disable
        camera_serial: str | None = None,
        depth_vfov_deg: float = 58.0,
        depth_frame_id: str = "camera_depth_link",  # TF frame name for the depth optical frame
        depth_fps: int = 60,  # target depth frame rate (Hz)
        **kwargs,
    ):
        if depth_resolution is None and color_resolution is None:
            raise ValueError("Both depth_resolution and color_resolution are None. Please enable at least one.")
        super().__init__(*args, **kwargs)
        self.depth_resolution = depth_resolution
        self.color_resolution = color_resolution
        self.camera_serial = camera_serial
        self.depth_vfov_deg = depth_vfov_deg
        self.depth_frame_id = depth_frame_id
        self.depth_fps = depth_fps
        self._depth_data: np.ndarray | None = None
        self._color_data: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _try_log(self, level: str, msg: str) -> None:
        """Log via ROS logger if running inside a ROS Node; otherwise stay silent."""
        if isinstance(self, Node):
            logger = self.get_logger()
            getattr(logger, level)(msg)

    # ------------------------------------------------------------------
    # Abstract / overridable interface
    # ------------------------------------------------------------------
    @abstractmethod
    def refresh_camera_data(self) -> bool:
        """Refresh the camera data by reading from the hardware.

        Returns:
            True if new data was acquired, False otherwise.
        """
        ...

    def get_depth_image(self) -> np.ndarray | None:
        """Return the latest depth image (height, width) in metres."""
        return self._depth_data

    def get_color_image(self) -> np.ndarray | None:
        """Return the latest color image (height, width, 3) as uint8 RGB."""
        return self._color_data

    def depth_image_to_pointcloud_msg(self, depth: np.ndarray):
        """Reproject a depth image to a PointCloud2 message in the camera frame.

        Args:
            depth: Depth image as a (height, width) float32 array in metres.

        Returns:
            sensor_msgs.msg.PointCloud2 message.
        """
        stamp = None
        if isinstance(self, Node):
            stamp = self.get_clock().now().to_msg()
        return _depth_to_ros_pointcloud_msg(
            depth=depth,
            frame_id=self.depth_frame_id,
            vfov_deg=self.depth_vfov_deg,
            stamp=stamp,
        )

    def destroy_node(self):
        """Cleanly shut down the camera and release resources."""
        if hasattr(super(), "destroy_node"):
            super().destroy_node()


# ======================================================================
# Default subprocess function
# ======================================================================


def camera_subprocess_func(
    camera_cls: type,
    depth_resolution: tuple[int, int] | None,
    color_resolution: tuple[int, int] | None,
    camera_kwargs: dict,
    depth_shm_name: str | None,
    color_shm_name: str | None,
    affinity: set[int] | None,
) -> None:
    """Run a :class:`CameraBase` implementation in a subprocess.

    The function instantiates *camera_cls*, attaches to pre-created shared
    memory blocks, and loops: read hardware → write shared memory → check
    termination.

    Parameters
    ----------
    camera_cls:
        The :class:`CameraBase` subclass to instantiate in this process.
    depth_resolution / color_resolution:
        Passed to *camera_cls* constructor.
    camera_kwargs:
        Additional keyword arguments forwarded to *camera_cls*.
    depth_shm_name / color_shm_name:
        Names of shared memory blocks created by :class:`CameraProcessSpawner`.
        ``None`` means that stream is disabled.
    affinity:
        Optional CPU affinity set for this process.
    """
    if affinity is not None:
        os.sched_setaffinity(os.getpid(), affinity)

    # -- instantiate the camera in this subprocess ---------------------------
    camera: CameraBase = camera_cls(
        depth_resolution=depth_resolution,
        color_resolution=color_resolution,
        **camera_kwargs,
    )

    # -- attach shared memory ------------------------------------------------
    depth_shm = None
    color_shm = None
    depth_header = None
    color_header = None
    depth_buffer = None
    color_buffer = None

    if depth_shm_name is not None:
        depth_shm = mp_shm.SharedMemory(name=depth_shm_name)
        depth_header = MpSharedHeader.from_buffer(depth_shm.buf)
        depth_buffer = np.ndarray(
            depth_resolution[::-1],
            dtype=np.float32,
            buffer=depth_shm.buf,
            offset=SIZE_OF_MP_SHARED_HEADER,
        )

    if color_shm_name is not None:
        color_shm = mp_shm.SharedMemory(name=color_shm_name)
        color_header = MpSharedHeader.from_buffer(color_shm.buf)
        color_buffer = np.ndarray(
            (*color_resolution[::-1], 3),
            dtype=np.uint8,
            buffer=color_shm.buf,
            offset=SIZE_OF_MP_SHARED_HEADER,
        )

    # -- main loop -----------------------------------------------------------
    camera_process_counter = 0
    camera_process_start_time = time.time()
    try:
        while True:
            camera.refresh_camera_data()

            if depth_buffer is not None:
                depth_data = camera.get_depth_image()
                if depth_data is not None:
                    depth_header.writer_status = 1
                    depth_buffer[:] = depth_data
                    depth_header.frame_counter += 1
                    depth_header.timestamp = time.time()
                    _release_barrier()  # ensure buffer stores are visible before unlock
                    depth_header.writer_status = 0

            if color_buffer is not None:
                color_data = camera.get_color_image()
                if color_data is not None:
                    color_header.writer_status = 1
                    color_buffer[:] = color_data
                    color_header.frame_counter += 1
                    color_header.timestamp = time.time()
                    _release_barrier()  # ensure buffer stores are visible before unlock
                    color_header.writer_status = 0

            # -- periodic frequency log ----------------------------------------
            camera_process_counter += 1
            if camera_process_counter % CAMERA_PROCESS_FREQUENCY_CHECK_INTERVAL == 0:
                elapsed = time.time() - camera_process_start_time
                print(f"Camera process running at" f" {camera_process_counter / elapsed:.4f} Hz.")
                camera_process_counter = 0
                camera_process_start_time = time.time()

            # -- termination check: all enabled streams must be signalled ----
            depth_terminated = (depth_header is None) or (depth_header.writer_termination_signal == 1)
            color_terminated = (color_header is None) or (color_header.writer_termination_signal == 1)
            if depth_terminated and color_terminated:
                break
    finally:
        camera.destroy_node()
        if depth_shm is not None:
            depth_shm.close()
        if color_shm is not None:
            color_shm.close()


# ======================================================================
# CameraProcessSpawner
# ======================================================================


class CameraProcessSpawner(CameraBase):
    """Mixin that runs a :class:`CameraBase` implementation in a separate process.

    Shared memory is used to exchange depth and color images between the
    camera subprocess and the main (ROS node) process.  Each enabled stream
    gets its own shared memory block with an :class:`MpSharedHeader` prefix.

    Intended for use via multiple inheritance in entry scripts::

        class G1ParkourNode(RealsenseMPCamera, UnitreeNode):
            ...
    """

    def __init__(
        self,
        *args,
        camera_cls: type,
        camera_process_affinity: set[int] | None = None,
        camera_dead_behavior: Literal["restart", "raise_error"] = "restart",
        subprocess_func: Callable = camera_subprocess_func,
        camera_kwargs: dict | None = None,
        **kwargs,
    ):
        # Consume our own parameters; everything else flows up the MRO.
        self.camera_cls = camera_cls
        self.camera_process_affinity = camera_process_affinity
        if camera_dead_behavior not in ("restart", "raise_error"):
            raise ValueError(f"camera_dead_behavior must be 'restart' or 'raise_error', got '{camera_dead_behavior}'")
        self.camera_dead_behavior = camera_dead_behavior
        self.subprocess_func = subprocess_func
        self._camera_kwargs = camera_kwargs or {}

        super().__init__(*args, **kwargs)

        # Ensure camera-level params also reach the subprocess via camera_kwargs.
        self._camera_kwargs.setdefault("camera_serial", self.camera_serial)
        self._camera_kwargs.setdefault("depth_vfov_deg", self.depth_vfov_deg)

        # Shared memory and process handles — created AFTER super().__init__()
        # so that CameraBase has already stored depth_resolution etc.
        self._depth_shm: mp_shm.SharedMemory | None = None
        self._color_shm: mp_shm.SharedMemory | None = None
        self._depth_shm_name: str | None = None
        self._color_shm_name: str | None = None
        self._depth_header: MpSharedHeader | None = None
        self._color_header: MpSharedHeader | None = None
        self._depth_buffer: np.ndarray | None = None
        self._color_buffer: np.ndarray | None = None
        self.camera_process: mp.Process | None = None

        # Per-stream frame counters to detect new data (prevents reading
        # zero-initialised SHM as real frames before the first write).
        self._last_depth_frame_counter: int = 0
        self._last_color_frame_counter: int = 0

        self._create_shared_memory()
        self.start_process()

    # ------------------------------------------------------------------
    # Shared memory
    # ------------------------------------------------------------------
    def _make_shm_name(self, stream: str) -> str:
        """Build a unique shared memory name for *stream* (``"depth"`` or ``"color"``).

        Always reads ``camera_serial`` from :attr:`_camera_kwargs` so the SHM
        name matches the device the subprocess will open — never from
        ``self.camera_serial`` directly, which could differ from the kwargs value.
        """
        serial = self._camera_kwargs.get("camera_serial")
        if serial:
            suffix = serial
        else:
            suffix = str(os.getpid())
        return f"camera_{stream}_{suffix}"

    def _create_shared_memory(self) -> None:
        """Create shared memory blocks for enabled streams."""
        if self.depth_resolution is not None:
            self._depth_data = np.zeros(self.depth_resolution[::-1], dtype=np.float32)
            shm_size = (
                SIZE_OF_MP_SHARED_HEADER + int(np.prod(self.depth_resolution[::-1])) * np.dtype(np.float32).itemsize
            )
            self._depth_shm_name = self._make_shm_name("depth")
            self._depth_shm = mp_shm.SharedMemory(name=self._depth_shm_name, create=True, size=shm_size)
            self._depth_header = MpSharedHeader.from_buffer(self._depth_shm.buf)
            self._depth_buffer = np.ndarray(
                self.depth_resolution[::-1],
                dtype=np.float32,
                buffer=self._depth_shm.buf,
                offset=SIZE_OF_MP_SHARED_HEADER,
            )

        if self.color_resolution is not None:
            self._color_data = np.zeros((*self.color_resolution[::-1], 3), dtype=np.uint8)
            shm_size = (
                SIZE_OF_MP_SHARED_HEADER + int(np.prod(self.color_resolution[::-1])) * 3 * np.dtype(np.uint8).itemsize
            )
            self._color_shm_name = self._make_shm_name("color")
            self._color_shm = mp_shm.SharedMemory(name=self._color_shm_name, create=True, size=shm_size)
            self._color_header = MpSharedHeader.from_buffer(self._color_shm.buf)
            self._color_buffer = np.ndarray(
                (*self.color_resolution[::-1], 3),
                dtype=np.uint8,
                buffer=self._color_shm.buf,
                offset=SIZE_OF_MP_SHARED_HEADER,
            )

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------
    def start_process(self) -> None:
        """Spawn the camera subprocess (or re-spawn on restart).

        On restart the existing shared memory blocks are reused so that
        the main-process reader never sees an empty buffer.
        """
        if self.camera_process is not None and self.camera_process.is_alive():
            self._try_log("warn", "Camera process is already alive; not starting a new one.")
            return
        self.camera_process = mp.Process(
            target=self.subprocess_func,
            args=(
                self.camera_cls,
                self.depth_resolution,
                self.color_resolution,
                self._camera_kwargs,
                self._depth_shm_name,
                self._color_shm_name,
                self.camera_process_affinity,
            ),
            daemon=True,
        )
        self.camera_process.start()

    def stop_process(self) -> None:
        """Signal the subprocess to terminate and clean up."""
        if self._depth_header is not None:
            self._depth_header.writer_termination_signal = 1
        if self._color_header is not None:
            self._color_header.writer_termination_signal = 1

        if self.camera_process is not None:
            self.camera_process.join(timeout=1.0)
            if self.camera_process.is_alive():
                self._try_log("warn", "Camera process still alive after timeout. Terminating.")
                self.camera_process.terminate()
                self.camera_process.join()
            self.camera_process = None

        self._destroy_shared_memory()

    def _destroy_shared_memory(self) -> None:
        """Release buffer views and destroy shared memory blocks.

        Separated from :meth:`stop_process` so it can be reused when
        restart logic needs to recreate SHM from scratch.
        """
        self._depth_buffer = None
        self._color_buffer = None
        self._depth_header = None
        self._color_header = None

        if self._depth_shm is not None:
            self._depth_shm.close()
            self._depth_shm.unlink()
            self._depth_shm = None
            self._depth_shm_name = None

        if self._color_shm is not None:
            self._color_shm.close()
            self._color_shm.unlink()
            self._color_shm = None
            self._color_shm_name = None

    def prevent_camera_process_dead(self) -> None:
        """Check camera process health and act according to ``camera_dead_behavior``."""
        if self.camera_process is not None and self.camera_process.is_alive():
            return
        # kill that process in memory if it is dead.
        if self.camera_process is not None:
            # join() is safe here: is_alive() already returned False above,
            # so the OS has already exited the process and waitpid is instant.
            self.camera_process.join()
            self.camera_process = None
        if self.camera_dead_behavior == "restart":
            self._try_log("error", "Camera process is not alive. Restarting.")
            self.start_process()
        elif self.camera_dead_behavior == "raise_error":
            raise RuntimeError("Camera process is not alive.")

    # ------------------------------------------------------------------
    # CameraBase interface overrides
    # ------------------------------------------------------------------
    def refresh_camera_data(self) -> bool:
        """Read the latest frames from shared memory.

        Always checks process health first.  Copies data from shared memory
        into the local ``_depth_data`` / ``_color_data`` buffers when the
        writer is idle.
        """
        refreshed = False
        self.prevent_camera_process_dead()

        if self._depth_buffer is not None and self._depth_header is not None:
            if self._depth_header.writer_status == 0:
                _acquire_barrier()  # ensure buffer reads see the writer's stores
                fc = self._depth_header.frame_counter
                if fc != self._last_depth_frame_counter:
                    self._depth_data[:] = self._depth_buffer
                    self._last_depth_frame_counter = fc
                    refreshed = True

        if self._color_buffer is not None and self._color_header is not None:
            if self._color_header.writer_status == 0:
                _acquire_barrier()  # ensure buffer reads see the writer's stores
                fc = self._color_header.frame_counter
                if fc != self._last_color_frame_counter:
                    self._color_data[:] = self._color_buffer
                    self._last_color_frame_counter = fc
                    refreshed = True  # consider refreshed if at least colour arrived

        return refreshed

    def destroy_node(self):
        """Shut down the camera process and release shared memory, then chain up."""
        self.stop_process()
        super().destroy_node()
