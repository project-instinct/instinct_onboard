from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetJointState:
    """The target joint state for the robot.

    This is the single command object exchanged between agent and ROS node.
    It is always full-size (``num_joints == ros_node.NUM_JOINTS``) with five
    parallel arrays: ``position``, ``velocity``, ``effort``, ``kp``, ``kd``.

    A joint with ``effort == 0 and kp == 0 and kd == 0`` is treated as
    "not commanded" by this object (see ``enable_mask``).

    ``__add__`` combines two non-overlapping ``TargetJointState``\\s by
    field-wise addition, provided their ``enable_mask``\\s are disjoint.

    ``merge(other)`` combines two ``TargetJointState``\\s that may target
    the same joints through **different** value fields (position, velocity,
    effort). It checks per-field overlap on those three value fields and
    resolves ``kp``/``kd`` overlaps by taking the element-wise maximum.

    ``fill_disabled_from(other)`` is an overlay op: joints where
    ``self.enable_mask == False`` take their values from ``other``; joints
    already commanded by ``self`` are preserved. Used to compose a fallback
    command onto an authoritative aggregate without requiring disjoint masks.
    """

    # critical fields
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray  # element with 0 as disabled
    kp: np.ndarray  # element with 0 as disabled
    kd: np.ndarray  # element with 0 as disabled

    def __post_init__(self) -> None:
        """Validate that all five arrays share the same length."""
        n = self.position.shape[0]
        for name in ("velocity", "effort", "kp", "kd"):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"TargetJointState: position has {n} joints but {name} has {arr.shape[0]}")

    @property
    def num_joints(self) -> int:
        return self.position.shape[0]

    @property
    def enable_mask(self) -> np.ndarray:
        """The mask of enabled joints.

        A joint is considered commanded if ANY of ``effort``, ``kp``, or ``kd``
        is non-zero at that index. Parentheses are required because ``&`` binds
        tighter than ``!=`` in Python.
        """
        return ((self.effort != 0) | (self.kp != 0) | (self.kd != 0)).astype(np.bool_)

    @property
    def isnan_any(self) -> bool:
        """True if any field (position, velocity, effort, kp, kd) contains a NaN."""
        return (
            np.isnan(self.position).any()
            or np.isnan(self.velocity).any()
            or np.isnan(self.effort).any()
            or np.isnan(self.kp).any()
            or np.isnan(self.kd).any()
        )

    def __len__(self) -> int:
        return self.num_joints

    def copy(self, dtype: np.dtype | type = np.float32) -> TargetJointState:
        """Return a value-copy of this TargetJointState with all fields cast to ``dtype``.

        Preferred over ``copy.deepcopy`` on per-step hot paths: ~10-50x faster for flat ndarrays,
        normalizes dtype (``float32`` default matches ONNX / ROS publish expectations), and stays
        immune to future non-copyable fields (e.g. ROS time stamps, node back-references). Pass
        ``dtype=None`` to preserve each field's original dtype.
        """
        return TargetJointState(
            position=np.array(self.position, dtype=dtype, copy=True),
            velocity=np.array(self.velocity, dtype=dtype, copy=True),
            effort=np.array(self.effort, dtype=dtype, copy=True),
            kp=np.array(self.kp, dtype=dtype, copy=True),
            kd=np.array(self.kd, dtype=dtype, copy=True),
        )

    def as_dtype(self, dtype: np.dtype | type = np.float32) -> TargetJointState:
        """Return a view-cast of this TargetJointState with all fields normalised to ``dtype``.

        Uses ``np.asarray`` internally, so **no copy is made** when a field already has the
        requested dtype — the returned arrays may share memory with the original. This makes
        it suitable for dtype-normalisation at trust boundaries (e.g. before ROS publish)
        without paying a copy penalty on every step.

        Use :meth:`copy` when you need a guaranteed deep copy (e.g. for mutable state that
        must not alias).
        """
        return TargetJointState(
            position=np.asarray(self.position, dtype=dtype),
            velocity=np.asarray(self.velocity, dtype=dtype),
            effort=np.asarray(self.effort, dtype=dtype),
            kp=np.asarray(self.kp, dtype=dtype),
            kd=np.asarray(self.kd, dtype=dtype),
        )

    def __add__(self, other: TargetJointState) -> TargetJointState:
        """Combine two target joint states with the same number of joints.
        It also checks whether the targeted joints are overlapped.
        """
        if self.num_joints != other.num_joints:
            raise ValueError("The number of joints in two target joint states must be the same.")
        if (self.enable_mask & other.enable_mask).any():
            raise ValueError("The enabled joint mask must be non-overlapping.")
        return TargetJointState(
            position=self.position + other.position,
            velocity=self.velocity + other.velocity,
            effort=self.effort + other.effort,
            kp=self.kp + other.kp,
            kd=self.kd + other.kd,
        )

    def merge(self, other: TargetJointState) -> TargetJointState:
        """Combine two ``TargetJointState``\\s that may target the same joints
        through **different** value fields (position, velocity, effort).

        Unlike :meth:`__add__`, this does **not** require disjoint
        ``enable_mask``\\s. Instead it checks per-field overlap on the three
        *value* fields (position, velocity, effort) and resolves ``kp``/``kd``
        overlaps by taking the element-wise maximum — those fields are auxiliary
        control parameters that multiple term types (e.g. position + velocity) may
        legitimately set on the same joint.

        Raises ``ValueError`` when the same value field is non-zero at the same
        joint in both operands — that indicates a genuine write conflict and
        should be caught at config-parsing time by
        :func:`~instinct_onboard.agents.action_term.parse_action_cfgs`.
        """
        if self.num_joints != other.num_joints:
            raise ValueError("The number of joints in two target joint states must be the same.")
        for field in ("position", "velocity", "effort"):
            s = getattr(self, field)
            o = getattr(other, field)
            if ((s != 0) & (o != 0)).any():
                raise ValueError(
                    f"Cannot merge: value field '{field}' has non-zero overlap "
                    f"between the two TargetJointState operands."
                )
        return TargetJointState(
            position=self.position + other.position,
            velocity=self.velocity + other.velocity,
            effort=self.effort + other.effort,
            kp=np.maximum(self.kp, other.kp),
            kd=np.maximum(self.kd, other.kd),
        )

    def fill_disabled_from(self, other: TargetJointState) -> TargetJointState:
        """Overlay ``other`` onto joints where ``self.enable_mask`` is False.

        Unlike :meth:`__add__`, this does **not** require disjoint masks. For
        every joint where ``self`` is "not commanded" (``enable_mask == False``),
        all five fields are taken from ``other``; joints already commanded by
        ``self`` keep their values. Designed for composing a fallback command
        (e.g. default-PD fill-in) on top of an authoritative aggregate.
        """
        if self.num_joints != other.num_joints:
            raise ValueError("The number of joints in two target joint states must be the same.")
        fill_mask = np.logical_not(self.enable_mask)
        return TargetJointState(
            position=np.where(fill_mask, other.position, self.position),
            velocity=np.where(fill_mask, other.velocity, self.velocity),
            effort=np.where(fill_mask, other.effort, self.effort),
            kp=np.where(fill_mask, other.kp, self.kp),
            kd=np.where(fill_mask, other.kd, self.kd),
        )
