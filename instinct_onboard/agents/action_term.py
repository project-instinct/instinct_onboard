from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import prettytable

from instinct_onboard.ros_nodes.base import RealNode
from instinct_onboard.target_joint_state import TargetJointState


class ActionTermBase(ABC):
    """Base class for action terms, mirroring IsaacLab's ``JointAction``.

    Each term owns the mapping from a slice of the raw policy action vector to a
    full-size :class:`TargetJointState`. Subclasses fill their specific target
    field (position / velocity / effort) at the joint indices they own and leave
    everything else zero. The agent aggregates all terms by summing their
    :meth:`to_target_joint_state` outputs.
    """

    #: Field of :class:`TargetJointState` this term writes. Override in subclass.
    target_field: str = "position"

    #: Joint indices (sim order) this term operates on. Set by the base constructor.
    joint_indices: np.ndarray

    def __init__(
        self,
        name: str,
        action_cfg: dict[str, Any],
        ros_node: RealNode,
        default_joint_pos: np.ndarray,
        p_gains: np.ndarray,
        d_gains: np.ndarray,
        default_joint_vel: np.ndarray | None = None,
        action_cursor: int = 0,
    ) -> None:
        self.name = name
        self.action_cfg = action_cfg
        self.ros_node = ros_node
        self._num_joints_total = int(ros_node.NUM_JOINTS)
        sim_joint_names = list(ros_node.sim_joint_names)

        default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        p_gains = np.asarray(p_gains, dtype=np.float32).reshape(-1)
        d_gains = np.asarray(d_gains, dtype=np.float32).reshape(-1)
        expected_shape = (self._num_joints_total,)
        for arr_name, arr in (
            ("default_joint_pos", default_joint_pos),
            ("p_gains", p_gains),
            ("d_gains", d_gains),
        ):
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{arr_name} shape mismatch for action term '{name}': "
                    f"expected {expected_shape}, got {arr.shape}"
                )
        self._default_joint_pos = default_joint_pos
        self._p_gains = p_gains
        self._d_gains = d_gains

        if default_joint_vel is not None:
            default_joint_vel = np.asarray(default_joint_vel, dtype=np.float32).reshape(-1)
            if default_joint_vel.shape != expected_shape:
                raise ValueError(
                    f"default_joint_vel shape mismatch for action term '{name}': "
                    f"expected {expected_shape}, got {default_joint_vel.shape}"
                )
        self._default_joint_vel = default_joint_vel

        joint_name_exprs = action_cfg.get("joint_names", None)
        if joint_name_exprs is None:
            raise ValueError(f"Action term '{name}' missing 'joint_names'")
        if not isinstance(joint_name_exprs, (list, tuple)):
            raise ValueError(f"Action term '{name}' has invalid joint_names: {joint_name_exprs!r}")
        (
            self.joint_indices,
            self._matched_joint_names,
            self._matched_joint_exprs,
        ) = _resolve_joint_matches(sim_joint_names, list(joint_name_exprs))
        if self.joint_indices.size == 0:
            raise ValueError(f"Action term '{name}' matched no joints")

        self._scale = _resolve_per_joint_values(
            value_cfg=action_cfg.get("scale", 1.0),
            joint_indices=self.joint_indices,
            matched_joint_names=self._matched_joint_names,
            matched_joint_exprs=self._matched_joint_exprs,
            num_joints_total=self._num_joints_total,
            default_value=1.0,
            value_name=f"{name}.scale",
        )
        self._offset = self._resolve_default_offset(
            cfg_offset=action_cfg.get("offset", 0.0),
        )

        self._action_slice = slice(
            action_cursor,
            action_cursor + int(self.joint_indices.shape[0]),
        )
        self._raw_action = np.zeros(self.action_dim, dtype=np.float32)
        self._processed_action = np.zeros_like(self._raw_action)

        # Pre-allocated template for _empty_full_size_arrays; five zero arrays
        # of shape (NUM_JOINTS,) shared across to_target_joint_state() calls
        # via copy-on-read to keep each TJS independent.
        self._empty_arrays_template: dict[str, np.ndarray] = {
            "position": np.zeros(self._num_joints_total, dtype=np.float32),
            "velocity": np.zeros(self._num_joints_total, dtype=np.float32),
            "effort": np.zeros(self._num_joints_total, dtype=np.float32),
            "kp": np.zeros(self._num_joints_total, dtype=np.float32),
            "kd": np.zeros(self._num_joints_total, dtype=np.float32),
        }

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return int(self.joint_indices.shape[0])

    @property
    def action_slice(self) -> slice:
        return self._action_slice

    @property
    def scale(self) -> np.ndarray:
        return self._scale

    @property
    def offset(self) -> np.ndarray:
        return self._offset

    @property
    def raw_action(self) -> np.ndarray:
        return self._raw_action

    @property
    def processed_action(self) -> np.ndarray:
        return self._processed_action

    """
    Operations.
    """

    def update(self, action: np.ndarray) -> None:
        """Store the raw action slice and apply the affine transform.

        ``action`` must have shape ``(action_dim,)`` and correspond to this term's
        ``action_slice`` of the full policy action vector.
        """
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape != (self.action_dim,):
            raise ValueError(
                f"Action slice shape mismatch for '{self.name}': "
                f"expected ({self.action_dim},), got {action_arr.shape}"
            )
        self._raw_action[:] = action_arr
        self._processed_action = self._raw_action * self._scale + self._offset

    @abstractmethod
    def to_target_joint_state(self) -> TargetJointState:
        """Pack this term's processed output into a full-size TargetJointState."""
        ...

    @abstractmethod
    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        """Decode a TargetJointState back into this term's raw action slice.

        Used so an agent can reconstruct a ``last_action`` observation from a
        previously sent command.
        """
        ...

    """
    Subclass hooks / helpers.
    """

    def _resolve_default_offset(self, cfg_offset: Any) -> np.ndarray:
        """Resolve the per-joint offset. Subclasses override to apply default semantics."""
        return _resolve_per_joint_values(
            value_cfg=cfg_offset,
            joint_indices=self.joint_indices,
            matched_joint_names=self._matched_joint_names,
            matched_joint_exprs=self._matched_joint_exprs,
            num_joints_total=self._num_joints_total,
            default_value=0.0,
            value_name=f"{self.name}.offset",
        )

    def _empty_full_size_arrays(self) -> dict[str, np.ndarray]:
        """Return a fresh copy of the pre-allocated zero-array template.

        Each call produces independent arrays so callers can mutate slices
        without aliasing across ``to_target_joint_state()`` invocations.
        """
        return {k: v.copy() for k, v in self._empty_arrays_template.items()}

    def _invert_affine(self, target_values: np.ndarray) -> np.ndarray:
        """Invert ``target = raw * scale + offset`` with a zero-scale guard."""
        target_arr = np.asarray(target_values, dtype=np.float32).reshape(-1)
        if target_arr.shape != self._scale.shape:
            raise ValueError(
                f"Inverse-affine shape mismatch for '{self.name}': "
                f"target={target_arr.shape}, scale={self._scale.shape}"
            )
        result = np.zeros_like(target_arr)
        non_zero = np.logical_not(np.isclose(self._scale, 0.0))
        result[non_zero] = (target_arr[non_zero] - self._offset[non_zero]) / self._scale[non_zero]
        zero_mask = np.logical_not(non_zero)
        if np.any(zero_mask):
            invalid = zero_mask & (~np.isclose(target_arr, self._offset))
            if np.any(invalid):
                bad_indices = self.joint_indices[invalid]
                raise ValueError(
                    f"Cannot invert affine for term '{self.name}': scale is zero "
                    f"while target differs from offset at joints {bad_indices.tolist()}"
                )
        return result


class JointPositionAction(ActionTermBase):
    """Joint position command; ``position = scale * raw + offset``.

    When ``action_cfg['use_default_offset']`` is True (default), the per-joint
    offset is set to ``default_joint_pos`` at the controlled indices, matching
    Isaac Lab's ``JointPositionAction.use_default_offset``. Controlled joints
    also receive ``kp``/``kd`` from the agent's configured PD gains so the
    aggregate :class:`TargetJointState` marks them as active.
    """

    target_field: str = "position"

    def _resolve_default_offset(self, cfg_offset: Any) -> np.ndarray:
        if bool(self.action_cfg.get("use_default_offset", True)):
            return self._default_joint_pos[self.joint_indices].copy()
        return super()._resolve_default_offset(cfg_offset)

    def to_target_joint_state(self) -> TargetJointState:
        fields = self._empty_full_size_arrays()
        fields["position"][self.joint_indices] = self._processed_action
        fields["kp"][self.joint_indices] = self._p_gains[self.joint_indices]
        fields["kd"][self.joint_indices] = self._d_gains[self.joint_indices]
        return TargetJointState(**fields)

    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        term_target = target_joint_state.position[self.joint_indices]
        self._raw_action[:] = self._invert_affine(term_target)
        self._processed_action = self._raw_action * self._scale + self._offset


class RelativeJointPositionAction(ActionTermBase):
    """Joint position command relative to the current joint positions.

    Final target is ``current_joint_pos + processed_action``. Use with care: if
    the robot state drifts between steps the effective command can accumulate.
    """

    target_field: str = "position"

    def _resolve_default_offset(self, cfg_offset: Any) -> np.ndarray:
        if bool(self.action_cfg.get("use_zero_offset", True)):
            return np.zeros(int(self.joint_indices.shape[0]), dtype=np.float32)
        return super()._resolve_default_offset(cfg_offset)

    def to_target_joint_state(self) -> TargetJointState:
        fields = self._empty_full_size_arrays()
        current_pos = np.asarray(self.ros_node.joint_pos_, dtype=np.float32)
        fields["position"][self.joint_indices] = current_pos[self.joint_indices] + self._processed_action
        fields["kp"][self.joint_indices] = self._p_gains[self.joint_indices]
        fields["kd"][self.joint_indices] = self._d_gains[self.joint_indices]
        return TargetJointState(**fields)

    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        current_pos = np.asarray(self.ros_node.joint_pos_, dtype=np.float32)
        delta = target_joint_state.position[self.joint_indices] - current_pos[self.joint_indices]
        self._raw_action[:] = self._invert_affine(delta)
        self._processed_action = self._raw_action * self._scale + self._offset


class JointVelocityAction(ActionTermBase):
    """Joint velocity command; writes ``velocity`` for controlled joints.

    When ``action_cfg['use_default_offset']`` is True, the per-joint offset is
    set to ``default_joint_vel`` at the controlled indices, mirroring Isaac
    Lab's ``JointVelocityAction.use_default_offset``. Controlled joints also
    receive ``kd`` from the agent's configured damping gains so the
    ``enable_mask`` still marks them as active.
    """

    target_field: str = "velocity"

    def _resolve_default_offset(self, cfg_offset: Any) -> np.ndarray:
        if bool(self.action_cfg.get("use_default_offset", False)):
            if self._default_joint_vel is None:
                raise ValueError(
                    f"Action term '{self.name}': use_default_offset=True but "
                    "default_joint_vel was not provided. "
                    "Pass default_joint_vel through parse_action_cfgs."
                )
            return self._default_joint_vel[self.joint_indices].copy()
        return super()._resolve_default_offset(cfg_offset)

    def to_target_joint_state(self) -> TargetJointState:
        fields = self._empty_full_size_arrays()
        fields["velocity"][self.joint_indices] = self._processed_action
        fields["kd"][self.joint_indices] = self._d_gains[self.joint_indices]
        return TargetJointState(**fields)

    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        term_target = target_joint_state.velocity[self.joint_indices]
        self._raw_action[:] = self._invert_affine(term_target)
        self._processed_action = self._raw_action * self._scale + self._offset


class JointEffortAction(ActionTermBase):
    """Joint effort (torque) command; writes ``effort``. ``kp``/``kd`` stay zero."""

    target_field: str = "effort"

    def to_target_joint_state(self) -> TargetJointState:
        fields = self._empty_full_size_arrays()
        fields["effort"][self.joint_indices] = self._processed_action
        return TargetJointState(**fields)

    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        term_target = target_joint_state.effort[self.joint_indices]
        self._raw_action[:] = self._invert_affine(term_target)
        self._processed_action = self._raw_action * self._scale + self._offset


class JointPositionToLimitsAction(ActionTermBase):
    """Joint position command rescaled to the joint limits.

    The post-affine value is clipped to ``[-1, 1]`` and (if
    ``rescale_to_limits`` is True, the default) mapped linearly to
    ``[joint_limits_low, joint_limits_high]`` read from ``ros_node``.
    """

    target_field: str = "position"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        joint_limits_low = np.asarray(self.ros_node.joint_limits_low, dtype=np.float32)
        joint_limits_high = np.asarray(self.ros_node.joint_limits_high, dtype=np.float32)
        self._joint_limits_low = joint_limits_low[self.joint_indices]
        self._joint_limits_high = joint_limits_high[self.joint_indices]
        self._rescale_to_limits = bool(self.action_cfg.get("rescale_to_limits", True))

    def _map_to_limits(self, clipped: np.ndarray) -> np.ndarray:
        if not self._rescale_to_limits:
            return clipped
        lo, hi = self._joint_limits_low, self._joint_limits_high
        return lo + 0.5 * (clipped + 1.0) * (hi - lo)

    def to_target_joint_state(self) -> TargetJointState:
        fields = self._empty_full_size_arrays()
        clipped = np.clip(self._processed_action, -1.0, 1.0)
        position = self._map_to_limits(clipped)
        fields["position"][self.joint_indices] = position
        fields["kp"][self.joint_indices] = self._p_gains[self.joint_indices]
        fields["kd"][self.joint_indices] = self._d_gains[self.joint_indices]
        return TargetJointState(**fields)

    def from_target_joint_state(self, target_joint_state: TargetJointState) -> None:
        position = target_joint_state.position[self.joint_indices]
        if self._rescale_to_limits:
            lo, hi = self._joint_limits_low, self._joint_limits_high
            span = np.where(np.isclose(hi, lo), 1.0, hi - lo)
            clipped = 2.0 * (position - lo) / span - 1.0
        else:
            clipped = position
        self._raw_action[:] = self._invert_affine(clipped)
        self._processed_action = self._raw_action * self._scale + self._offset


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    """Exponential-moving-average variant of :class:`JointPositionToLimitsAction`.

    ``alpha`` is the blend weight for the new processed action; 1.0 disables
    smoothing. ``alpha`` may be a scalar or a per-joint dict mapping regex to
    value (same resolution rules as ``scale``).
    """

    target_field: str = "position"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ema_alpha = _resolve_per_joint_values(
            value_cfg=self.action_cfg.get("alpha", 1.0),
            joint_indices=self.joint_indices,
            matched_joint_names=self._matched_joint_names,
            matched_joint_exprs=self._matched_joint_exprs,
            num_joints_total=self._num_joints_total,
            default_value=1.0,
            value_name=f"{self.name}.alpha",
        )
        self._prev_processed_action = np.zeros_like(self._processed_action)

    def update(self, action: np.ndarray) -> None:
        super().update(action)
        blended = self._ema_alpha * self._processed_action + (1.0 - self._ema_alpha) * self._prev_processed_action
        self._processed_action = blended.astype(np.float32)
        self._prev_processed_action = self._processed_action.copy()


"""
Configuration helpers.
"""


def _resolve_joint_matches(
    sim_joint_names: list[str],
    joint_name_exprs: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (indices, matched_names, matched_exprs) in sim-joint order; first regex wins."""
    joint_indices: list[int] = []
    matched_joint_names: list[str] = []
    matched_joint_exprs: list[str] = []
    for joint_idx, joint_name in enumerate(sim_joint_names):
        matched_expr = None
        for joint_name_expr in joint_name_exprs:
            if re.search(joint_name_expr, joint_name):
                matched_expr = joint_name_expr
                break
        if matched_expr is None:
            continue
        joint_indices.append(joint_idx)
        matched_joint_names.append(joint_name)
        matched_joint_exprs.append(matched_expr)
    return (
        np.asarray(joint_indices, dtype=np.int64),
        matched_joint_names,
        matched_joint_exprs,
    )


def _resolve_per_joint_values(
    value_cfg: Any,
    joint_indices: np.ndarray,
    matched_joint_names: list[str],
    matched_joint_exprs: list[str],
    num_joints_total: int,
    default_value: float,
    value_name: str,
) -> np.ndarray:
    """Resolve a per-joint scalar/dict/list config into a ``(term_dim,)`` array."""
    term_dim = int(joint_indices.shape[0])
    if value_cfg is None:
        return np.full(term_dim, default_value, dtype=np.float32)
    if isinstance(value_cfg, (float, int)):
        return np.full(term_dim, float(value_cfg), dtype=np.float32)
    if isinstance(value_cfg, dict):
        resolved = np.zeros(term_dim, dtype=np.float32)
        for i, (joint_name, joint_expr) in enumerate(zip(matched_joint_names, matched_joint_exprs)):
            resolved[i] = _resolve_dict_value(
                value_cfg=value_cfg,
                joint_name=joint_name,
                joint_expr=joint_expr,
                value_name=value_name,
            )
        return resolved
    if isinstance(value_cfg, (list, tuple, np.ndarray)):
        values = np.asarray(value_cfg, dtype=np.float32).reshape(-1)
        if values.shape[0] == term_dim:
            return values
        if values.shape[0] == num_joints_total:
            return values[joint_indices]
        raise ValueError(
            f"Invalid length for {value_name}: expected {term_dim} (term-local) or "
            f"{num_joints_total} (global), got {values.shape[0]}"
        )
    raise ValueError(f"Unsupported config type for {value_name}: {type(value_cfg)}")


def _resolve_dict_value(
    value_cfg: dict[str, Any],
    joint_name: str,
    joint_expr: str,
    value_name: str,
) -> float:
    """Resolve a dict-based per-joint value; longest matching regex wins on ties."""
    if joint_expr in value_cfg:
        return float(value_cfg[joint_expr])
    matched = [(p, v) for p, v in value_cfg.items() if re.search(p, joint_name)]
    if matched:
        matched.sort(key=lambda item: len(item[0]), reverse=True)
        return float(matched[0][1])
    raise ValueError(
        f"Unable to resolve {value_name} from dict for joint '{joint_name}'. "
        f"Available patterns: {list(value_cfg.keys())}"
    )


def _resolve_action_term_cls(name: str, cfg: dict[str, Any]) -> type[ActionTermBase]:
    """Resolve an :class:`ActionTermBase` subclass from a cfg ``class_type`` string.

    The cfg ``class_type`` string follows IsaacLab's convention, e.g.
    ``"isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"`` or just
    ``"JointPositionAction"``. Resolution is attempted in order:

    1. **`globals()` lookup** — the bare class name is matched against this
       module's namespace. This is the fast path for locally-defined subclasses.

    2. **`importlib` lookup** — when ``class_type`` contains a module path
       (``module.submodule:ClassName`` or ``module.submodule.ClassName``), the
       named module is imported and the class is retrieved from its namespace.
       This allows third-party / user-defined ``ActionTermBase`` subclasses in
       external packages to be resolved without polluting this module's imports.

    Raises ``ValueError`` if the ``class_type`` cannot be resolved by either
    strategy, or if ``class_type`` is missing / empty.
    """
    class_type = str(cfg.get("class_type", ""))
    if not class_type:
        raise ValueError(f"Action term '{name}': 'class_type' is required but was missing or empty.")
    bare_name = class_type.rsplit(":", 1)[-1].rsplit(".", 1)[-1].strip()

    # Strategy 1 — globals() lookup (fast path for local subclasses).
    cls = globals().get(bare_name)
    if isinstance(cls, type) and issubclass(cls, ActionTermBase) and cls is not ActionTermBase:
        return cls

    # Strategy 2 — importlib lookup for fully-qualified class_type strings
    # (e.g. "mypackage.actions:MyCustomAction").
    module_path, _, attr = class_type.partition(":")
    if not attr:
        # "a.b.ClassName" form — treat the last segment as the class name.
        parts = module_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr = parts
    if module_path and attr:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ValueError(
                f"Action term '{name}': could not import module '{module_path}' " f"from class_type '{class_type}'."
            ) from exc
        imported_cls = getattr(module, attr, None)
        if (
            isinstance(imported_cls, type)
            and issubclass(imported_cls, ActionTermBase)
            and imported_cls is not ActionTermBase
        ):
            return imported_cls

    raise ValueError(
        f"Action term '{name}': class_type '{class_type}' did not resolve to "
        f"a concrete ActionTermBase subclass. "
        f"Resolved bare name: '{bare_name}'. "
        f"Check for typos, missing imports, or verify the module is importable."
    )


def parse_action_cfgs(
    action_cfgs: dict[str, dict[str, Any]],
    ros_node: RealNode,
    default_joint_pos: np.ndarray,
    p_gains: np.ndarray,
    d_gains: np.ndarray,
    default_joint_vel: np.ndarray | None = None,
) -> list[ActionTermBase]:
    """Parse ``cfg['actions']`` into ordered :class:`ActionTermBase` instances.

    Layout rules:
    - term concatenation follows ``action_cfgs`` insertion order;
    - joints inside each term follow ``ros_node.sim_joint_names`` order;
    - within a term, the first matching regex in ``joint_names`` wins.

    The parser enforces that no two terms write to the same
    ``(target_field, joint_index)`` pair. This per-``(field, index)`` check
    intentionally allows two terms to control the **same joint** when they write
    **different** target fields (e.g. position vs. velocity). This is valid for
    a single agent whose model training may design for joint-multiplexed
    control. Aggregation uses :meth:`TargetJointState.merge` which resolves
    ``kp``/``kd`` overlaps by taking the element-wise maximum, so two terms
    targeting different value fields on the same joint compose correctly.
    """
    if action_cfgs is None:
        raise ValueError("action_cfgs cannot be None")
    terms: list[ActionTermBase] = []
    seen_writes: dict[tuple[str, int], str] = {}
    action_cursor = 0
    for name, cfg in action_cfgs.items():
        if cfg is None:
            continue
        if cfg.get("asset_name", "robot") != "robot":
            continue
        term_cls = _resolve_action_term_cls(name, cfg)
        term = term_cls(
            name=name,
            action_cfg=cfg,
            ros_node=ros_node,
            default_joint_pos=default_joint_pos,
            p_gains=p_gains,
            d_gains=d_gains,
            default_joint_vel=default_joint_vel,
            action_cursor=action_cursor,
        )
        for joint_idx in term.joint_indices.tolist():
            write_key = (term.target_field, int(joint_idx))
            if write_key in seen_writes:
                existing = seen_writes[write_key]
                raise ValueError(
                    f"Action term write conflict for field '{term.target_field}' "
                    f"and joint index {joint_idx}: both '{existing}' and '{name}' "
                    f"write to the same target."
                )
            seen_writes[write_key] = name
        terms.append(term)
        action_cursor += term.action_dim
    return terms


def get_policy_action_dim(action_terms: list[ActionTermBase]) -> int:
    """Total policy-action vector dimension implied by ``action_terms``."""
    return sum(term.action_dim for term in action_terms)


def build_default_uncontrolled_joint_action(
    ros_node: RealNode,
    default_joint_pos: np.ndarray,
    p_gains: np.ndarray,
    d_gains: np.ndarray,
    action_terms: list[ActionTermBase],
) -> ActionTermBase | None:
    """Build an agent-owned default action for joints untouched by ``action_terms``.

    The returned term is a :class:`JointPositionAction` configured as a constant
    hold-at-defaults emitter: ``scale = 0`` and ``use_default_offset = True``,
    so ``processed_action = 0 * raw + default_joint_pos = default_joint_pos``
    for every step. ``kp`` / ``kd`` come from the actuator config so the joints
    are properly marked as commanded in the aggregate ``enable_mask``.

    The term is created outside of ``cfg["actions"]`` and is stored separately
    on the agent (``OnboardAgent._default_uncontrolled_joint_action``). It is
    NOT added to ``self._action_terms`` and must not be, because it has
    ``action_dim > 0`` that would otherwise inflate the policy-action vector
    dimension. Its role is purely to be overlaid via
    :meth:`TargetJointState.fill_disabled_from` inside
    :func:`pack_policy_action_to_target_joint_state`.

    Returns ``None`` when every joint is already covered by an explicit action
    term (no fallback needed).

    Opting out: ``OnboardAgent`` exposes
    ``use_default_uncontrolled_joint_action=False`` to skip building this
    term entirely. Use that when an external process (e.g. arm teleop, a
    safety controller, another policy) is responsible for the joints outside
    the agent's explicit action terms; the aggregate ``TargetJointState``
    will then leave those joints uncommanded
    (``effort == kp == kd == 0``), so their ``enable_mask`` entries stay
    False and the ROS node will not overwrite the external owner's commands.
    The True branch is the IsaacLab-style default where the agent owns every
    joint.
    """
    num_joints_total = int(ros_node.NUM_JOINTS)
    sim_joint_names = list(ros_node.sim_joint_names)
    controlled: set[int] = set()
    for term in action_terms:
        controlled.update(int(j) for j in term.joint_indices.tolist())
    uncontrolled_indices = sorted(set(range(num_joints_total)) - controlled)
    if not uncontrolled_indices:
        return None
    # Anchor each regex with ^...$ so it matches only the exact joint name; the
    # base ActionTerm resolver iterates sim_joint_names and picks the first
    # regex that matches, which preserves sim-order for joint_indices.
    uncontrolled_name_exprs = [f"^{re.escape(sim_joint_names[i])}$" for i in uncontrolled_indices]
    term = JointPositionAction(
        name="default_uncontrolled_joint_action",
        action_cfg={
            "joint_names": uncontrolled_name_exprs,
            "scale": 0.0,
            "use_default_offset": True,
        },
        ros_node=ros_node,
        default_joint_pos=default_joint_pos,
        p_gains=p_gains,
        d_gains=d_gains,
        action_cursor=0,
    )
    # One-shot materialisation: with scale == 0 the processed action is
    # time-invariant and equals default_joint_pos[joint_indices]. Skipping
    # per-step update() is safe, but we must seed _processed_action once so
    # to_target_joint_state() emits the defaults rather than zeros.
    term.update(np.zeros(term.action_dim, dtype=np.float32))
    return term


def summarize_action_terms(action_terms: list[ActionTermBase], sim_joint_names: list[str]) -> str:
    """Render parsed action terms as a `prettytable` for the agent log.

    Returns the rendered table as a single multi-line string. One row per term;
    the joint column lists every joint name controlled by the term in the order
    of ``term.joint_indices``.
    """
    table = prettytable.PrettyTable()
    table.field_names = ["name", "target_field", "dim", "action_slice", "joints"]
    table.align = "l"
    for term in action_terms:
        joint_names = [sim_joint_names[joint_idx] for joint_idx in term.joint_indices.tolist()]
        action_slice_str = (
            f"[{term.action_slice.start}:{term.action_slice.stop}]"
            if isinstance(term.action_slice, slice)
            else str(term.action_slice)
        )
        table.add_row(
            [
                term.name,
                term.target_field,
                term.action_dim,
                action_slice_str,
                ", ".join(joint_names) if joint_names else "-",
            ]
        )
    return table.get_string()


def pack_policy_action_to_target_joint_state(
    policy_action: np.ndarray,
    action_terms: list[ActionTermBase],
    default_uncontrolled_joint_action: ActionTermBase | None = None,
) -> TargetJointState:
    """Update each term with its slice and combine their TargetJointState outputs.

    Explicit ``action_terms`` are aggregated via :meth:`TargetJointState.merge`,
    which allows the same joint to be targeted through **different** value
    fields (e.g. position + velocity) and resolves ``kp``/``kd`` overlaps by
    taking the element-wise maximum.

    ``default_uncontrolled_joint_action`` (if provided, typically built by
    :func:`build_default_uncontrolled_joint_action`) is composed last via
    :meth:`TargetJointState.fill_disabled_from` as a **fallback overlay**: it
    only writes to joints that the explicit aggregate left uncommanded. This
    keeps the fill-in semantically a fallback rather than a peer, so the
    contract that explicit terms be disjoint is not coupled to how the
    fill-in term happens to compute its coverage. ``update()`` is NOT called
    on ``default_uncontrolled_joint_action`` here; the factory is expected to
    have materialised a time-invariant ``_processed_action`` already.
    """
    action_values = np.asarray(policy_action, dtype=np.float32).reshape(-1)
    expected_dim = get_policy_action_dim(action_terms)
    if action_values.shape[0] != expected_dim:
        raise ValueError(f"Policy action dim mismatch: expected {expected_dim}, got {action_values.shape[0]}")
    aggregate: TargetJointState | None = None
    for term in action_terms:
        term.update(action_values[term.action_slice])
        term_tjs = term.to_target_joint_state()
        aggregate = term_tjs if aggregate is None else aggregate.merge(term_tjs)
    if default_uncontrolled_joint_action is not None:
        default_tjs = default_uncontrolled_joint_action.to_target_joint_state()
        if aggregate is None:
            aggregate = default_tjs
        else:
            aggregate = aggregate.fill_disabled_from(default_tjs)
    if aggregate is None:
        raise ValueError("No action terms provided to pack into a TargetJointState")
    return aggregate


def action_term_to_policy_action(
    target_joint_state: TargetJointState | None,
    action_terms: list[ActionTermBase],
) -> np.ndarray:
    """Back-solve a TargetJointState into a raw policy action vector.

    Each action term inverts its own affine mapping (and any additional
    transforms, e.g. joint-limits rescaling) to recover the ``raw_action``
    that would have been needed to produce the given ``target_joint_state``.

    Returns zeros when ``target_joint_state`` is None (no prior command).
    """
    policy_action = np.zeros(get_policy_action_dim(action_terms), dtype=np.float32)
    if target_joint_state is None:
        return policy_action
    for term in action_terms:
        term.from_target_joint_state(target_joint_state)
        policy_action[term.action_slice] = term.raw_action
    return policy_action
