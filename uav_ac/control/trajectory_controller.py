"""Generic trajectory-tracking interface and scheduler.

The scheduler owns trajectory time/index progression and actuator application.
Concrete controllers own their control law.  A controller may use only the
current reference sample (e.g. cascaded control / RL) or request a future
horizon from :class:`TrajectoryReference` (e.g. MPC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from uav_ac.quadrotor.quad import Quad


@dataclass(frozen=True)
class ControlCommand:
    """Controller output consumed by the common quadrotor actuator interface."""

    thrust: float
    moment: np.ndarray

    def __post_init__(self) -> None:
        moment = np.asarray(self.moment, dtype=float)
        if moment.shape != (3,):
            raise ValueError("moment must have shape (3,)")
        if not np.isfinite(self.thrust) or not np.all(np.isfinite(moment)):
            raise ValueError("control command must contain only finite values")
        object.__setattr__(self, "thrust", float(self.thrust))
        object.__setattr__(self, "moment", moment)


@dataclass(frozen=True)
class TrajectoryReference:
    """Read-only view of the active trajectory sample and its future horizon.

    Trajectory rows follow the existing controller-ready convention:
    ``[x, y, z, vx, vy, vz, ax, ay, az, yaw, ...]``.
    Additional columns are preserved so future controllers may consume them.
    """

    trajectory: np.ndarray
    index: int
    dt: float
    is_terminal: bool = False
    is_control_tick: bool = True

    @property
    def current(self) -> np.ndarray:
        """Current trajectory sample."""
        return self.trajectory[self.index]

    @property
    def time(self) -> float:
        """Reference time corresponding to the current sample."""
        return self.index * self.dt

    def horizon(self, length: int) -> np.ndarray:
        """Return ``length`` samples starting at the current sample.

        The final sample is repeated when the requested horizon extends past
        the end of the trajectory.  This is convenient for fixed-horizon MPC
        and for RL policies that consume future waypoints.
        """
        if length < 1:
            raise ValueError("horizon length must be positive")

        indices = np.minimum(
            np.arange(self.index, self.index + length),
            len(self.trajectory) - 1,
        )
        return self.trajectory[indices]


@runtime_checkable
class TrajectoryTrackingController(Protocol):
    """Interface implemented by cascaded, MPC, RL, or other controllers.

    The interface deliberately exposes the complete trajectory through
    ``reference``.  Simple controllers can use ``reference.current`` while
    predictive controllers can use ``reference.horizon(N)``.
    """

    def reset(self) -> None:
        """Clear state accumulated across control cycles."""
        ...

    def step(self, quad: Quad, reference: TrajectoryReference) -> ControlCommand:
        """Compute one control command from the current vehicle/reference state."""
        ...


class TrajectoryController:
    """Generic trajectory scheduler for any ``TrajectoryTrackingController``.

    ``steps_per_reference`` is the number of MuJoCo/inner-loop steps for which
    one trajectory sample is active.  It is a stride, not a frequency in Hz.
    """

    def __init__(
            self,
            controller: TrajectoryTrackingController,
            quad: Quad,
            trajectory: np.ndarray,
            steps_per_reference: int,
    ):
        trajectory = np.asarray(trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 10 or len(trajectory) == 0:
            raise ValueError("trajectory must have shape (n, m) with n >= 1 and m >= 10")
        if not np.all(np.isfinite(trajectory)):
            raise ValueError("trajectory must contain only finite values")
        if steps_per_reference < 1:
            raise ValueError("steps_per_reference must be positive")

        self.controller = controller
        self.quad = quad
        self.trajectory = trajectory
        self.steps_per_reference = int(steps_per_reference)
        self.trajectory_dt = self.quad.dt * self.steps_per_reference

        self.trajectory_index = 0
        self.inner_step = 0

    def reset(self) -> None:
        """Restart trajectory tracking and reset the concrete controller."""
        self.controller.reset()
        self.trajectory_index = 0
        self.inner_step = 0

    def step(self) -> None:
        """Execute one simulation/control cycle."""
        self.trajectory_index = min(
            self.inner_step // self.steps_per_reference,
            len(self.trajectory) - 1,
        )
        reference = TrajectoryReference(
            trajectory=self.trajectory,
            index=self.trajectory_index,
            dt=self.trajectory_dt,
            is_terminal=self.trajectory_index == len(self.trajectory) - 1,
            is_control_tick=self.inner_step % self.steps_per_reference == 0,
        )

        command = self.controller.step(self.quad, reference)
        self.quad.set_propeller_speed(command.thrust, command.moment)
        self.inner_step += 1


__all__ = [
    "ControlCommand",
    "TrajectoryController",
    "TrajectoryReference",
    "TrajectoryTrackingController",
]
