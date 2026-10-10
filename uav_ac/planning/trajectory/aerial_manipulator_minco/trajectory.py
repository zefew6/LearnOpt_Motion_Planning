"""Result types and controller-reference conversion for 8-D MINCO."""

from dataclasses import dataclass

import numpy as np

from ..gcopter.trajectory import evaluate_piecewise_quintic
from .flatness import recover_state


def gripper_gap_motion(start, end, elapsed, duration):
    a = float(np.clip(elapsed/duration, 0., 1.))
    shape = 10*a**3 - 15*a**4 + 6*a**5
    rate = (30*a**2 - 60*a**3 + 30*a**4)/duration
    acceleration = (60*a - 180*a**2 + 120*a**3)/duration**2
    return np.array([start+(end-start)*shape, (end-start)*rate, (end-start)*acceleration])


@dataclass(frozen=True)
class AerialManipulatorSearchResult:
    """Exact, continuously edge-validated 8-D RRT initial path."""

    path: np.ndarray
    metrics: dict
    exact_solution: bool = True

    def __post_init__(self):
        path = np.asarray(self.path, dtype=float).copy()
        if (path.ndim != 2 or path.shape[1] != 8 or len(path) < 2
                or not np.all(np.isfinite(path)) or not self.exact_solution):
            raise ValueError("search result must contain a finite exact 8-D path")
        path.setflags(write=False)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True)
class AerialManipulatorTrajectory:
    durations: np.ndarray
    coefficients: np.ndarray
    rrt_path: np.ndarray
    cost: float
    iterations: int
    optimizer_converged: bool
    optimizer_message: str
    validation_passed: bool | None = None
    minimum_clearance: float = float("nan")
    maximum_violation: float = float("inf")
    validation_sample_dt: float = float("nan")
    validation_performed: bool = False

    def __post_init__(self):
        if self.validation_passed is not None:
            object.__setattr__(self, "validation_performed", True)
        durations = np.asarray(self.durations, dtype=float).copy()
        coefficients = np.asarray(self.coefficients, dtype=float).copy()
        path = np.asarray(self.rrt_path, dtype=float).copy()
        if (durations.ndim != 1 or len(durations) < 1 or np.any(durations <= 0)
                or coefficients.shape != (len(durations), 6, 8)
                or path.ndim != 2 or path.shape[1] != 8
                or not np.all(np.isfinite(durations))
                or not np.all(np.isfinite(coefficients))
                or not np.all(np.isfinite(path))):
            raise ValueError("trajectory arrays must be finite and shaped correctly")
        for value in (durations, coefficients, path):
            value.setflags(write=False)
        object.__setattr__(self, "durations", durations)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "rrt_path", path)

    @property
    def total_time(self):
        return float(np.sum(self.durations))

    def evaluate(self, time, derivative=0):
        return evaluate_piecewise_quintic(
            self.durations, self.coefficients, time, derivative)

    def reference(self, time, robot, *, gripper_opening):
        """Convert flat-output state into a full NED/FRD robot reference."""
        sigma, velocity, acceleration, jerk = (
            self.evaluate(time, order) for order in range(4))
        q, v = recover_state(np.stack((sigma, velocity, acceleration, jerk)), gripper_opening)
        if np.dot(q[3:7], robot.configuration[3:7]) < 0.0:
            q[3:7] *= -1.
        a = np.r_[acceleration[:3], np.zeros(3), acceleration[4:8], 0.]
        from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
        return AerialManipulatorReference(q, v, a)


@dataclass(frozen=True)
class JointTaskTrajectory:
    """One planned task timeline; runtime confirmation may pause its clock."""

    pick: AerialManipulatorTrajectory
    place: AerialManipulatorTrajectory
    gap_open: float
    gap_closed: float
    dwell_time: float

    def __post_init__(self):
        if not np.isfinite(self.dwell_time) or self.dwell_time <= 0:
            raise ValueError('task dwell time must be positive and finite')

    @property
    def total_time(self):
        return self.pick.total_time + self.place.total_time + 2*self.dwell_time

    def gap_motion(self, event, elapsed):
        if event not in {'grasp', 'release'}:
            raise ValueError('gripper event must be grasp or release')
        start, end = ((self.gap_open, self.gap_closed) if event == 'grasp'
                      else (self.gap_closed, self.gap_open))
        return gripper_gap_motion(start, end, elapsed, self.dwell_time)

    def reference(self, time, robot):
        from uav_ac.robot.aerial_manipulator import AerialManipulatorReference
        t = float(np.clip(time, 0., self.total_time))
        if t < self.pick.total_time:
            return self.pick.reference(t, robot, gripper_opening=self.gap_open)
        if t < self.pick.total_time+self.dwell_time:
            trajectory, event, local = self.pick, 'grasp', t-self.pick.total_time
        elif t < self.pick.total_time+self.dwell_time+self.place.total_time:
            return self.place.reference(t-self.pick.total_time-self.dwell_time, robot,
                                        gripper_opening=self.gap_closed)
        else:
            trajectory, event = self.place, 'release'
            local = t-self.pick.total_time-self.dwell_time-self.place.total_time
        gap, rate, acceleration = self.gap_motion(event, local)
        ref = trajectory.reference(trajectory.total_time, robot, gripper_opening=gap)
        velocity, acc = ref.velocity.copy(), ref.acceleration.copy()
        velocity[-1], acc[-1] = rate, acceleration
        return AerialManipulatorReference(ref.configuration, velocity, acc)
