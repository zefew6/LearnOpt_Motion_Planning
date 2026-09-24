"""Public state, command and model-query interfaces for the aerial manipulator.

Configurations use NED position, scalar-first FRD-to-NED quaternion, four arm
angles, and gripper inner-face gap (12 scalars). Tangent velocities use NED
linear velocity, body FRD angular velocity, four joint rates, and gap rate
(11 scalars). Tool twists and Jacobians are expressed in the NED world frame.
"""

from dataclasses import dataclass
import numpy as np

from .model import AerialManipulatorModel, ConfigurationLimits

GRIPPER_MIN_OPENING = 0.020
GRIPPER_MAX_OPENING = 0.070
CONFIGURATION_SIZE = 12
VELOCITY_SIZE = 11


class AerialManipulator:
    """Live robot handle with isolated numerical planning queries."""
    def __init__(self, simulation):
        if not simulation._has_arm:
            raise ValueError("simulation does not contain an aerial manipulator")
        self._simulation = simulation
        self._model = AerialManipulatorModel(simulation.model, lambda: simulation.data)

    def rebind(self, model, data_getter):
        """Refresh model addresses after a topology-preserving scene rebuild."""
        self._model = AerialManipulatorModel(model, data_getter)

    @property
    def state(self) -> "AerialManipulatorState":
        return self._model.state()

    @property
    def configuration(self) -> np.ndarray:
        return self._model.configuration()

    @property
    def velocity(self) -> np.ndarray:
        return self._model.velocity()

    @property
    def limits(self) -> ConfigurationLimits:
        return self._model.limits

    @property
    def mass(self) -> float:
        return self._model.mass_properties()["mass"]

    @property
    def center_of_mass(self) -> np.ndarray:
        return self._model.mass_properties()["center_of_mass"].copy()

    @property
    def end_effector_pose(self):
        return self.forward_kinematics()

    @property
    def geometric_jacobian(self) -> np.ndarray:
        return self.jacobian()

    @property
    def gripper_opening(self) -> float:
        return self.state.gripper_opening

    @property
    def gripper_sync_error(self) -> float:
        return self.state.gripper_sync_error

    def forward_kinematics(self, configuration=None, frame="tool"):
        return self._model.forward_kinematics(configuration, frame)

    def jacobian(self, configuration=None, frame="tool") -> np.ndarray:
        return self._model.jacobian(configuration, frame)

    def mass_properties(self, configuration=None) -> dict:
        return self._model.mass_properties(configuration)

    def dynamics(self, configuration=None, velocity=None) -> dict:
        return self._model.dynamics(configuration, velocity)

    def check_collision(self, configuration=None, clearance=0.0) -> dict:
        return self._model.check_collision(configuration, clearance)

    def integrate(self, configuration, tangent_delta) -> np.ndarray:
        return self._model.integrate(configuration, tangent_delta)

    def difference(self, configuration_from, configuration_to) -> np.ndarray:
        return self._model.difference(configuration_from, configuration_to)

    def apply(self, command: "AerialManipulatorCommand") -> None:
        self._simulation.apply_robot_command(command)

    def in_collision(self, joint_positions) -> bool:
        """Legacy convenience wrapper; prefer check_collision(full_configuration)."""
        return self._simulation.configuration_collision(joint_positions)

    def reset(self, configuration=None, velocity=None, motor_speeds=None):
        return self._simulation.reset_robot(configuration, velocity, motor_speeds)


@dataclass(frozen=True)
class AerialManipulatorState:
    base_state: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    gripper_opening: float
    gripper_velocity: float
    gripper_sync_error: float

    @property
    def configuration(self) -> np.ndarray:
        return np.r_[self.base_state[:7], self.joint_positions, self.gripper_opening]

    @property
    def velocity(self) -> np.ndarray:
        return np.r_[self.base_state[7:], self.joint_velocities, self.gripper_velocity]


@dataclass(frozen=True)
class AerialManipulatorCommand:
    """Collective thrust [N], FRD body moment [Nm], arm torques [Nm], gap [m]."""
    thrust: float
    moment: np.ndarray
    joint_torques: np.ndarray
    gripper_opening: float = 0.060

    def __post_init__(self):
        moment = np.asarray(self.moment, dtype=float)
        torques = np.asarray(self.joint_torques, dtype=float)
        opening = float(self.gripper_opening)
        if moment.shape != (3,) or torques.shape != (4,):
            raise ValueError("moment and joint_torques must have shapes (3,) and (4,)")
        if (not np.isfinite(self.thrust) or not np.all(np.isfinite(moment))
                or not np.all(np.isfinite(torques)) or not np.isfinite(opening)):
            raise ValueError("command values must be finite")
        if not GRIPPER_MIN_OPENING <= opening <= GRIPPER_MAX_OPENING:
            raise ValueError(f"gripper_opening must be in [{GRIPPER_MIN_OPENING}, {GRIPPER_MAX_OPENING}] m")
        object.__setattr__(self, "thrust", float(self.thrust))
        object.__setattr__(self, "moment", moment.copy())
        object.__setattr__(self, "joint_torques", torques.copy())
        object.__setattr__(self, "gripper_opening", opening)


@dataclass(frozen=True)
class AerialManipulatorReference:
    configuration: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    def __post_init__(self):
        for name, size in (("configuration", 12), ("velocity", 11), ("acceleration", 11)):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain {size} finite values")
            if name == "configuration":
                norm = np.linalg.norm(value[3:7])
                if norm < 1e-12:
                    raise ValueError("configuration quaternion cannot be zero")
                value = value.copy()
                value[3:7] /= norm
            object.__setattr__(self, name, value.copy())


__all__ = [
    "AerialManipulator", "AerialManipulatorCommand", "AerialManipulatorReference",
    "AerialManipulatorState", "AerialManipulatorModel", "ConfigurationLimits",
    "CONFIGURATION_SIZE", "VELOCITY_SIZE", "GRIPPER_MAX_OPENING", "GRIPPER_MIN_OPENING",
]
