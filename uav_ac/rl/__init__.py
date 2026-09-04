"""Reinforcement-learning trajectory tracking utilities.

Importing :mod:`uav_ac` does not import this package, so Gymnasium and SB3 stay
optional for conventional planning and control workflows.
"""

__all__ = ["MujocoTrajectoryTrackingEnv"]


def __getattr__(name: str):
    if name == "MujocoTrajectoryTrackingEnv":
        from .environment import MujocoTrajectoryTrackingEnv

        return MujocoTrajectoryTrackingEnv
    raise AttributeError(name)
