"""Compatibility entry point for shared RL training.

New callers should import :mod:`uav_ac.rl.training`.
"""

from .. import training as _training
from ..training import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_training, name)


if __name__ == "__main__":
    _training.main()
