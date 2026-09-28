"""Callback-driven RRT-Connect for bounded configuration spaces."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np


class ExtendStatus(Enum):
    TRAPPED = auto()
    ADVANCED = auto()
    REACHED = auto()


@dataclass
class _Tree:
    states: list[np.ndarray]
    parents: list[int]
    rooted_at_start: bool


class RRTConnect:
    """Two-tree RRT-Connect with caller-owned metric and collision semantics."""

    def __init__(
            self, lower_bounds, upper_bounds, distance_fn, interpolate_fn,
            state_valid_fn, edge_valid_fn, step_size, *, max_iterations=4000,
            goal_bias=0.05, rng=None, distance_batch_fn=None,
            simplify_edge_valid_fn=None, sample_fn=None):
        self.lower = np.asarray(lower_bounds, dtype=float)
        self.upper = np.asarray(upper_bounds, dtype=float)
        if (self.lower.ndim != 1 or self.upper.shape != self.lower.shape
                or np.any(self.upper <= self.lower)):
            raise ValueError("bounds must be ordered vectors with equal dimensions")
        if step_size <= 0 or max_iterations <= 0 or not 0 <= goal_bias <= 1:
            raise ValueError("invalid RRT-Connect settings")
        self.distance_fn = distance_fn
        self.distance_batch_fn = distance_batch_fn
        self.interpolate_fn = interpolate_fn
        self.state_valid_fn = state_valid_fn
        self.edge_valid_fn = edge_valid_fn
        self.simplify_edge_valid_fn = (edge_valid_fn if simplify_edge_valid_fn is None
                                       else simplify_edge_valid_fn)
        self.sample_fn = sample_fn
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.goal_bias = float(goal_bias)
        self.rng = np.random.default_rng() if rng is None else rng
        self.last_iterations = 0
        self.last_node_count = 0

    def plan(self, start, goal) -> np.ndarray:
        start, goal = self._state(start), self._state(goal)
        if not self.state_valid_fn(start) or not self.state_valid_fn(goal):
            self.last_iterations = 0
            self.last_node_count = 2
            raise ValueError("RRT start and goal must be valid states")
        if self.edge_valid_fn(start, goal):
            return np.vstack((start, goal))
        first = _Tree([start.copy()], [-1], True)
        second = _Tree([goal.copy()], [-1], False)
        self.last_iterations = 0
        self.last_node_count = 2
        for iteration in range(self.max_iterations):
            self.last_iterations = iteration+1
            if self.rng.random() < self.goal_bias:
                target = goal
            elif self.sample_fn is None:
                target = self.rng.uniform(self.lower, self.upper)
            else:
                target = self._state(self.sample_fn())
            status, new_index = self._extend(first, target)
            if status is not ExtendStatus.TRAPPED:
                connected, other_index = self._connect(second, first.states[new_index])
                if connected:
                    self.last_node_count = len(first.states)+len(second.states)
                    return self._join(first, new_index, second, other_index)
            first, second = second, first
            self.last_node_count = len(first.states)+len(second.states)
        raise RuntimeError(f"RRT-Connect failed after {self.max_iterations} iterations")

    def simplify(self, path: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != len(self.lower) or len(path) < 2:
            raise ValueError("path must contain at least two states")
        kept = [path[0]]
        index = 0
        while index < len(path)-1:
            candidate = len(path)-1
            while (candidate > index+1
                   and not self.simplify_edge_valid_fn(path[index], path[candidate])):
                candidate -= 1
            if not self.simplify_edge_valid_fn(path[index], path[candidate]):
                raise ValueError("input path contains an invalid edge")
            kept.append(path[candidate])
            index = candidate
        return np.asarray(kept)

    def _state(self, value):
        state = np.asarray(value, dtype=float)
        if state.shape != self.lower.shape or not np.all(np.isfinite(state)):
            raise ValueError("RRT state has invalid shape or non-finite values")
        return state.copy()

    def _extend(self, tree: _Tree, target: np.ndarray):
        if self.distance_batch_fn is None:
            nearest = min(range(len(tree.states)),
                          key=lambda i: self.distance_fn(tree.states[i], target))
        else:
            nearest = int(np.argmin(self.distance_batch_fn(np.asarray(tree.states), target)))
        source = tree.states[nearest]
        distance = float(self.distance_fn(source, target))
        if distance <= 1.0e-12:
            return ExtendStatus.REACHED, nearest
        candidate = self.interpolate_fn(source, target, min(1.0, self.step_size/distance))
        candidate = self._state(candidate)
        if (not self.state_valid_fn(candidate)
                or not self.edge_valid_fn(source, candidate)):
            return ExtendStatus.TRAPPED, nearest
        tree.states.append(candidate)
        tree.parents.append(nearest)
        index = len(tree.states)-1
        reached = float(self.distance_fn(candidate, target)) <= 1.0e-9
        return (ExtendStatus.REACHED if reached else ExtendStatus.ADVANCED), index

    def _connect(self, tree: _Tree, target: np.ndarray):
        while True:
            status, index = self._extend(tree, target)
            if status is ExtendStatus.TRAPPED:
                return False, index
            if status is ExtendStatus.REACHED:
                return True, index

    @staticmethod
    def _root_path(tree: _Tree, index: int) -> list[np.ndarray]:
        result = []
        while index >= 0:
            result.append(tree.states[index])
            index = tree.parents[index]
        return result[::-1]

    def _join(self, left: _Tree, left_index: int, right: _Tree, right_index: int):
        left_path = self._root_path(left, left_index)
        right_path = self._root_path(right, right_index)
        if left.rooted_at_start:
            return np.asarray(left_path + right_path[-2::-1])
        return np.asarray(right_path + left_path[-2::-1])


__all__ = ["ExtendStatus", "RRTConnect"]
