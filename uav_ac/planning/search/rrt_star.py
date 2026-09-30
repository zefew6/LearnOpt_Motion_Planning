"""Callback-driven finite-dimensional RRT* search."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import numpy as np


class RRTStar:
    """Rapidly-exploring Random Tree Star over a numeric state vector.

    Geometry and collision semantics are supplied by ``state_valid`` and
    ``edge_valid``. The defaults only enforce the configured box bounds.
    """

    def __init__(
            self, space_limits, start, goal, max_distance, max_iterations,
            *, state_valid: Callable | None = None,
            edge_valid: Callable | None = None,
            sampler: Callable | None = None,
            distance: Callable | None = None,
            rng: np.random.Generator | None = None,
    ):
        limits = np.asarray(space_limits, dtype=float)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if (limits.ndim != 2 or limits.shape[0] != 2 or limits.shape[1] < 1
                or not np.all(np.isfinite(limits)) or np.any(limits[1] <= limits[0])):
            raise ValueError("space_limits must have shape (2, dimensions) with ordered finite bounds")
        dimension = limits.shape[1]
        if (start.shape != (dimension,) or goal.shape != (dimension,)
                or not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal))
                or np.any(start < limits[0]) or np.any(start > limits[1])
                or np.any(goal < limits[0]) or np.any(goal > limits[1])):
            raise ValueError("start and goal must match the space_limits dimension and lie inside bounds")
        if (not np.isfinite(max_distance) or max_distance <= 0.
                or isinstance(max_iterations, bool) or int(max_iterations) < 1):
            raise ValueError("max_distance must be positive and max_iterations must be a positive integer")

        self.space_limits_lw = limits[0].copy()
        self.space_limits_up = limits[1].copy()
        self.dimension = dimension
        self.start = np.round(start, 2)
        self.goal = np.round(goal, 2)
        self.step_size = float(max_distance)
        self.max_iterations = int(max_iterations)
        self.rng = np.random.default_rng() if rng is None else rng
        self.state_valid = self._default_state_valid if state_valid is None else state_valid
        self.edge_valid = self._default_edge_valid if edge_valid is None else edge_valid
        self.sampler = self._default_sampler if sampler is None else sampler
        self.distance = self._default_distance if distance is None else distance
        self.epsilon = 0.15

        self.neighborhood_radius = 1.5 * self.step_size
        self.all_nodes = [self.start]
        self.tree = {}
        self.best_path = None
        self.best_tree = None
        self.dynamic_it_counter = 0
        self.dynamic_break_at = max(1., self.max_iterations / 10.)

        if not self._is_valid_state(self.start) or not self._is_valid_state(self.goal):
            raise ValueError("start and goal must satisfy state_valid")

    def run(self, verbose: bool = True):
        old_cost = np.inf
        for it in range(self.max_iterations):
            new_node = self._generate_random_node()
            nearest_node = self._find_nearest_node(new_node)
            new_node = self._adapt_random_node_position(new_node, nearest_node)
            neighbors = self._find_valid_neighbors(new_node)
            if len(neighbors) == 0:
                continue

            best_neighbor = self._find_best_neighbor(neighbors, new_node)
            self._update_tree(best_neighbor, new_node)
            has_rewired = self._rewire_safely(neighbors, new_node)

            if self._is_path_found(self.tree):
                path, cost = self.get_path(self.tree)
                if has_rewired and cost > old_cost:
                    raise RuntimeError("cost increased after rewiring")
                if cost < old_cost:
                    if verbose:
                        print(f"Iteration: {it} | Cost: {cost}")
                    self.store_best_tree()
                    old_cost = cost
                    self.dynamic_it_counter = 0
                else:
                    self.dynamic_it_counter += 1
                    if verbose:
                        print(
                            "\r Percentage to stop unless better path is found: "
                            f"{np.round(self.dynamic_it_counter / self.dynamic_break_at * 100, 2)}%",
                            end="\t")
                if self.dynamic_it_counter >= self.dynamic_break_at:
                    break

        if not self._is_path_found(self.best_tree):
            raise RuntimeError("no path found")
        self.best_path, cost = self.get_path(self.best_tree)
        if verbose:
            print(f"\nBest path found with cost: {cost}")

    def store_best_tree(self):
        self.best_tree = copy.deepcopy(self.tree)

    @staticmethod
    def path_cost(path):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or len(path) < 1:
            raise ValueError("path must be a non-empty two-dimensional array")
        return float(sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1)))

    def simplify_path(self, path: np.ndarray) -> np.ndarray:
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != self.dimension:
            raise ValueError("path must have shape (n, dimensions)")
        if len(path) <= 2:
            return path

        simplified_path = [path[0]]
        current_index = 0
        while current_index < len(path)-1:
            next_index = len(path)-1
            while next_index > current_index+1:
                if self._is_valid_connection(path[current_index], path[next_index]):
                    break
                next_index -= 1
            simplified_path.append(path[next_index])
            current_index = next_index
        return np.asarray(simplified_path)

    def _generate_random_node(self):
        if self.rng.uniform(0., 1.) < self.epsilon:
            return self.goal.copy()
        node = np.asarray(self.sampler(), dtype=float)
        if node.shape != (self.dimension,) or not np.all(np.isfinite(node)):
            raise ValueError("sampler must return a finite state vector")
        return np.round(node, 2)

    def _find_nearest_node(self, new_node):
        return min(self.all_nodes, key=lambda node: self._distance(node, new_node))

    def _adapt_random_node_position(self, new_node, nearest_node):
        distance_nearest = self._distance(new_node, nearest_node)
        if distance_nearest > self.step_size:
            new_node = nearest_node + (new_node-nearest_node)*self.step_size/distance_nearest
            new_node = np.round(new_node, 2)
        return new_node

    def _find_valid_neighbors(self, new_node):
        return [
            node for node in self.all_nodes
            if self._distance(node, new_node) <= self.neighborhood_radius
            and self._is_valid_connection(node, new_node)
        ]

    @staticmethod
    def _node_key(node: np.ndarray) -> str:
        return str(np.round(node, 2).tolist())

    def _cost_to_come(self, node: np.ndarray) -> float:
        cost = 0.0
        current = np.asarray(node, dtype=float)
        while not np.array_equal(current, self.start):
            parent = self.tree[self._node_key(current)]
            cost += self._distance(current, parent)
            current = parent
        return float(cost)

    def _find_best_neighbor(self, neighbors, new_node):
        return min(
            neighbors,
            key=lambda node: self._cost_to_come(node)+self._distance(node, new_node),
        )

    def _update_tree(self, node, new_node):
        node_key = self._node_key(new_node)
        node_parent = np.round(node, 2)
        if np.array_equal(node_parent, new_node):
            return
        if node_key in self.tree:
            current_cost = self._cost_to_come(new_node)
            candidate_cost = self._cost_to_come(node_parent)+self._distance(new_node, node_parent)
            if current_cost <= candidate_cost:
                return
        self.all_nodes.append(new_node)
        self.tree[node_key] = node_parent

    def _rewire_safely(self, neighbors, new_node):
        has_rewired = False
        new_node_cost = self._cost_to_come(new_node)
        for neighbor in neighbors:
            if np.array_equal(neighbor, self.start):
                continue
            if np.array_equal(neighbor, self.tree[self._node_key(new_node)]):
                continue
            current_cost = self._cost_to_come(neighbor)
            cost_through_new_node = new_node_cost+self._distance(neighbor, new_node)
            if cost_through_new_node < current_cost:
                self.tree[self._node_key(neighbor)] = np.round(new_node, 2)
                has_rewired = True
        return has_rewired

    def _is_valid_connection(self, node, new_node):
        return bool(self._is_valid_state(node) and self._is_valid_state(new_node)
                    and self.edge_valid(np.asarray(node), np.asarray(new_node)))

    def _is_valid_state(self, state):
        state = np.asarray(state, dtype=float)
        return state.shape == (self.dimension,) and bool(self.state_valid(state))

    def _default_state_valid(self, state):
        return bool(np.all(state >= self.space_limits_lw)
                    and np.all(state <= self.space_limits_up))

    @staticmethod
    def _default_edge_valid(first, second):
        return True

    def _default_sampler(self):
        return self.rng.uniform(self.space_limits_lw, self.space_limits_up)

    @staticmethod
    def _default_distance(first, second):
        return float(np.linalg.norm(np.asarray(first)-np.asarray(second)))

    def _distance(self, first, second):
        return float(self.distance(np.asarray(first), np.asarray(second)))

    def _is_path_found(self, tree):
        return tree is not None and self._node_key(self.goal) in tree

    def get_path(self, tree):
        if not self._is_path_found(tree):
            raise RuntimeError("tree does not contain the goal")
        path = [self.goal]
        node = self.goal
        started = time.perf_counter()
        while not np.array_equal(node, self.start):
            node = tree[self._node_key(node)]
            path.append(node)
            if time.perf_counter()-started > 5.:
                raise RuntimeError("path reconstruction exceeded five seconds")
        path = np.asarray(path[::-1]).reshape(-1, self.dimension)
        return path, self.path_cost(path)
