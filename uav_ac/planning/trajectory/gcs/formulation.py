"""CVXPY formulations for GCS relaxation and fixed-path restriction."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy import sparse

from .bezier import derivative_energy_factor, endpoint_derivative_coefficients
from .config import GCSConfig
from .graph import GCSGraph
from .types import GCSRelaxation


@dataclass(frozen=True)
class RestrictionSolution:
    control_points: np.ndarray
    objective: float
    iterations: int
    status: str


def solve_relaxation(
    graph: GCSGraph,
    start: np.ndarray,
    goal: np.ndarray,
    config: GCSConfig,
) -> GCSRelaxation:
    """Solve the perspective SOCP relaxation of a Bezier GCS problem."""
    edge_count = len(graph.edges)
    degree = config.relaxation_degree
    point_count = degree + 1
    dimension = 3
    phi = cp.Variable(edge_count, nonneg=True, name="phi")
    coordinates = point_count * dimension
    tail = cp.Variable((edge_count, coordinates), name="tail")
    head = cp.Variable((edge_count, coordinates), name="head")
    tail_vector = cp.vec(tail, order="C")
    head_vector = cp.vec(head, order="C")
    incidence = _flow_incidence(graph)
    supply = np.zeros(len(graph.regions))
    supply[graph.source] = 1.0
    supply[graph.target] = -1.0
    spatial_tail, spatial_head = _spatial_conservation_operators(
        graph, coordinates)
    tail_membership, tail_bounds = _membership_operators(
        graph, point_count, dimension, use_head=False)
    head_membership, head_bounds = _membership_operators(
        graph, point_count, dimension, use_head=True)
    constraints: list[cp.Constraint] = [
        phi <= 1.0,
        incidence @ phi == supply,
        spatial_tail @ tail_vector == spatial_head @ head_vector,
        tail_membership @ tail_vector <= tail_bounds @ phi,
        head_membership @ head_vector <= head_bounds @ phi,
    ]

    continuity_tail, continuity_head = _edge_continuity_operators(
        edge_count, degree, min(config.continuity, degree - 1), dimension)
    constraints.append(
        continuity_tail @ tail_vector == continuity_head @ head_vector)
    for edge_index, (tail_vertex, head_vertex) in enumerate(graph.edges):
        tail_points = cp.reshape(tail[edge_index], (point_count, dimension), order="C")
        head_points = cp.reshape(head[edge_index], (point_count, dimension), order="C")
        if tail_vertex == graph.source:
            constraints.append(tail_points[0] == phi[edge_index] * start)
        if head_vertex == graph.target:
            constraints.append(head_points[-1] == phi[edge_index] * goal)

    differences = _control_difference_operator(edge_count, degree, dimension)
    difference_vectors = cp.reshape(
        differences @ tail_vector, (edge_count * degree, dimension), order="C")
    objective = config.length_weight * cp.sum(cp.norm(difference_vectors, axis=1))
    if config.derivative_weight > 0.0 and config.derivative_order <= degree:
        factor = derivative_energy_factor(degree, config.derivative_order)
        energy_terms = []
        for edge_index, (tail_vertex, head_vertex) in enumerate(graph.edges):
            tail_points = cp.reshape(
                tail[edge_index], (point_count, dimension), order="C")
            energy_terms.append(cp.quad_over_lin(
                cp.vec(factor @ tail_points, order="C"), phi[edge_index]))
            if head_vertex == graph.target:
                head_points = cp.reshape(
                    head[edge_index], (point_count, dimension), order="C")
                energy_terms.append(cp.quad_over_lin(
                    cp.vec(factor @ head_points, order="C"), phi[edge_index]))
        objective += config.derivative_weight * cp.sum(cp.hstack(energy_terms))
    if config.regularization > 0.0:
        objective += config.regularization * (
            cp.sum_squares(tail) + cp.sum_squares(head))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    _solve(problem, config)
    _require_solution(problem, "GCS relaxation")
    return GCSRelaxation(
        flows=np.asarray(phi.value).reshape(-1),
        tail_control_points=np.asarray(tail.value).reshape(
            edge_count, point_count, dimension),
        head_control_points=np.asarray(head.value).reshape(
            edge_count, point_count, dimension),
        objective=float(problem.value),
        iterations=_iterations(problem),
        status=str(problem.status),
    )


def solve_restriction(
    graph: GCSGraph,
    vertex_path: tuple[int, ...],
    start: np.ndarray,
    goal: np.ndarray,
    config: GCSConfig,
) -> RestrictionSolution:
    """Optimize Bezier segments after fixing the discrete GCS path."""
    segment_count = len(vertex_path)
    point_count = config.degree + 1
    control = cp.Variable((segment_count, point_count * 3), name="control")
    control_vector = cp.vec(control, order="C")
    segments = [
        cp.reshape(control[index], (point_count, 3), order="C")
        for index in range(segment_count)
    ]
    membership, bounds = _restriction_membership_operator(
        graph, vertex_path, point_count, 3)
    continuity = _restriction_continuity_operator(
        segment_count, config.degree, config.continuity, 3)
    constraints: list[cp.Constraint] = [
        membership @ control_vector <= bounds,
        continuity @ control_vector == 0.0,
        segments[0][0] == start,
        segments[-1][-1] == goal,
    ]
    for derivative in range(1, config.zero_endpoint_derivatives + 1):
        begin = endpoint_derivative_coefficients(
            config.degree, derivative, at_end=False)
        end = endpoint_derivative_coefficients(
            config.degree, derivative, at_end=True)
        constraints.extend((
            begin @ segments[0] == 0.0,
            end @ segments[-1] == 0.0,
        ))
    differences = _control_difference_operator(segment_count, config.degree, 3)
    difference_vectors = cp.reshape(
        differences @ control_vector, (segment_count * config.degree, 3), order="C")
    objective = config.length_weight * cp.sum(cp.norm(difference_vectors, axis=1))
    if config.derivative_weight > 0.0:
        factor = derivative_energy_factor(config.degree, config.derivative_order)
        objective += config.derivative_weight * sum(
            cp.sum_squares(factor @ segment) for segment in segments)
    if config.regularization > 0.0:
        objective += config.regularization * cp.sum_squares(control)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    _solve(problem, config)
    _require_solution(problem, "GCS convex restriction")
    return RestrictionSolution(
        np.asarray(control.value).reshape(segment_count, point_count, 3),
        float(problem.value), _iterations(problem),
        str(problem.status))


def _flow_incidence(graph: GCSGraph) -> sparse.csr_matrix:
    edge_count = len(graph.edges)
    rows = np.repeat(np.arange(edge_count), 2)
    columns = np.asarray(graph.edges, dtype=int).reshape(-1)
    values = np.tile((1.0, -1.0), edge_count)
    return sparse.coo_matrix(
        (values, (columns, rows)), shape=(len(graph.regions), edge_count)).tocsr()


def _spatial_conservation_operators(
    graph: GCSGraph, coordinates: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    vertices = [
        vertex for vertex in range(len(graph.regions))
        if vertex not in (graph.source, graph.target)
    ]
    row_of = {vertex: row for row, vertex in enumerate(vertices)}
    tail_rows: list[int] = []
    tail_columns: list[int] = []
    head_rows: list[int] = []
    head_columns: list[int] = []
    for edge, (tail, head) in enumerate(graph.edges):
        for coordinate in range(coordinates):
            if tail in row_of:
                tail_rows.append(row_of[tail] * coordinates + coordinate)
                tail_columns.append(edge * coordinates + coordinate)
            if head in row_of:
                head_rows.append(row_of[head] * coordinates + coordinate)
                head_columns.append(edge * coordinates + coordinate)
    shape = (len(vertices) * coordinates, len(graph.edges) * coordinates)
    tail_matrix = sparse.coo_matrix(
        (np.ones(len(tail_rows)), (tail_rows, tail_columns)), shape=shape).tocsr()
    head_matrix = sparse.coo_matrix(
        (np.ones(len(head_rows)), (head_rows, head_columns)), shape=shape).tocsr()
    return tail_matrix, head_matrix


def _membership_operators(
    graph: GCSGraph, point_count: int, dimension: int, *, use_head: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    coordinate_count = point_count * dimension
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    bound_rows: list[int] = []
    bound_columns: list[int] = []
    bound_values: list[float] = []
    row = 0
    for edge, vertices in enumerate(graph.edges):
        region = graph.regions[vertices[1 if use_head else 0]]
        for normal, offset in zip(region.A, region.b, strict=True):
            for point in range(point_count):
                for axis in range(dimension):
                    matrix_rows.append(row)
                    matrix_columns.append(edge * coordinate_count + point * dimension + axis)
                    matrix_values.append(normal[axis])
                bound_rows.append(row)
                bound_columns.append(edge)
                bound_values.append(offset)
                row += 1
    return (
        sparse.coo_matrix(
            (matrix_values, (matrix_rows, matrix_columns)),
            shape=(row, len(graph.edges) * coordinate_count)).tocsr(),
        sparse.coo_matrix(
            (bound_values, (bound_rows, bound_columns)),
            shape=(row, len(graph.edges))).tocsr(),
    )


def _edge_continuity_operators(
    edge_count: int, degree: int, continuity: int, dimension: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    tail_blocks = []
    head_blocks = []
    for derivative in range(continuity + 1):
        end = endpoint_derivative_coefficients(degree, derivative, at_end=True)
        begin = endpoint_derivative_coefficients(degree, derivative, at_end=False)
        tail_blocks.append(sparse.kron(end[None, :], sparse.eye(dimension)))
        head_blocks.append(sparse.kron(begin[None, :], sparse.eye(dimension)))
    tail_one = sparse.vstack(tail_blocks, format="csr")
    head_one = sparse.vstack(head_blocks, format="csr")
    return (
        sparse.kron(sparse.eye(edge_count), tail_one, format="csr"),
        sparse.kron(sparse.eye(edge_count), head_one, format="csr"),
    )


def _control_difference_operator(
    segment_count: int, degree: int, dimension: int,
) -> sparse.csr_matrix:
    difference = sparse.diags((-np.ones(degree), np.ones(degree)), (0, 1),
                              shape=(degree, degree + 1), format="csr")
    return sparse.kron(
        sparse.eye(segment_count), sparse.kron(difference, sparse.eye(dimension)),
        format="csr")


def _restriction_membership_operator(
    graph: GCSGraph, vertex_path: tuple[int, ...], point_count: int, dimension: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    blocks = []
    bounds = []
    for vertex in vertex_path:
        region = graph.regions[vertex]
        # kron(A, I) uses axis-major ordering; permute rows to plane-major.
        block = sparse.kron(
            sparse.eye(point_count), region.A, format="csr")
        permutation = np.arange(point_count * len(region.A)).reshape(
            point_count, len(region.A)).T.reshape(-1)
        blocks.append(block[permutation])
        bounds.append(np.repeat(region.b, point_count))
    return sparse.block_diag(blocks, format="csr"), np.concatenate(bounds)


def _restriction_continuity_operator(
    segment_count: int, degree: int, continuity: int, dimension: int,
) -> sparse.csr_matrix:
    if segment_count <= 1:
        return sparse.csr_matrix((0, segment_count * (degree + 1) * dimension))
    point_count = degree + 1
    coordinate_count = point_count * dimension
    rows = []
    for segment in range(segment_count - 1):
        for derivative in range(continuity + 1):
            end = endpoint_derivative_coefficients(degree, derivative, at_end=True)
            begin = endpoint_derivative_coefficients(degree, derivative, at_end=False)
            local = sparse.hstack((
                sparse.kron(end[None, :], sparse.eye(dimension)),
                -sparse.kron(begin[None, :], sparse.eye(dimension)),
            ), format="csr")
            left = segment * coordinate_count
            right = (segment_count - segment - 2) * coordinate_count
            rows.append(sparse.hstack((
                sparse.csr_matrix((dimension, left)), local,
                sparse.csr_matrix((dimension, right)),
            ), format="csr"))
    return sparse.vstack(rows, format="csr")


def _solve(problem: cp.Problem, config: GCSConfig) -> None:
    options: dict[str, float | int | bool] = {"verbose": config.verbose}
    if config.solver.upper() == "CLARABEL":
        options.update({
            "max_iter": config.max_iterations,
            "tol_gap_abs": config.feasibility_tolerance,
            "tol_gap_rel": config.feasibility_tolerance,
            "tol_feas": config.feasibility_tolerance,
        })
    problem.solve(solver=config.solver, **options)


def _require_solution(problem: cp.Problem, label: str) -> None:
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{label} failed with status {problem.status}")


def _iterations(problem: cp.Problem) -> int:
    iterations = problem.solver_stats.num_iters
    return 0 if iterations is None else int(iterations)


__all__ = ["RestrictionSolution", "solve_relaxation", "solve_restriction"]
