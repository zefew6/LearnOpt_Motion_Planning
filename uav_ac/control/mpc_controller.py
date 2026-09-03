"""13-state quaternion NMPC for the MuJoCo quadrotor.

The controller implements the common trajectory-tracking interface used by
``TrajectoryController``:

    reset() -> None
    step(quad, reference) -> ControlCommand

State convention (matching ``Quad.X``):
    x = [p_W(3), q_WB(4, scalar-first), v_W(3), omega_B(3)]

Control convention (matching ``Quad.set_propeller_speed``):
    u = [collective_thrust, tau_x, tau_y, tau_z]

The prediction model is the rigid-body model only.  Rotor allocation and the
first-order motor lag remain in ``Quad`` / MuJoCo, so they are plant dynamics
not yet represented inside this first MPC baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .trajectory_controller import ControlCommand, TrajectoryReference


def _load_acados():
    """Import optional NMPC dependencies only when MPC is instantiated.

    Keeping the imports lazy means the cascaded controller can still be used on
    machines where CasADi/acados are not installed.
    """
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise ImportError(
            "MPCController requires CasADi and acados_template. Install acados "
            "and its Python interface before selecting the MPC controller."
        ) from exc
    return ca, AcadosModel, AcadosOcp, AcadosOcpSolver


@dataclass(frozen=True)
class MPCConfig:
    """Configuration for the first rigid-body NMPC baseline."""

    dt: float
    horizon_steps: int = 20

    # Tracking weights.  Attitude is represented by the 9 entries of R(q), so
    # ``attitude_weight`` is applied to each matrix element.
    position_weights: tuple[float, float, float] = (40.0, 40.0, 60.0)
    velocity_weights: tuple[float, float, float] = (6.0, 6.0, 8.0)
    attitude_weight: float = 6.0
    body_rate_weights: tuple[float, float, float] = (0.25, 0.25, 0.15)

    # Input regularization around feed-forward thrust and zero moment.
    thrust_weight: float = 0.08
    moment_weights: tuple[float, float, float] = (0.8, 0.8, 0.4)
    terminal_scale: float = 4.0

    # Solver settings.
    nlp_solver_type: str = "SQP_RTI"  # change to "SQP" for a more conservative debug baseline
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    integrator_type: str = "ERK"
    print_level: int = 0
    levenberg_marquardt: float = 1.0e-5

    # If the solver fails, hold the previous command rather than crashing the
    # MuJoCo loop.  ``raise_on_failure=True`` is useful while debugging.
    raise_on_failure: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("MPC dt must be positive")
        if self.horizon_steps < 2:
            raise ValueError("MPC horizon_steps must be at least 2")
        if self.terminal_scale <= 0.0:
            raise ValueError("terminal_scale must be positive")
        if self.nlp_solver_type not in {"SQP", "SQP_RTI"}:
            raise ValueError("nlp_solver_type must be 'SQP' or 'SQP_RTI'")


class MPCController:
    """Quaternion rigid-body nonlinear MPC with wrench input.

    The controller solves once per *reference sample*.  ``TrajectoryController``
    may call ``step`` at the MuJoCo physics rate; while the reference index is
    unchanged this class holds the previous MPC command.  Therefore, with the
    current scheduler, the MPC update period must equal ``reference.dt``.
    """

    NX = 13
    NU = 4
    _CACHE_VERSION = 1

    @staticmethod
    def _generated_code_directory() -> Path:
        """Return the repository-local directory used for acados artifacts."""
        return Path(__file__).resolve().parents[2] / "c_generated_code"

    def __init__(self, quad, config: MPCConfig):
        self.config = config
        self._quad_parameters = self._extract_quad_parameters(quad)
        self._solver = self._build_solver()

        self._last_reference_index: int | None = None
        self._last_command = self._hover_command()
        self._has_solution = False

        self.last_status: int | None = None
        self.last_solve_time: float = np.nan

    def reset(self) -> None:
        """Forget controller scheduling state and the previous applied command."""
        self._last_reference_index = None
        self._last_command = self._hover_command()
        self._has_solution = False
        self.last_status = None
        self.last_solve_time = np.nan

    def step(self, quad, reference: TrajectoryReference) -> ControlCommand:
        """Compute/hold one MPC wrench command."""
        if not np.isclose(reference.dt, self.config.dt, rtol=1.0e-6, atol=1.0e-9):
            raise ValueError(
                "Current TrajectoryReference.dt does not match MPCConfig.dt. "
                "With the current scheduler, MPC is solved once per reference sample."
            )

        # MuJoCo runs faster than the MPC. Hold u_0 with zero-order hold until
        # the next trajectory/reference sample becomes active.
        if (reference.index == self._last_reference_index
                and not (reference.is_terminal and reference.is_control_tick)):
            return self._last_command

        x0 = self._normalized_state(quad.X)
        refs = reference.horizon(self.config.horizon_steps + 1)
        rotations, body_rates, feedforward_thrusts = self._reference_quantities(refs)

        if self._has_solution:
            self._shift_warm_start()
        else:
            self._initialize_guess(x0)

        # Measured state equality constraint at the first shooting node.
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)
        self._solver.set(0, "x", x0)

        # Stage references.
        for stage in range(self.config.horizon_steps):
            yref = self._stage_yref(
                refs[stage],
                rotations[stage],
                body_rates[stage],
                feedforward_thrusts[stage],
            )
            self._solver.set(stage, "yref", yref)

        terminal_yref = self._terminal_yref(
            refs[-1], rotations[-1], body_rates[-1]
        )
        self._solver.set(self.config.horizon_steps, "yref", terminal_yref)

        status = int(self._solver.solve())
        self.last_status = status
        try:
            self.last_solve_time = float(
                np.asarray(self._solver.get_stats("time_tot")).squeeze()
            )
        except Exception:  # pragma: no cover - version-dependent diagnostics only
            self.last_solve_time = np.nan

        if status != 0:
            if self.config.raise_on_failure:
                self._solver.print_statistics()
                raise RuntimeError(f"acados NMPC failed with status {status}")
            # Keep the last feasible/known command. On the first failure this is
            # simply hover thrust and zero moment.
            self._last_reference_index = reference.index
            return self._last_command

        control = np.asarray(self._solver.get(0, "u"), dtype=float).reshape(self.NU)
        command = ControlCommand(
            thrust=float(control[0]),
            moment=control[1:4],
        )

        self._last_command = command
        self._last_reference_index = reference.index
        self._has_solution = True
        return command

    # ------------------------------------------------------------------
    # acados model / OCP
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_quad_parameters(quad) -> dict[str, float | np.ndarray]:
        inertia = np.array([quad.i_x, quad.i_y, quad.i_z], dtype=float)
        if np.any(inertia <= 0.0):
            raise ValueError("quadrotor inertia must be positive")
        if quad.m <= 0.0 or quad.l <= 0.0 or quad.kappa <= 0.0:
            raise ValueError("mass, arm length and drag_to_thrust must be positive")
        if quad.max_thrust <= quad.min_thrust:
            raise ValueError("max_thrust must exceed min_thrust")

        return {
            "g": float(quad.g),
            "m": float(quad.m),
            "inertia": inertia,
            "l": float(quad.l),
            "kappa": float(quad.kappa),
            "min_thrust": float(quad.min_thrust),
            "max_thrust": float(quad.max_thrust),
        }

    def _cache_signature(self, model_name: str) -> str:
        """Return a signature for all data that affects generated solver code."""
        parameters = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in self._quad_parameters.items()
        }
        payload = {
            "cache_version": self._CACHE_VERSION,
            "model_name": model_name,
            "code_export_directory": str(self._generated_code_directory()),
            "config": asdict(self.config),
            "quad_parameters": parameters,
            # Invalidate the generated solver when this model implementation
            # changes, even if the public configuration stays the same.
            "controller_source": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _cache_is_valid(
            json_file: Path, code_export_directory: Path,
            signature_file: Path, signature: str, model_name: str,
    ) -> bool:
        """Check that generated code belongs to this exact MPC definition."""
        solver_pattern = f"libacados_ocp_solver_{model_name}.*"
        try:
            stored_signature = signature_file.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        return (
            stored_signature == signature
            and json_file.is_file()
            and any(path.is_file() for path in code_export_directory.glob(solver_pattern))
        )

    def _build_solver(self):
        ca, AcadosModel, AcadosOcp, AcadosOcpSolver = _load_acados()

        p_cfg = self._quad_parameters
        g = p_cfg["g"]
        mass = p_cfg["m"]
        inertia = np.asarray(p_cfg["inertia"], dtype=float)

        model = AcadosModel()
        model.name = "quadrotor_wrench_nmpc"

        x = ca.SX.sym("x", self.NX)
        xdot = ca.SX.sym("xdot", self.NX)
        u = ca.SX.sym("u", self.NU)

        position = x[0:3]
        quat = x[3:7]
        velocity = x[7:10]
        omega = x[10:13]

        thrust = u[0]
        moment = u[1:4]

        rotation = self._casadi_rotation(ca, quat)
        e3 = ca.DM([0.0, 0.0, 1.0])

        position_dot = velocity
        quat_dot = self._casadi_quaternion_derivative(ca, quat, omega)
        velocity_dot = g * e3 - (thrust / mass) * (rotation @ e3)

        inertia_matrix = ca.DM(np.diag(inertia))
        inverse_inertia = ca.DM(np.diag(1.0 / inertia))
        omega_dot = inverse_inertia @ (
            moment - ca.cross(omega, inertia_matrix @ omega)
        )

        f_expl = ca.vertcat(position_dot, quat_dot, velocity_dot, omega_dot)

        model.x = x
        model.xdot = xdot
        model.u = u
        model.f_expl_expr = f_expl
        model.f_impl_expr = xdot - f_expl

        # NONLINEAR_LS output.  Tracking R(q) instead of q avoids the q/-q
        # double-cover ambiguity while keeping quaternion dynamics in the state.
        rotation_vector = ca.reshape(rotation, 9, 1)
        model.cost_y_expr = ca.vertcat(
            position,
            velocity,
            rotation_vector,
            omega,
            u,
        )
        model.cost_y_expr_e = ca.vertcat(
            position,
            velocity,
            rotation_vector,
            omega,
        )

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = self.config.horizon_steps
        ocp.solver_options.tf = self.config.horizon_steps * self.config.dt

        stage_weights = np.concatenate((
            np.asarray(self.config.position_weights, dtype=float),
            np.asarray(self.config.velocity_weights, dtype=float),
            np.full(9, self.config.attitude_weight, dtype=float),
            np.asarray(self.config.body_rate_weights, dtype=float),
            np.array([self.config.thrust_weight], dtype=float),
            np.asarray(self.config.moment_weights, dtype=float),
        ))
        terminal_weights = self.config.terminal_scale * stage_weights[:-self.NU]

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.diag(stage_weights)
        ocp.cost.W_e = np.diag(terminal_weights)
        ocp.cost.yref = np.zeros(stage_weights.size)
        ocp.cost.yref_e = np.zeros(terminal_weights.size)

        # Exact rotor-feasibility constraints, using the same mixer as Quad:
        #   f = 1/4 M [tau_x/l, tau_y/l, -tau_z/kappa, T]^T
        # and u = [T, tau_x, tau_y, tau_z].
        rotor_map = self._rotor_force_map()
        ocp.constraints.C = np.zeros((4, self.NX))
        ocp.constraints.D = rotor_map
        ocp.constraints.lg = np.full(4, p_cfg["min_thrust"], dtype=float)
        ocp.constraints.ug = np.full(4, p_cfg["max_thrust"], dtype=float)

        # acados needs an initial x0 at code-generation time; it is overwritten
        # on every closed-loop solve with the measured MuJoCo state.
        x0 = np.zeros(self.NX)
        x0[3] = 1.0
        ocp.constraints.x0 = x0

        ocp.solver_options.qp_solver = self.config.qp_solver
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = self.config.integrator_type
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        ocp.solver_options.nlp_solver_type = self.config.nlp_solver_type
        ocp.solver_options.levenberg_marquardt = self.config.levenberg_marquardt
        ocp.solver_options.print_level = self.config.print_level

        model_name = model.name
        code_export_directory = self._generated_code_directory()
        json_file = code_export_directory.parent / f"{model_name}_ocp.json"
        signature_file = code_export_directory / f".{model_name}.signature"
        signature = self._cache_signature(model_name)

        ocp.code_gen_opts.code_export_directory = str(code_export_directory)
        ocp.code_gen_opts.json_file = str(json_file)
        cache_valid = self._cache_is_valid(
            json_file, code_export_directory, signature_file,
            signature, model_name,
        )
        solver = AcadosOcpSolver(
            ocp,
            json_file=str(json_file),
            build=not cache_valid,
            generate=not cache_valid,
        )
        if not cache_valid:
            code_export_directory.mkdir(parents=True, exist_ok=True)
            signature_file.write_text(signature + "\n", encoding="utf-8")
        return solver

    def _rotor_force_map(self) -> np.ndarray:
        l = float(self._quad_parameters["l"])
        kappa = float(self._quad_parameters["kappa"])

        mixer = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0],
        ])
        # Maps u=[T,tau_x,tau_y,tau_z] to
        # [tau_x/l, tau_y/l, -tau_z/kappa, T].
        wrench_reorder = np.array([
            [0.0, 1.0 / l, 0.0, 0.0],
            [0.0, 0.0, 1.0 / l, 0.0],
            [0.0, 0.0, 0.0, -1.0 / kappa],
            [1.0, 0.0, 0.0, 0.0],
        ])
        return 0.25 * mixer @ wrench_reorder

    # ------------------------------------------------------------------
    # Reference construction
    # ------------------------------------------------------------------

    def _reference_quantities(
            self, refs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotations = np.stack([
            self._desired_rotation(sample[6:9], sample[9])
            for sample in refs
        ])

        body_rates = np.zeros((len(refs), 3), dtype=float)
        for index in range(len(refs) - 1):
            body_rates[index] = self._relative_body_rate(
                rotations[index], rotations[index + 1], self.config.dt
            )
        if len(refs) > 1:
            body_rates[-1] = body_rates[-2]

        feedforward_thrusts = np.array([
            self._feedforward_thrust(sample[6:9])
            for sample in refs
        ])
        return rotations, body_rates, feedforward_thrusts

    def _desired_rotation(self, acceleration: np.ndarray, yaw: float) -> np.ndarray:
        """Recover desired body orientation from flat acceleration + yaw (NED/FRD)."""
        acceleration = np.asarray(acceleration, dtype=float)
        force_direction = np.array([
            -acceleration[0],
            -acceleration[1],
            float(self._quad_parameters["g"]) - acceleration[2],
        ])
        norm = np.linalg.norm(force_direction)
        if norm < 1.0e-9:
            force_direction = np.array([0.0, 0.0, 1.0])
            norm = 1.0
        b3 = force_direction / norm

        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b2 = np.cross(b3, heading)
        b2_norm = np.linalg.norm(b2)
        if b2_norm < 1.0e-9:
            # This should not occur for normal quadrotor tilt limits (< 90 deg),
            # but keep the map well-defined for pathological references.
            fallback = np.array([0.0, 1.0, 0.0])
            b2 = np.cross(b3, fallback)
            b2_norm = np.linalg.norm(b2)
            if b2_norm < 1.0e-9:
                fallback = np.array([1.0, 0.0, 0.0])
                b2 = np.cross(b3, fallback)
                b2_norm = np.linalg.norm(b2)
        b2 /= b2_norm
        b1 = np.cross(b2, b3)
        b1 /= np.linalg.norm(b1)
        return np.column_stack((b1, b2, b3))

    def _feedforward_thrust(self, acceleration: np.ndarray) -> float:
        acceleration = np.asarray(acceleration, dtype=float)
        force_direction = np.array([
            -acceleration[0],
            -acceleration[1],
            float(self._quad_parameters["g"]) - acceleration[2],
        ])
        thrust = float(self._quad_parameters["m"]) * np.linalg.norm(force_direction)
        return float(np.clip(
            thrust,
            4.0 * float(self._quad_parameters["min_thrust"]),
            4.0 * float(self._quad_parameters["max_thrust"]),
        ))

    @staticmethod
    def _relative_body_rate(rotation: np.ndarray, next_rotation: np.ndarray, dt: float) -> np.ndarray:
        """Finite-difference desired body rate from two body-to-world rotations."""
        delta = rotation.T @ next_rotation
        cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        vee = np.array([
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ])

        if angle < 1.0e-7:
            rotation_vector = 0.5 * vee
        else:
            sine = np.sin(angle)
            if abs(sine) < 1.0e-8:
                # Near pi the log map is poorly conditioned. References from a
                # smooth trajectory should almost never land here in one MPC dt;
                # use the first-order skew part instead of injecting a huge rate.
                rotation_vector = 0.5 * vee
            else:
                rotation_vector = (angle / (2.0 * sine)) * vee
        return rotation_vector / dt

    def _stage_yref(
            self,
            sample: np.ndarray,
            rotation: np.ndarray,
            body_rate: np.ndarray,
            feedforward_thrust: float,
    ) -> np.ndarray:
        return np.concatenate((
            sample[0:3],
            sample[3:6],
            rotation.reshape(9, order="F"),
            body_rate,
            np.array([feedforward_thrust, 0.0, 0.0, 0.0]),
        ))

    @staticmethod
    def _terminal_yref(
            sample: np.ndarray,
            rotation: np.ndarray,
            body_rate: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate((
            sample[0:3],
            sample[3:6],
            rotation.reshape(9, order="F"),
            body_rate,
        ))

    # ------------------------------------------------------------------
    # State / solver utilities
    # ------------------------------------------------------------------

    def _hover_command(self) -> ControlCommand:
        thrust = float(self._quad_parameters["m"]) * float(self._quad_parameters["g"])
        thrust = float(np.clip(
            thrust,
            4.0 * float(self._quad_parameters["min_thrust"]),
            4.0 * float(self._quad_parameters["max_thrust"]),
        ))
        return ControlCommand(thrust=thrust, moment=np.zeros(3))

    @staticmethod
    def _normalized_state(state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=float).reshape(MPCController.NX).copy()
        q_norm = np.linalg.norm(x[3:7])
        if q_norm < 1.0e-12:
            x[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            x[3:7] /= q_norm
        if not np.all(np.isfinite(x)):
            raise ValueError("quadrotor state contains non-finite values")
        return x

    def _initialize_guess(self, x0: np.ndarray) -> None:
        hover = np.array([
            self._hover_command().thrust,
            0.0,
            0.0,
            0.0,
        ])
        for stage in range(self.config.horizon_steps + 1):
            self._solver.set(stage, "x", x0)
        for stage in range(self.config.horizon_steps):
            self._solver.set(stage, "u", hover)

    def _shift_warm_start(self) -> None:
        """Shift the previous optimal trajectory by one shooting interval."""
        horizon = self.config.horizon_steps
        previous_x = [
            np.asarray(self._solver.get(stage, "x"), dtype=float).copy()
            for stage in range(horizon + 1)
        ]
        previous_u = [
            np.asarray(self._solver.get(stage, "u"), dtype=float).copy()
            for stage in range(horizon)
        ]

        for stage in range(horizon):
            self._solver.set(stage, "x", previous_x[stage + 1])
        self._solver.set(horizon, "x", previous_x[-1])

        for stage in range(horizon - 1):
            self._solver.set(stage, "u", previous_u[stage + 1])
        self._solver.set(horizon - 1, "u", previous_u[-1])

    @staticmethod
    def _casadi_skew(ca, vector):
        return ca.vertcat(
            ca.horzcat(0.0, -vector[2], vector[1]),
            ca.horzcat(vector[2], 0.0, -vector[0]),
            ca.horzcat(-vector[1], vector[0], 0.0),
        )

    @staticmethod
    def _casadi_rotation(ca, quat):
        # quat is scalar-first q=[q0,q1,q2,q3]. Normalizing inside R makes the
        # output robust to tiny integration drift while qdot preserves norm in
        # the continuous model.
        q_norm = ca.sqrt(ca.dot(quat, quat) + 1.0e-12)
        q = quat / q_norm
        q0 = q[0]
        qv = q[1:4]
        skew = MPCController._casadi_skew(ca, qv)
        return ca.SX.eye(3) + 2.0 * (skew @ skew) + 2.0 * q0 * skew

    @staticmethod
    def _casadi_quaternion_derivative(ca, quat, omega):
        q0 = quat[0]
        qv = quat[1:4]
        scalar_dot = -0.5 * ca.dot(qv, omega)
        vector_dot = 0.5 * (q0 * omega + ca.cross(qv, omega))
        return ca.vertcat(scalar_dot, vector_dot)


__all__ = ["MPCConfig", "MPCController"]
