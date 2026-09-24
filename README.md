## LearnOpt Motion Planning

Learning and Optimization for Robot Motion Planning and Control.

LearnOpt implements and explores optimization-based planning and learning-based control for quadrotors in MuJoCo. It provides reusable planning libraries, traditional and reinforcement-learning controllers, wind disturbances, and configurable training and deployment experiments.

Built on [Mdhvince/UAV-Autonomous-control](https://github.com/Mdhvince/UAV-Autonomous-control), the project extends the original simulation and tracking framework with FIRI, GCOPTER/MINCO, GCS, BMTP, nonlinear MPC, and PPO with MLP or differentiable MPC policies.

## Current work and future work

- [x] Construct collision-free convex flight corridors with FIRI
- [x] Optimize constrained multicopter trajectories with GCOPTER/MINCO
- [x] Plan routes and Bézier trajectories using GCS
- [x] Track trajectories with acados nonlinear MPC
- [x] Simulate deterministic and randomized wind disturbances
- [x] Train PPO trajectory-tracking policies with an MLP baseline
- [x] Integrate differentiable MPC into PPO policies (ACMPC)
- [x] Independently implement biconvex minimum-time planning (BMTP)
- [x] Add a unified YAML entry for planning, training, deployment, and replay
- [x] Add reference-free gate racing with PPO-MLP and PPO-ACMPC
- [ ] Implement diffusion-based motion planning

The flight CLI supports trajectory tracking; the RL training/evaluation entrypoints also support gate racing without a planner or trajectory bank. Implementation details and experiment workflows are documented in the [planning library guide](uav_ac/planning/README.md), [RL workflow guide](uav_ac/rl/README.md), [gate racing guide](docs/gate_racing.md), and [contributor guide](AGENT.md).

## Experiment videos

The videos show the implemented planning and tracking pipelines in MuJoCo. The
orange curve is the planned trajectory and the blue curve is the actual flight
path. Convex regions are not drawn in the videos; they are shown separately in
the FIRI figures below.

### GCOPTER + MPC

https://github.com/user-attachments/assets/b8622729-85a3-48b4-996e-c6940f82b589

**Planner:** FIRI + GCOPTER  
**Controller:** MPC

### Minimum Snap + Cascaded

https://github.com/user-attachments/assets/21c21263-08e6-4232-9e44-911eec212278

**Planner:** Minimum Snap baseline  
**Controller:** Cascaded

### GCS + MPC

https://github.com/user-attachments/assets/0cb63fd2-9fe3-4f56-8253-874900b20b00

**Planner:** GCS  
**Controller:** Nonlinear MPC

## Methods

### FIRI

FIRI builds collision-free convex regions around seed points or path segments. Each region has the half-space representation

```math
\mathcal{P}=\{\mathbf{x}\in\mathbb{R}^{3}\mid A\mathbf{x}\le\mathbf{b}\}.
```

It alternates obstacle-separating half-spaces with maximum-volume inscribed ellipsoid (MVIE) optimization. Overlapping regions form a safe corridor for trajectory optimization.

![FIRI corridor overview](docs/firi_corridor_overview.png)

### GCOPTER / MINCO

GCOPTER optimizes smooth multicopter trajectories inside the corridor using MINCO. Polynomial coefficients are recovered from compact spatial and temporal variables:

```math
\mathbf{p}_i(t)=C_i^\top[1,t,\ldots,t^{2s-1}]^\top,\qquad
C=\mathcal{M}(\mathbf{q},\mathbf{T}).
```

This implementation uses $s=3$, giving piecewise quintic trajectories. The objective combines smoothness, duration, and geometric/dynamic feasibility penalties:

```math
J=\int_0^{T_\Sigma}\|\mathbf{p}^{(s)}(t)\|_2^2\,dt
+\rho T_\Sigma+J_{\mathrm{pen}}.
```

L-BFGS optimizes the compact variables; the mission adapter samples the result into position, velocity, acceleration, and yaw references.

### GCS

GCS starts from a cover of collision-free convex sets $\{\mathcal X_v\}_{v\in V}$. Each set is a graph vertex; a directed edge $(u,v)\in E$ is added when $\mathcal X_u\cap\mathcal X_v\neq\varnothing$, with singleton source and goal vertices added in the same way. The preliminary optimization is therefore a shortest-path problem coupled to continuous trajectory variables: binary edge flows select one source-to-goal path, while Bézier control points remain in the selected convex sets.

```math
\min_{\phi,\tilde P}\ \sum_{e\in E}\ell_e(\tilde P_e)
\quad\mathrm{s.t.}\quad B\phi=b,\quad
\phi_e\in\{0,1\},\quad
\tilde P_{e,k}\in\phi_e\mathcal X_{\mathrm{tail}(e)},\quad
\tilde P'_{e,k}\in\phi_e\mathcal X_{\mathrm{head}(e)}.
```

This is a mixed-integer convex program. The implementation replaces $\phi_e\in\{0,1\}$ with $0\le\phi_e\le1$ and uses perspective constraints, yielding an SOCP relaxation. It rounds the resulting flow to one path and solves a convex Bézier restriction on that path with endpoint and derivative-continuity constraints. The convex-hull property then keeps every Bézier segment inside its assigned region before time parameterization for tracking.

### Differentiable MPC and reinforcement learning (ACMPC)

Actor-Critic Model Predictive Control (ACMPC) places a differentiable MPC layer at the end of the actor. A neural cost map turns the observation into time-varying quadratic cost residuals; together with known quadrotor dynamics and bounded inputs, they define the MPC action mean:

```math
c_\theta(o_t)=\{Q_k,q_k\}_{k=0}^{N},\qquad
\mu_\theta(o_t)=u_0^\star,\qquad
\pi_\theta(a\mid o_t)=\mathcal N(\mu_\theta(o_t),\Sigma),
\quad x_{k+1}=f_d(x_k,u_k),\ u_k\in\mathcal U.
```

The first optimized input is the Gaussian actor mean, while a separate critic estimates the long-horizon return. PPO therefore learns the MPC cost map from reward rather than directly regressing an action. Gradients pass through the solver to the cost network:

```math
\nabla_\theta L_{\mathrm{RL}}
=
\left(\frac{\partial c_\theta}{\partial\theta}\right)^\top
\left(\frac{\partial\mathbf{u}_0^\star}{\partial c_\theta}\right)^\top
\nabla_{\mathbf{u}_0^\star}L_{\mathrm{RL}}.
```

This repository follows the paper's one-iLQR-update actor and uses an analytic differentiable backward pass. Its actor learns diagonal quadratic-cost residuals around a tracking cost; the critic is an MLP trained by PPO. Model-Predictive Value Expansion (MPVE) from the extended paper is not included.

### Biconvex Minimum-Time Planning (BMTP)

BMTP jointly represents a Bézier trajectory $p(\tau)$ and time-varying separating planes $(a(\tau),b(\tau))$. For every convex obstacle $\mathcal O$, the required margin $\delta>0$ is

```math
a(\tau)^\top p(\tau)+b(\tau)\le-\delta,
\qquad
a(\tau)^\top x+b(\tau)\ge\delta
\quad\forall x\in\mathcal O.
```

The term $a(\tau)^\top p(\tau)$ is bilinear. BMTP alternates between plane fitting with $p$ fixed and smooth minimum-time trajectory optimization with $(a,b)$ fixed:

```math
(a,b)\leftarrow\arg\min\ \mathrm{plane\ violation}\mid p,
\qquad
(p,h)\leftarrow\arg\min\ Mh\mid(a,b),\ \text{smoothness and derivative limits}.
```

Each update is convex; the overall alternating method is local and depends on its collision-free initialization.

## Setup and running

Python 3.13, [uv](https://docs.astral.sh/uv/), and a graphical desktop for MuJoCo are required.

```bash
uv sync --python 3.13
```

Set `scene`, `planner`, and `controller` in `configs/flight.yaml`, then run:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight.yaml
```

The generic four-axis aerial manipulator uses a floating quadrotor base, a
simulation-only hanging `yaw + 3 pitch` arm, and a symmetric actuated parallel
gripper. The `gripper_opening` command sets the inner-finger gap from 0.020 m
closed to 0.070 m open. Run its 20-second headless hover
and slow joint-motion check with `configs/aerial_manipulator_hover.yaml`; set
`visualize: true` there to open the MuJoCo viewer. The arm dimensions and
inertias are initial simulation assumptions: four arm links total 0.43 m and
0.09 kg, and the gripper weighs 0.023 kg. With the 0.50 kg quadrotor body,
the modeled takeoff mass is 0.613 kg. Planning/control in this demo
do not provide whole-body obstacle avoidance or grasping.

The public entry point is `simulation.robot`. Its 12-value configuration is
`[position_NED(3), quaternion_FRD_to_NED_wxyz(4), arm_q(4), gripper_gap(1)]`;
its 11-value velocity is `[linear_velocity_NED(3), angular_velocity_FRD_body(3),
arm_qdot(4), gripper_gap_rate(1)]`. `robot.jacobian(q, frame)` returns a
6×11 analytic geometric Jacobian with both twist rows in world NED. Frames
`"tool"` and `"grasp"` select the wrist tool frame and center of the gripper.
Configuration queries use separate MuJoCo data and do not change the running
simulation. A minimal control loop is:

```python
state = simulation.robot.state
q = simulation.robot.configuration
J = simulation.robot.jacobian(q, frame="grasp")
model = simulation.robot.dynamics(q, state.velocity)
collision = simulation.robot.check_collision(q, clearance=0.02)
command = controller.step(reference)  # AerialManipulatorReference
simulation.robot.apply(command)
simulation.step()
```

`robot.dynamics` returns the reduced mass matrix, bias and passive forces, and
an actuation matrix for four actual rotor forces, four arm torques, and the
left gripper servo force. Its gripper model assumes ideally synchronized
fingers; MuJoCo enforces synchronization with a soft equality constraint and
drives one finger with a position servo. Rotor commands are allocated to
targets when applied, and motor response advances exactly once per physics
step. `robot.check_collision` checks the supplied full configuration against
collidable environment geometry and non-adjacent robot parts, returning named
pairs and their distances. Physics masses, dimensions and inertias remain
simulation assumptions rather than measured hardware properties.

## Flight configuration

`scene` names an XML file under `uav_ac/simulation/models/`, with or without the `.xml` suffix. Standard trajectory flights omit `task` (defaulting to `trajectory_tracking`). Reference-free gate-racing deployment uses `task: gate_racing`, `scene: gate_racing`, `planner: none`, and `controller: rl`; see [flight_gate_racing.yaml](configs/flight_gate_racing.yaml).

| Field | Values | Notes |
| --- | --- | --- |
| `planner` | `none`, `mini_snap`, `gcopter`, `gcs`, `bmtp` | `none` is valid for gate racing and the aerial-manipulator demo; GCS needs scene guide regions; BMTP needs `bmtp_route_*` sites |
| `controller` | `cascaded`, `mpc`, `rl` | MPC requires acados; RL requires `rl.checkpoint` |
| `wind` | `none`, `fixed_gust` | Add `wind_options` only to override fixed-gust defaults |
| `visualize` | `true`, `false` | Shows corridor geometry; BMTP always shows its dashed initialization and solid result |

Planner/controller-specific blocks can coexist as presets; only the block belonging to the selected planner or controller is used. `configs/flight.yaml` includes complete presets for `gcopter`, `gcs`, `bmtp`, `cascaded`, `mpc`, and `rl`. For example:

```yaml
scene: bmtp_village
planner: bmtp
controller: cascaded
speed: 3.0
bmtp:
  initial_route: 3
  segments: 8
  clearance: 0.15
```

Use `scene: gcs_building` with `planner: gcs`. `gcopter:` and `gcs:` can override their native dataclass settings; `mpc:` configures the nonlinear controller. Traditional controllers default to a 0.01 s `control_dt`, which can be overridden with an integer multiple of the XML timestep. RL checkpoints own their control period. Viewer runs do not create output directories or overwrite previous results. Use `uav_ac.record_experiments` for videos.

To open a trained reference-free gate-racing policy in the native viewer:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight_gate_racing.yaml
```

This path uses `planner: none`; the policy observes the next gates directly and drives normalized thrust/body moments. It does not construct a trajectory or invoke GCOPTER.

For GCOPTER, keep `speed` as the shared velocity bound. `gcopter:` exposes trajectory scale (`length_per_piece`, `time_weight`), dynamic limits (`max_acceleration`, `max_body_rate`), soft-constraint weights, and optimizer convergence settings. `gcs:` exposes the Bézier graph optimization and solver settings; `mpc:` exposes NMPC horizon, tracking weights, and solver settings; `cascaded:` exposes response time constants, damping, and altitude integration. Mass, thrust, tilt, and flight-speed limits remain in the selected XML vehicle.

## Optional MPC and RL workflows

### acados MPC

Build acados using its [installation instructions](https://docs.acados.org/installation/index.html), then install the Python interface into the project environment:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
uv pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
.venv/bin/python -c "from acados_template import AcadosOcpSolver"
```

Replace `/path/to/acados` with your installation path. Select the controller in `configs/flight.yaml`:

```yaml
controller: mpc
mpc:
  horizon_steps: 10
  nlp_solver_type: SQP_RTI
```

### Train and deploy RL

Trajectory banks and checkpoints under `runs/` are local artifacts and are **not included in the repository**.

First prepare a trajectory bank. The default configuration generates 200 training, 20 validation, and 20 test trajectories in the open-field scene and validates them with acados MPC:

```bash
.venv/bin/python -m uav_ac.rl.training --config configs/ppo_trajectory.yaml \
  --run-dir runs/ppo_trajectory/multitraj01 --prepare-only
```

Generation can take substantial time. Train ACMPC with its independent training configuration and a new run directory:

```bash
.venv/bin/python -m uav_ac.rl.training \
  --config configs/acmpc_trajectory.yaml \
  --run-dir runs/acmpc_trajectory/multitraj01
```

Edit `configs/acmpc_trajectory.yaml` for its trajectory bank, device, MPC, PPO, wind, and training scale. Use `configs/ppo_trajectory.yaml` for the MLP baseline.

To deploy either an MLP or ACMPC policy, let `main` plan the reference and select the explicit checkpoint:

```yaml
scene: open_field
planner: gcopter
controller: rl
rl:
  checkpoint: ../runs/acmpc_trajectory/multitraj01/best_model.zip
  device: cuda
```

Checkpoint paths are relative to the YAML file. Keep `rl_config.json` beside the model; the deployed controller adopts and validates the trained control period automatically. Dataset-based metrics, interactive evaluation, and recording are available through `uav_ac.rl.evaluate`; it selects the task from run metadata.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q
```

See [AGENT.md](AGENT.md) for contributor guidance, module contracts, and coverage checks.

## Repository structure

```text
configs/
├── flight.yaml               Short interactive-flight configuration
├── flight_gate_racing.yaml   Gate-racing policy viewer configuration
├── ppo_trajectory.yaml       MLP training and trajectory-bank settings
├── acmpc_trajectory.yaml     ACMPC training settings
└── bmtp.yaml                 Standalone BMTP experiment settings
uav_ac/
├── main.py                   Scene, planner, controller, wind, and viewer entry
├── scenes/                   XML loading and scene metadata
├── tasks/                    Task protocol and trajectory tracking
├── envs/                     Generic MuJoCo Gym environment
├── planning/
│   ├── geometry/             Convex geometry and collision utilities
│   ├── search/               RRT* path search
│   ├── corridor/firi/        Convex safe-corridor construction
│   ├── trajectory/           GCOPTER, GCS, BMTP, and Minimum Snap
│   └── pipeline/             Mission and trajectory conversion
├── control/                  Cascaded/MPC control and tracking interfaces
├── rl/
│   ├── training.py           Unified task-aware MLP/ACMPC training entry point
│   ├── evaluate.py           Unified task-aware evaluation/viewer entry point
│   ├── tasks/                Task-owned RL workflows, including gate racing
│   ├── common/               Registry, shared config, and trajectory-bank assets
│   ├── acmpc/                Differentiable MPC policy and solver
│   └── mlp_baseline/         Existing training/evaluation entry points
├── simulation/
│   └── models/               XML scenes and vehicle definitions
├── robot/
│   ├── quadrotor/             Quadrotor state and rotor allocation
│   └── aerial_manipulator/    Four-joint arm model and robot state/commands
├── quadrotor/                Compatibility import for the original package
└── visualization/            Planning overlays and plots
tests/                        Unit and integration tests
docs/                         Figures and documentation media
runs/                         Local trajectory banks and trained models
```

For reusable APIs and extension points, see the [planning guide](uav_ac/planning/README.md) and [contributor guide](AGENT.md).

## References

1. Mdhvince, **UAV-Autonomous-control**. GitHub repository.  
   https://github.com/Mdhvince/UAV-Autonomous-control

2. Z. Wang, X. Zhou, C. Xu, and F. Gao, “Geometrically Constrained Trajectory
   Optimization for Multicopters,” *IEEE Transactions on Robotics*, vol. 38,
   no. 5, pp. 3259–3278, 2022.  
   DOI: https://doi.org/10.1109/TRO.2022.3160022

3. Q. Wang, Z. Wang, M. Wang, J. Ji, Z. Han, T. Wu, R. Jin, Y. Gao, C. Xu, and
   F. Gao, “Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex
   Regions of Obstacle-Free Space,” *IEEE Transactions on Robotics*, vol. 41,
   pp. 3223–3243, 2025.  
   DOI: https://doi.org/10.1109/TRO.2025.3562482

4. T. Marcucci, M. Petersen, D. von Wrangel, and R. Tedrake, “Motion Planning
   around Obstacles with Convex Optimization,” *Science Robotics*, vol. 8,
   no. 84, eadf7843, 2023.  
   DOI: https://doi.org/10.1126/scirobotics.adf7843

5. D. Mellinger and V. Kumar, “Minimum Snap Trajectory Generation and Control
   for Quadrotors,” in *2011 IEEE International Conference on Robotics and
   Automation (ICRA)*, pp. 2520–2525, 2011.  
   DOI: https://doi.org/10.1109/ICRA.2011.5980409

6. R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren,
   A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl,
   “acados—a modular open-source framework for fast embedded optimal control,”
   *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022.  
   DOI: https://doi.org/10.1007/s12532-021-00208-8

7. P. Werner, T. Marcucci, and D. Rus, “Biconvex Optimization for Smooth
   Minimum-Time Trajectories around Convex Obstacles,” *arXiv preprint
   arXiv:2608.02834*, 2026.
   https://arxiv.org/abs/2608.02834

8. A. Romero, E. Aljalbout, Y. Song, and D. Scaramuzza, “Actor–Critic Model
   Predictive Control: Differentiable Optimization Meets Reinforcement Learning
   for Agile Flight,” *IEEE Transactions on Robotics*, 2025.
   DOI: https://doi.org/10.1109/TRO.2025.3644945
