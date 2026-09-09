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
- [ ] Extend RL training and deployment to tasks beyond trajectory tracking
- [ ] Implement diffusion-based motion planning

The current CLI supports trajectory tracking. A generic MuJoCo task interface is available for developing additional tasks. Implementation details and experiment workflows are documented in the [planning library guide](uav_ac/planning/README.md) and [contributor guide](AGENT.md).

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

Graph of Convex Sets represents collision-free convex regions as graph vertices and their connections as edges. A segment within region $\mathcal{X}_v$ is a Bézier curve:

```math
\mathbf{r}_v(\tau)=\sum_{k=0}^{d}B_{k,d}(\tau)\mathbf{P}_{v,k},
\qquad \mathbf{P}_{v,k}\in\mathcal{X}_v.
```

The convex-hull property keeps each segment inside its region. Convex relaxation couples region selection with continuous curve optimization; the mission adapter then time-parameterizes the geometric path for tracking.

### Nonlinear MPC

The acados controller tracks the reference through a finite-horizon optimal-control problem:

```math
\min_{\mathbf{u}_{0:N-1}}
\sum_{k=0}^{N-1}
\left(\|\mathbf{x}_k-\mathbf{x}^{\mathrm{ref}}_k\|_Q^2
+\|\mathbf{u}_k-\mathbf{u}^{\mathrm{ref}}_k\|_R^2\right)
+\|\mathbf{x}_N-\mathbf{x}^{\mathrm{ref}}_N\|_{Q_f}^2,
\quad
\mathbf{x}_{k+1}=f_d(\mathbf{x}_k,\mathbf{u}_k).
```

It applies the first optimized input and replans at the next control tick. Terminal feedback improves end-of-trajectory tracking. Cascaded control and Minimum Snap provide conventional tracking and planning baselines.

### Differentiable MPC and reinforcement learning (ACMPC)

Actor-Critic Model Predictive Control (ACMPC) embeds a differentiable MPC solver inside the actor. A neural cost map converts the observation $o_t$ into quadratic MPC cost parameters, and MPC uses these costs, the dynamics, and input constraints to optimize a short-horizon control sequence:

```math
c_\theta=g_\theta(o_t),\qquad
\mathbf{u}_{0:N-1}^{\star}=
\arg\min_{\mathbf{u}}
\sum_{k=0}^{N-1}\ell_{c_\theta}(\mathbf{x}_k,\mathbf{u}_k)
+\ell_{c_\theta,N}(\mathbf{x}_N),
\quad
\mathbf{x}_{k+1}=f_d(\mathbf{x}_k,\mathbf{u}_k),\quad \mathbf{u}_k\in\mathcal U.
```

The first optimized input $\mathbf{u}_0^\star$ becomes the actor's action mean after normalization, while a separate critic estimates the long-horizon return. Reinforcement learning therefore learns the MPC cost from interaction rewards rather than predicting the action directly. Since the solver is differentiable, the policy gradient can pass through the optimization solution to the neural cost map:

```math
\nabla_\theta L_{\mathrm{RL}}
=
\left(\frac{\partial c_\theta}{\partial\theta}\right)^\top
\left(\frac{\partial\mathbf{u}_0^\star}{\partial c_\theta}\right)^\top
\nabla_{\mathbf{u}_0^\star}L_{\mathrm{RL}}.
```

This combines RL's reward-driven exploration with MPC's model-based prediction and online replanning. In this repository, PPO supplies the actor-critic update, and the actor uses a box-constrained iLQR-based differentiable solver with an approximate fixed-point backward pass. The paper's Model Predictive Value Expansion (MPVE) component is not implemented here.

### Biconvex Minimum-Time Planning (BMTP)

BMTP alternates between optimizing a smooth trajectory and the planes separating it from convex obstacles. For a curve $p(\tau)$ and obstacle $\mathcal O$, separation is expressed as

```math
a(\tau)^\top p(\tau)+b(\tau)\le-\delta,\qquad
a(\tau)^\top v+b(\tau)\ge\delta
\quad\forall v\in\mathcal O,\ \forall\tau\in[0,1],
```

where $\delta>0$ is a separation margin. The product $a(\tau)^\top p(\tau)$ couples the plane and trajectory variables bilinearly. Fixing one block makes the separation constraints convex in the other.

Starting from a collision-free polygonal path, BMTP repeats two convex subproblems: **fix the trajectory and update the separating planes**, then **fix the planes and optimize the trajectory and duration**, subject to smoothness and derivative limits. Bézier representations and a conic time formulation make these updates tractable. The iterations reduce travel time while maintaining separation; they do not guarantee a global optimum. See the [BMTP experiments](uav_ac/planning/README.md) for comparisons across initial paths.

The trajectory is represented by Bézier control points $P$ and separating planes $(a,b)$. Their collision constraint contains the bilinear term $a(\tau)^\top p(\tau)$:

```math
a(\tau)^\top p(\tau)+b(\tau)\le-\delta,\qquad
a(\tau)^\top x+b(\tau)\ge\delta
\quad \forall x\in\mathcal O.
```

BMTP alternates two convex subproblems. First, it fixes $P$ and updates the obstacle-separating planes. Then it fixes $(a,b)$ and optimizes the Bézier trajectory, smoothness, derivative limits, and segment duration:

```math
\min_{P,h}\;T=Mh
\quad\text{s.t.}\quad
p(0)=p_{\mathrm{start}},\quad p(1)=p_{\mathrm{goal}},\quad
\|p^{(k)}(t)\|_2\le d_k,\quad
p(t)\ \text{stays inside the fixed corridor}.
```

Fixing either block removes the bilinearity, so each update is convex. The alternating procedure improves the trajectory locally, depends on the initial collision-free path, and does not guarantee a global optimum.

## Setup and running

### Requirements and installation

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A graphical desktop for the interactive MuJoCo viewer
- FFmpeg and a working MuJoCo rendering backend for video recording
- A separately built acados installation for nonlinear MPC and trajectory-bank validation

Run all commands from the repository root:

```bash
uv sync --python 3.13
```

This installs the Python dependencies, including MuJoCo, the planners, PyTorch, SB3, and test tools. Cascaded control, MLP policies, and ACMPC inference do not require acados.

### Run an experiment

Start with GCOPTER and the cascaded controller; no trained checkpoint is needed:

```bash
.venv/bin/python -m uav_ac.main --config configs/experiments/lab_gcopter_cascaded.yaml
```

To try BMTP with its dedicated scene:

```bash
.venv/bin/python -m uav_ac.main --config configs/experiments/bmtp_cascaded.yaml
```

Copy an example YAML to create your own experiment. Edit its settings rather than Python constants:

| Section | What to configure |
| --- | --- |
| `mode` | `plan`, `train`, `evaluate`, `deploy`, or saved-state `replay` |
| `scene` | XML file, directly or through a descriptor in `configs/scenes/` |
| `task` | Task, reference source, initialization, and success settings |
| `planner` | Planner, initial path, limits, and algorithm options |
| `agent` | A controller, an RL training policy, or a deployment checkpoint |
| `disturbance.wind` | `none`, `fixed_gust`, or `random_gust`; forces are in newtons |
| `execution` | Action period, seed, and deployment episode limits |
| `training` | PPO hyperparameters, environment count, and curriculum |
| `output` | Result directory, viewer, recording, and planning visualization |

Paths are relative to the YAML that defines them. XML files remain in `uav_ac/simulation/models/`; `configs/scenes/` holds their YAML descriptors. Selecting a planner does not change the selected scene.

Available planners and controllers are:

| Configuration field | Available values | Notes |
| --- | --- | --- |
| `planner.name` | `mini_snap`, `gcopter`, `gcs`, `bmtp` | `gcopter` uses an FIRI corridor; `gcs` is intended for `gcs_building.xml`; `bmtp` requires `bmtp_village.xml` and its `bmtp_route_*` sites |
| `agent.type: controller` | `agent.name: cascaded`, `agent.name: mpc` | Cascaded control needs no external solver; MPC requires acados |
| `agent.type: rl` | `agent.policy: mlp`, `agent.policy: acmpc` | For deployment, set `agent.checkpoint` to an explicit `.zip` model |

The available XML scenes are `lab_course`, `open_field`, `gcs_building`, and
`bmtp_village`, selected through `scene.config`. A common combination is
`gcopter + cascaded + lab_course`; use `gcs + gcs_building` and
`bmtp + bmtp_village` for the planners that require dedicated scene metadata.

For tracking, choose one reference source: `planner`, `saved_trajectory`, or `trajectory_bank`. Omit `planner` when using a saved reference. Choose either `agent.type: controller` or `agent.type: rl`. The action period must be an integer multiple of the XML physics timestep.

Set `output.viewer: false` for headless runs, and `output.record: true` to save video during deployment or replay. Use a new `output.directory` for each run; an existing `resolved_config.yaml` prevents overwriting the experiment. Deployment saves episode states and metrics alongside the resolved settings. Replay uses those saved states without replanning.

### Optional: acados MPC

Build acados using its [installation instructions](https://docs.acados.org/installation/index.html), then install the Python interface into the project environment:

```bash
export ACADOS_SOURCE_DIR=/path/to/acados
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
uv pip install -e "$ACADOS_SOURCE_DIR/interfaces/acados_template"
.venv/bin/python -c "from acados_template import AcadosOcpSolver"
```

Replace `/path/to/acados` with your installation path. Select the controller in your experiment YAML:

```yaml
agent:
  type: controller
  name: mpc
  options:
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

Generation can take substantial time. For training, set `task.reference.path` to the compatible bank, choose `agent.device: cpu` or `cuda`, and adjust `training.n_envs`, `training.ppo`, and `output.directory`:

```bash
.venv/bin/python -m uav_ac.main --config configs/training/acmpc_open_field.yaml
```

The example requests 10 million timesteps with 24 environments. For an MLP baseline, copy it, set `agent.policy: mlp`, and remove `agent.mpc`. The unified training entry currently uses an existing trajectory bank and supports no wind or randomized gusts.

To deploy a trained policy, edit `configs/experiments/acmpc_deploy.yaml` with your trajectory-bank path, device, and explicit `.zip` checkpoint:

```bash
.venv/bin/python -m uav_ac.main --config configs/experiments/acmpc_deploy.yaml
```

Keep `rl_config.json` with the checkpoint. Use `execution.action_dt: from_model` to adopt its trained control period. The loader checks physics, timing, and observation/action compatibility.

### Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -o addopts='' -q
```

See [AGENT.md](AGENT.md) for contributor guidance, module contracts, and coverage checks.

## Repository structure

```text
configs/
├── scenes/                   YAML descriptors for XML scenes
├── experiments/              Planning and deployment examples
└── training/                 RL training examples
uav_ac/
├── main.py                   Unified --config entry
├── experiments/              Config validation, execution, and replay
├── scenes/                   XML loading and scene metadata
├── tasks/                    Task protocol and trajectory tracking
├── envs/                     Generic MuJoCo Gym environment
├── planning/
│   ├── api.py                Unified mission-planning adapter
│   ├── geometry/             Convex geometry and collision utilities
│   ├── search/               RRT* path search
│   ├── corridor/firi/        Convex safe-corridor construction
│   ├── trajectory/           GCOPTER, GCS, BMTP, and Minimum Snap
│   └── pipeline/             Mission and trajectory conversion
├── control/                  Cascaded/MPC control and tracking interfaces
├── deployment/               Controller and policy episode adapters
├── rl/
│   ├── training.py           Shared MLP/ACMPC trainer and bank preparation
│   ├── common/               Trajectory banks and initialization assets
│   ├── acmpc/                Differentiable MPC policy and solver
│   └── mlp_baseline/         Existing training/evaluation entry points
├── simulation/
│   └── models/               XML scenes and vehicle definitions
├── quadrotor/                Vehicle state and actuator allocation
└── visualization/            Planning overlays and plots
tests/                        Unit and integration tests
docs/                         Figures and documentation media
runs/                         Local datasets, models, and experiment outputs
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

8. ACMPC: [paper](https://arxiv.org/html/2306.09852v8) and
   [official implementation](https://github.com/uzh-rpg/acmpc_public).
   The bundled differentiable solver comes from
   [mpc.pytorch_acmpc](https://github.com/uzh-rpg/mpc.pytorch_acmpc);
   its pinned revision, MIT license, and local changes are documented in
   [SOURCE.md](uav_ac/rl/acmpc/vendor/SOURCE.md).
