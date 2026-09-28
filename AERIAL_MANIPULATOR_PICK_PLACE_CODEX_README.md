# Aerial Manipulator Pick-and-Place Planner
## Codex implementation specification for LearnOpt_Motion_Planning

> **Repository:** `zefew6/LearnOpt_Motion_Planning`  
> **Target module:** `uav_ac/planning/trajectory/gcopter/aerial_manipulator/`  
> **Goal:** implement the first complete whole-body motion-planning demo for a quadrotor + 4-DoF serial arm:
>
> \[
> A \rightarrow B\;(\text{grasp}) \rightarrow C\;(\text{release})
> \]
>
> The gripper action itself is **not** trajectory-optimized. The planner only needs to bring the grasp frame to the requested pick/place position. A small state machine performs close/open and payload attachment/detachment.

---

# 1. Scope and implementation contract

This document defines the first implementation milestone. Codex should treat the mathematical and architectural decisions below as fixed unless an existing repository API makes a small mechanical adjustment unavoidable.

The first version must provide the following pipeline:

\[
\boxed{
\text{8D RRT-Connect}
\rightarrow
\text{8D MINCO initialization}
\rightarrow
\text{L-BFGS whole-body refinement}
\rightarrow
\text{trajectory tracking}
\rightarrow
\text{grasp/release state machine}
}
\]

The motion planner handles two independent legs:

\[
A\rightarrow B,
\qquad
B\rightarrow C.
\]

Each leg has its own total optimized duration:

\[
T_{AB},\qquad T_{BC}.
\]

Do **not** optimize grasp closure/contact dynamics. The event logic is:

```text
PLAN_TO_PICK
    ↓
MOVE_TO_PICK
    ↓
GRASP
    ↓
PLAN_TO_PLACE
    ↓
MOVE_TO_PLACE
    ↓
RELEASE
    ↓
DONE
```

For a static scene it is also acceptable to pre-plan both legs before execution.

---

# 2. Explicit V1 non-goals

Do not implement the following in this milestone:

- contact-implicit trajectory optimization;
- force planning or force control;
- trajectory optimization of finger motion;
- arbitrary end-effector \(SE(3)\) terminal manifolds;
- full payload rigid-body dynamics in the optimizer;
- full coupled floating-base inverse-dynamics constraints inside L-BFGS;
- numerical finite differences inside the optimizer;
- GCS integration for this task;
- learning-based warm starts;
- online replanning;
- a new robot kinematics/dynamics implementation that duplicates `robot/aerial_manipulator/model.py`.

The first version should be simple enough to debug, but all gradients used by L-BFGS should have an analytic chain-rule form.

---

# 3. Existing repository structure

Relevant current structure:

```text
uav_ac/
├── control/
│   └── aerial_manipulator_controller.py
├── planning/
│   ├── geometry/
│   ├── search/
│   │   └── rrt_star.py
│   └── trajectory/
│       └── gcopter/
│           ├── config.py
│           ├── mappings.py
│           ├── minco.py
│           ├── optimizer.py
│           ├── penalties.py
│           ├── planner.py
│           └── types.py
├── robot/
│   └── aerial_manipulator/
│       ├── __init__.py
│       └── model.py
├── simulation/
│   └── model/
│       └── aerial_manipulator.xml
└── tasks/
```

The existing robot model already owns:

- public robot configuration and tangent velocity;
- manifold `integrate()` / `difference()`;
- forward kinematics;
- geometric Jacobian;
- mass properties;
- reduced mass matrix;
- bias/passive forces;
- actuation matrix;
- MuJoCo collision validation.

Do not copy these functions into the planner.

The robot model answers:

> “What is the robot geometry/dynamics at this configuration?”

The planner answers:

> “What is the cost and gradient of this complete MINCO trajectory?”

---

# 4. New files

Create:

```text
uav_ac/planning/geometry/esdf.py

uav_ac/planning/search/rrt_connect.py

uav_ac/planning/trajectory/gcopter/aerial_manipulator/
├── __init__.py
├── README.md
├── config.py
├── types.py
├── planner.py
├── evaluator.py
├── collision.py
└── task_targets.py

uav_ac/tasks/aerial_pick_place.py

configs/aerial_manipulator_pick_place.yaml
```

Tests:

```text
tests/unit/planning/test_rrt_connect.py
tests/unit/planning/test_aerial_manipulator_minco.py
tests/unit/planning/test_aerial_manipulator_gradients.py
tests/unit/planning/test_esdf.py
tests/integration/test_aerial_pick_place.py
```

Do not create separate changelog Markdown files. Add only a concise usage section to the existing root/planning documentation after the implementation is stable.

---

# 5. Robot and planning state

## 5.1 Physical robot

The aerial manipulator is a floating-base system with a 4R serial arm:

\[
R_z-R_y-R_y-R_y.
\]

The complete floating-base configuration conceptually belongs to

\[
SE(3)\times\mathbb R^4.
\]

The current public robot configuration also contains gripper opening.

## 5.2 MINCO planning state

The MINCO trajectory is **8-dimensional**:

\[
\boxed{
\sigma(t)
=
\begin{bmatrix}
p_x(t)\\
p_y(t)\\
p_z(t)\\
\psi(t)\\
q_1(t)\\
q_2(t)\\
q_3(t)\\
q_4(t)
\end{bmatrix}
\in\mathbb R^8.
}
\]

Interpretation:

\[
\underbrace{[p_x,p_y,p_z,\psi]}_{\text{quadrotor flat outputs}}
+
\underbrace{[q_1,q_2,q_3,q_4]}_{\text{arm joints}}.
\]

Roll and pitch are not optimization variables. They are recovered from the quadrotor flatness map for dynamic-feasibility penalties and controller references.

The gripper opening is not an optimization variable.

---

# 6. Segment polynomial definition

Let one leg contain \(M\) polynomial pieces.

Every piece is an 8-dimensional quintic polynomial.

For piece \(i\), local physical time \(t\in[0,h]\),

\[
\boxed{
\sigma_i(t)
=
C_i^T\beta(t)
}
\]

with

\[
C_i\in\mathbb R^{6\times 8}
\]

and

\[
\beta(t)
=
\begin{bmatrix}
1&t&t^2&t^3&t^4&t^5
\end{bmatrix}^T.
\]

Equivalently,

\[
\sigma_i(t)
=
c_{i,0}
+c_{i,1}t
+c_{i,2}t^2
+c_{i,3}t^3
+c_{i,4}t^4
+c_{i,5}t^5
\]

where each

\[
c_{i,k}\in\mathbb R^8.
\]

The derivatives are

\[
\dot\sigma_i
=
c_1+2c_2t+3c_3t^2+4c_4t^3+5c_5t^4,
\]

\[
\ddot\sigma_i
=
2c_2+6c_3t+12c_4t^2+20c_5t^3,
\]

\[
\sigma_i^{(3)}
=
6c_3+24c_4t+60c_5t^2,
\]

\[
\sigma_i^{(4)}
=
24c_4+120c_5t.
\]

MINCO must enforce the standard inter-piece continuity.

---

# 7. One total time variable only

This is a fixed design decision for V1.

Do not optimize \(M\) independent piece durations.

Optimize only one total leg duration:

\[
\boxed{T>0.}
\]

Every segment uses the same duration:

\[
\boxed{
h=\frac{T}{M}.
}
\]

Thus

\[
T_i=h,\qquad i=1,\dots,M.
\]

Use a scalar unconstrained variable \(\tau\) and a positive mapping

\[
T=\mathcal T(\tau)>0.
\]

Prefer reusing the repository's existing GCOPTER positive-time mapping. If a scalar helper is cleaner, it may call the existing vector mapping with length one.

For an existing per-piece duration gradient \(g_{T_i}\),

\[
\boxed{
\frac{\partial J}{\partial T}
=
\frac{1}{M}
\sum_{i=1}^{M}
\frac{\partial J}{\partial T_i}.
}
\]

Then apply the positive-time mapping derivative:

\[
\frac{\partial J}{\partial \tau}
=
\frac{\partial J}{\partial T}
\frac{dT}{d\tau}.
\]

This reduction is important for future time-scaling and continuous-time feasibility analysis.

---

# 8. Optimization variables

Let the leg use \(M-1\) interior waypoints:

\[
P_k\in\mathbb R^8,
\qquad
k=1,\dots,M-1.
\]

The optimization variable is

\[
\boxed{
z=
[P_1,\dots,P_{M-1},\tau].
}
\]

Dimension:

\[
\boxed{
n_z=8(M-1)+1.
}
\]

The polynomial coefficients are **not** nonlinear optimization variables.

MINCO maps

\[
\boxed{
(P_1,\dots,P_{M-1},T)
\longrightarrow
C.
}
\]

Keep the existing MINCO adjoint structure.

---

# 9. Generalize `MINCOQuintic` to N dimensions

Current `gcopter/minco.py` is mathematically dimension-independent but contains hard-coded `3` dimensions.

Refactor it once.

Infer

\[
D=\texttt{head\_pva.shape[1]}.
\]

All allocations and reshapes must use \(D\).

Examples:

```python
rhs = np.zeros((6 * pieces, D))
```

and

```python
blocks = coefficients.reshape(pieces, 6, D)
```

For ordinary GCOPTER,

\[
D=3.
\]

For this aerial-manipulator planner,

\[
\boxed{D=8.}
\]

Existing 3-D GCOPTER behavior and tests must remain unchanged.

Do not copy/fork `MINCOQuintic`.

---

# 10. Boundary conditions

Each pick/place event is a stop.

For one leg:

\[
\sigma(0)=\sigma_s,
\qquad
\dot\sigma(0)=\dot\sigma_s,
\qquad
\ddot\sigma(0)=\ddot\sigma_s,
\]

\[
\sigma(T)=\sigma_g,
\qquad
\dot\sigma(T)=0,
\qquad
\ddot\sigma(T)=0.
\]

For the first implementation it is acceptable to use zero initial arm velocity/acceleration and measured base translation/yaw derivatives where already available.

For the second leg:

\[
\sigma_s=\sigma_B,
\qquad
\sigma_g=\sigma_C
\]

with zero boundary derivatives after the grasp-settle interval.

---

# 11. Terminal pick/place state generation

V1 should use a **fixed 8-D terminal configuration** rather than an end-effector terminal manifold.

This simplifies both RRT-Connect and MINCO.

## 11.1 Inputs

For each event, configuration should provide:

- world grasp point \(p_\star\);
- nominal UAV yaw \(\psi_\star\);
- nominal 4-DoF arm posture \(q_a^{\rm nom}\);
- open/closed gripper gap used during evaluation.

## 11.2 Compute base position analytically

For the nominal yaw and arm posture, query the grasp-frame offset relative to a base at the origin:

\[
r_G
=
R_z(\psi_\star)\,
{}^Bp_G(q_a^{\rm nom}).
\]

Then choose

\[
\boxed{
p_B^\star=p_\star-r_G.
}
\]

Therefore

\[
p_G^W
=
p_B^\star+r_G
=
p_\star.
\]

The terminal planning state is

\[
\boxed{
\sigma_\star
=
[p_B^\star,\psi_\star,q_a^{\rm nom}].
}
\]

This is a deliberate V1 simplification.

Implement in:

```text
gcopter/aerial_manipulator/task_targets.py
```

Suggested API:

```python
def make_terminal_state(
    robot,
    target_position_ned: np.ndarray,
    nominal_yaw: float,
    nominal_joints: np.ndarray,
    *,
    gripper_opening: float,
) -> np.ndarray:
    """Return an 8-D [p, yaw, q_arm] terminal state."""
```

Validation:

1. nominal joints satisfy limits;
2. base target is inside workspace bounds;
3. reconstructed grasp-frame position error \(<10^{-6}\) m;
4. target configuration is collision-free except for any explicitly ignored target object.

A list of candidate nominal arm postures may optionally be supported.

General IK is not required for V1.

---

# 12. Kinematics used by the optimizer

All collision and terminal geometry gradients must use analytic kinematics.

## 12.1 End-effector position

Let

\[
p_G^B(q_a)
\]

be the grasp-frame position in the UAV body frame.

The world position is

\[
\boxed{
p_G^W
=
p_B+R_Bp_G^B(q_a).
}
\]

For the planner,

\[
R_B
=
R(\phi,\theta,\psi).
\]

For collision geometry in V1, use one of these two policies consistently:

### Preferred V1 policy

Use yaw-only geometry during optimization:

\[
R_B^{\rm geom}=R_z(\psi)
\]

and add conservative obstacle clearance to account for the actual roll/pitch during flight.

This keeps the collision gradient simple and fully analytic.

### Validation policy

Use the full simulated orientation in MuJoCo for final dense collision checking.

Do not mix these two definitions silently.

## 12.2 Serial-arm point Jacobian

For revolute joint \(j\), let:

- \(o_j\): world joint anchor;
- \(a_j\): world joint axis;
- \(p\): a downstream point.

Then

\[
\boxed{
\frac{\partial p}{\partial q_j}
=
a_j\times(p-o_j).
}
\]

If joint \(j\) is not an ancestor of the point body,

\[
\frac{\partial p}{\partial q_j}=0.
\]

The translational point Jacobian of an arbitrary robot-envelope point with respect to the 8-D planning state is

\[
\boxed{
J_p^\sigma
=
\begin{bmatrix}
I_3 &
\frac{\partial p}{\partial\psi} &
\frac{\partial p}{\partial q_1} &
\cdots &
\frac{\partial p}{\partial q_4}
\end{bmatrix}
\in\mathbb R^{3\times8}.
}
\]

Under yaw-only geometry,

\[
p
=
p_B+R_z(\psi)p^B(q_a).
\]

Therefore

\[
\boxed{
\frac{\partial p}{\partial\psi}
=
e_z\times
\left(
R_z(\psi)p^B
\right).
}
\]

Equivalently,

\[
\frac{\partial p}{\partial\psi}
=
R_z(\psi)
\left(
e_z\times p^B
\right).
\]

The arm columns use the revolute-joint cross-product formula.

Do not use finite differences.

---

# 13. Full physical dynamics

The robot is physically modeled as

\[
\boxed{
M(q)\dot\nu+h(q,\nu)
=
B(q)u+f_{\rm passive}(q,\nu)
}
\]

when there is no external contact.

The mass matrix contains base-arm coupling:

\[
M=
\begin{bmatrix}
M_{bb}&M_{ba}&\cdots\\
M_{ab}&M_{aa}&\cdots\\
\vdots&\vdots&\ddots
\end{bmatrix}.
\]

The term

\[
\boxed{M_{ba}}
\]

captures reaction of arm acceleration on the floating base.

The existing MuJoCo robot model already provides the numerical terms.

## 13.1 V1 rule

Do **not** place full coupled inverse-dynamics feasibility inside L-BFGS yet.

Use full dynamics for:

- simulation;
- controller feedforward;
- post-planning diagnostics;
- final saturation checks.

Use analytically differentiable simplified feasibility penalties inside the planner.

A future V2 may add:

\[
M(q)\dot\nu+h(q,\nu)-f_{\rm passive}
\in\operatorname{Range}(B)
\]

and actuator limits with exact/automatic derivatives.

---

# 14. Base flatness feasibility

Apply the existing GCOPTER point-mass flatness penalties to the first four planning dimensions.

Let

\[
p=[p_x,p_y,p_z]^T.
\]

At each quadrature sample obtain:

\[
p,\dot p,\ddot p,p^{(3)},p^{(4)},\psi,\dot\psi,\ddot\psi.
\]

Using the repository NED convention define

\[
f_d=
\begin{bmatrix}
-a_x\\
-a_y\\
g-a_z
\end{bmatrix}.
\]

Let

\[
\rho=\|f_d\|,
\qquad
b_3=\frac{f_d}{\rho}.
\]

Approximate total thrust:

\[
\boxed{
F=m_{\rm total}\rho.
}
\]

Use the complete aerial-manipulator mass.

Tilt:

\[
\boxed{
\theta
=
\arccos(e_3^Tb_3).
}
\]

Projected jerk:

\[
j_\perp
=
j-(b_3^Tj)b_3.
\]

Body-rate proxy:

\[
\boxed{
\|\omega_\perp\|^2
=
\frac{\|j_\perp\|^2}{\rho^2}.
}
\]

Reuse/refactor the analytic thrust, tilt, and body-rate gradients from the existing GCOPTER planner rather than creating numerical derivatives.

Yaw-rate and yaw-acceleration limits may additionally be applied directly:

\[
g_{\dot\psi}
=
\dot\psi^2-\dot\psi_{\max}^2,
\]

\[
g_{\ddot\psi}
=
\ddot\psi^2-\ddot\psi_{\max}^2.
\]

---

# 15. Arm feasibility penalties

Let

\[
q_a=\sigma_{4:8},
\qquad
\dot q_a=\dot\sigma_{4:8},
\qquad
\ddot q_a=\ddot\sigma_{4:8}.
\]

Use joint limits from the robot model.

## Position

\[
g_{q,j}^{+}
=
q_j-q_{j,\max},
\]

\[
g_{q,j}^{-}
=
q_{j,\min}-q_j.
\]

Gradients:

\[
\nabla_qg_{q,j}^{+}=e_j,
\]

\[
\nabla_qg_{q,j}^{-}=-e_j.
\]

## Velocity

\[
\boxed{
g_{\dot q,j}
=
\dot q_j^2-\dot q_{j,\max}^2.
}
\]

\[
\boxed{
\nabla_{\dot q}g_{\dot q,j}
=
2\dot q_je_j.
}
\]

## Acceleration

\[
\boxed{
g_{\ddot q,j}
=
\ddot q_j^2-\ddot q_{j,\max}^2.
}
\]

\[
\boxed{
\nabla_{\ddot q}g_{\ddot q,j}
=
2\ddot q_je_j.
}
\]

Torque limits are validation-only in V1.

---

# 16. Smooth inequality penalty

Reuse the existing GCOPTER smoothed positive-part penalty.

For

\[
g(y)\le0,
\]

define

\[
\phi_\epsilon(g)
=
\begin{cases}
0,&g\le0,\\
(\epsilon-\frac12g)(g/\epsilon)^3,&0<g\le\epsilon,\\
g-\frac12\epsilon,&g>\epsilon.
\end{cases}
\]

Its derivative is

\[
\phi_\epsilon'(g)
=
\begin{cases}
0,&g\le0,\\
(g/\epsilon)^2
\left[
-\frac12(g/\epsilon)
+
3(\epsilon-\frac12g)/\epsilon
\right],
&0<g\le\epsilon,\\
1,&g>\epsilon.
\end{cases}
\]

For weighted penalty

\[
L=w\phi_\epsilon(g(y)),
\]

the analytic gradient is

\[
\boxed{
\nabla_yL
=
w\phi_\epsilon'(g)\nabla_yg.
}
\]

---

# 17. ESDF environment representation

Environment collision avoidance inside MINCO should use ESDF.

Do not use MuJoCo `mj_geomDistance` as the primary trajectory-optimization collision gradient.

Create:

```text
uav_ac/planning/geometry/esdf.py
```

Suggested API:

```python
class ESDF:
    def distance(self, points: np.ndarray) -> np.ndarray:
        ...

    def distance_and_gradient(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        points: (N, 3)
        returns:
            distance: (N,)
            gradient: (N, 3)
        """
```

The class should know nothing about the aerial manipulator.

## 17.1 Grid query

For a query point inside one ESDF cell use trilinear interpolation.

Let local cell coordinates be

\[
u,v,w\in[0,1].
\]

Then

\[
D(p)
=
\sum_{a,b,c\in\{0,1\}}
W_{abc}(u,v,w)D_{i+a,j+b,k+c}.
\]

with

\[
W_{abc}
=
u^a(1-u)^{1-a}
v^b(1-v)^{1-b}
w^c(1-w)^{1-c}.
\]

Compute

\[
\boxed{
\nabla D(p)
}
\]

by analytic differentiation of the trilinear interpolation.

Do not finite-difference ESDF queries in the optimization loop.

The API must support batched point queries.

---

# 18. Whole-body collision approximation

Use simple smooth envelopes for optimization.

## 18.1 Environment collision

Represent the whole robot by a set of collision spheres.

Recommended V1 representation:

- several spheres for the UAV body/arms/rotor footprint;
- 2–5 spheres per serial-arm link;
- one or more spheres for the gripper;
- optional payload spheres during the carry leg.

Do not use a single giant whole-robot sphere.

For sphere \(s\):

- center \(p_s(\sigma)\);
- radius \(r_s\).

Safety condition:

\[
D(p_s)
\ge
r_s+d_{\rm safe}.
\]

Violation:

\[
\boxed{
g_s
=
r_s+d_{\rm safe}-D(p_s)
\le0.
}
\]

Because

\[
\frac{\partial D}{\partial\sigma}
=
J_s^T\nabla D,
\]

the violation gradient is

\[
\boxed{
\nabla_\sigma g_s
=
-J_s^T\nabla D(p_s).
}
\]

The collision penalty gradient is therefore

\[
\boxed{
\nabla_\sigma L_s
=
-w_{\rm obs}\,
\phi_\epsilon'(g_s)
J_s^T\nabla D(p_s).
}
\]

This is the primary whole-body obstacle gradient.

---

# 19. Sphere centers on serial links

For one link with two kinematic endpoints

\[
a(\sigma),\qquad b(\sigma),
\]

place fixed interpolation spheres using constants

\[
\alpha_s\in[0,1].
\]

Sphere center:

\[
\boxed{
p_s
=
(1-\alpha_s)a+\alpha_sb.
}
\]

Its Jacobian is

\[
\boxed{
J_s
=
(1-\alpha_s)J_a+\alpha_sJ_b.
}
\]

This gives a fully analytic configuration-space obstacle gradient.

Use enough spheres that uncovered gaps are smaller than the safety margin.

---

# 20. Self-collision model

V1 self-collision should be deliberately simple.

The main required self-collision check is:

\[
\boxed{
\text{UAV envelope}
\leftrightarrow
\text{arm envelope}.
}
\]

Arm-arm self-collision should primarily be prevented by conservative joint limits in V1. Non-adjacent arm-link sphere pairs may be added if needed.

For two robot spheres \(i,j\),

\[
d_{ij}
=
\|p_i-p_j\|-r_i-r_j.
\]

Require

\[
d_{ij}\ge d_{\rm self}.
\]

Define

\[
g_{ij}
=
r_i+r_j+d_{\rm self}
-\|p_i-p_j\|
\le0.
\]

Let

\[
n_{ij}
=
\frac{p_i-p_j}{\|p_i-p_j\|}.
\]

Then

\[
\boxed{
\nabla_\sigma g_{ij}
=
-(J_i-J_j)^Tn_{ij}.
}
\]

Ignore expected adjacent-link overlaps.

The exact pair list should be configured once, not rediscovered at every quadrature sample.

---

# 21. Payload collision

For leg \(B\rightarrow C\), add payload collision spheres rigidly attached to the grasp frame.

Let one payload sphere have fixed grasp-frame offset

\[
{}^Gp_o.
\]

Its world center is

\[
p_o^W
=
p_G^W+R_G\,{}^Gp_o.
\]

For V1, a position-only payload model is enough.

Its position Jacobian is derived from the grasp-frame Jacobian plus rotational contribution if the payload offset is nonzero.

If a single sphere is centered directly at the grasp frame,

\[
{}^Gp_o=0
\]

and

\[
J_o=J_G.
\]

Apply the same ESDF collision penalty.

This lets the carry planner account for payload size without introducing grasp/contact dynamics.

---

# 22. Exact MuJoCo collision role

The exact MuJoCo collision system remains valuable but has a different role.

Use it for:

- RRT state validation;
- RRT edge validation;
- terminal target validation;
- final dense trajectory validation;
- integration tests;
- simulator contact/collision reporting.

MINCO uses the differentiable ESDF/sphere approximation.

This gives:

\[
\boxed{
\text{exact collision for validity}
+
\text{smooth collision for optimization}.
}
\]

---

# 23. RRT-Connect front-end

Create a generic configuration-space RRT-Connect:

```text
uav_ac/planning/search/rrt_connect.py
```

Do not hard-code aerial-manipulator semantics inside it.

Suggested interface:

```python
planner = RRTConnect(
    lower_bounds=...,
    upper_bounds=...,
    distance_fn=...,
    interpolate_fn=...,
    state_valid_fn=...,
    edge_valid_fn=...,
    step_size=...,
    rng=...,
)

path = planner.plan(start, goal)
```

The output is an ordered array:

```text
(N, D)
```

For this task,

\[
D=8.
\]

---

# 24. 8-D RRT state

RRT state:

\[
\boxed{
x=
[p_x,p_y,p_z,\psi,q_1,q_2,q_3,q_4].
}
\]

Bounds:

- workspace bounds on \(p\);
- wrapped yaw interval;
- physical joint limits.

The gripper state is supplied separately to collision validation for each leg.

---

# 25. 8-D configuration metric

Do not use raw Euclidean distance because meters and radians have different scales.

Define

\[
\boxed{
d(x_a,x_b)^2
=
\frac{\|p_a-p_b\|^2}{s_p^2}
+
\frac{\Delta\psi^2}{s_\psi^2}
+
\sum_{j=1}^{4}
\frac{(q_{a,j}-q_{b,j})^2}{s_{q,j}^2}.
}
\]

with

\[
\Delta\psi
=
\operatorname{wrapToPi}(\psi_b-\psi_a).
\]

Recommended configurable scales:

```text
position_scale
yaw_scale
joint_scale[4]
```

These define the search geometry.

---

# 26. RRT interpolation

Interpolation must use the shortest wrapped yaw path.

For \(\alpha\in[0,1]\),

\[
p(\alpha)
=
p_a+\alpha(p_b-p_a),
\]

\[
\psi(\alpha)
=
\operatorname{wrapToPi}
\left(
\psi_a+\alpha\Delta\psi
\right),
\]

\[
q(\alpha)
=
q_a+\alpha(q_b-q_a).
\]

Return

\[
x(\alpha)\in\mathbb R^8.
\]

---

# 27. RRT steer

Given nearest state \(x_n\) and target \(x_t\), compute

\[
d=d(x_n,x_t).
\]

Use

\[
\alpha
=
\min
\left(
1,\frac{\Delta_{\rm step}}{d}
\right).
\]

Then

\[
x_{\rm new}
=
\operatorname{interpolate}(x_n,x_t,\alpha).
\]

---

# 28. RRT state validity

Convert the 8-D state to a complete public robot configuration:

1. base position from \(p\);
2. quaternion corresponding to yaw-only \(R_z(\psi)\);
3. four arm joints;
4. fixed leg-specific gripper opening.

Then validate:

- joint limits;
- workspace bounds;
- MuJoCo collision;
- payload collision if carrying.

For V1, MuJoCo exact collision is preferred for RRT state validity.

---

# 29. RRT edge validity

Endpoint validity is insufficient.

Interpolate the complete edge.

Choose maximum per-check changes:

\[
\Delta p_{\rm check},
\qquad
\Delta\psi_{\rm check},
\qquad
\Delta q_{\rm check}.
\]

For edge \(a\rightarrow b\), choose

\[
N
=
\max
\left(
\left\lceil
\frac{\|p_b-p_a\|_\infty}{\Delta p_{\rm check}}
\right\rceil,
\left\lceil
\frac{|\Delta\psi|}{\Delta\psi_{\rm check}}
\right\rceil,
\max_j
\left\lceil
\frac{|q_{b,j}-q_{a,j}|}{\Delta q_{\rm check}}
\right\rceil
\right).
\]

Check

\[
x(k/N),
\qquad
k=0,\dots,N.
\]

Every sample must be valid.

---

# 30. RRT-Connect algorithm

Implement the standard two-tree procedure.

Maintain:

```text
Tree A rooted at start
Tree B rooted at goal
```

Loop:

1. sample \(x_{\rm rand}\);
2. `EXTEND(TreeA, x_rand)`;
3. if an extension succeeds, repeatedly `CONNECT(TreeB, x_new)`;
4. if trees connect, reconstruct path;
5. swap trees;
6. continue until success or iteration limit.

Required statuses:

```python
TRAPPED
ADVANCED
REACHED
```

`CONNECT` repeatedly calls `EXTEND` toward the target until it is trapped or reached.

The path reconstruction must preserve the correct start-to-goal order regardless of which tree was swapped.

---

# 31. RRT path simplification

After RRT-Connect succeeds, greedily shortcut the path.

For current waypoint \(i\), find the farthest later waypoint \(j\) such that the direct 8-D edge is valid.

Keep only that waypoint and continue.

Then optionally resample the geometric path to obtain a reasonable number \(M+1\) of MINCO knot states.

Do not let the MINCO piece count explode with raw RRT nodes.

---

# 32. RRT path to MINCO initialization

Suppose the simplified/resampled path is

\[
X^0=
[x_0,x_1,\dots,x_M].
\]

Set:

\[
\sigma_s=x_0,
\qquad
\sigma_g=x_M,
\]

and initialize interior MINCO waypoints:

\[
\boxed{
P_k^0=x_k,\quad k=1,\dots,M-1.
}
\]

Initialize total time from a conservative geometric estimate.

For example:

\[
T_0
=
\max
\left(
\frac{L_p}{v_{\rm nominal}},
\frac{L_\psi}{\dot\psi_{\rm nominal}},
\max_j
\frac{L_{q,j}}{\dot q_{j,\rm nominal}}
\right)
\times s_T
\]

with safety factor

\[
s_T>1.
\]

Here:

\[
L_p=\sum_k\|p_{k+1}-p_k\|,
\]

\[
L_\psi=\sum_k|\operatorname{wrap}(\psi_{k+1}-\psi_k)|,
\]

\[
L_{q,j}=\sum_k|q_{k+1,j}-q_{k,j}|.
\]

---

# 33. Objective function

Use

\[
\boxed{
J
=
J_{\rm smooth}
+
\lambda_T T
+
J_{\rm dyn}
+
J_{\rm obs}
+
J_{\rm self}
+
J_{\rm ref}.
}
\]

`J_ref` should be optional and small.

---

# 34. Weighted jerk objective

Use a diagonal 8-D weight matrix

\[
W_J
=
\operatorname{diag}
(
w_p,w_p,w_p,
w_\psi,
w_{q1},w_{q2},w_{q3},w_{q4}
).
\]

Then

\[
\boxed{
J_{\rm smooth}
=
\int_0^T
\sigma^{(3)}(t)^T
W_J
\sigma^{(3)}(t)\,dt.
}
\]

For one piece with coefficients \(c_3,c_4,c_5\in\mathbb R^8\),

\[
\begin{aligned}
E_i
={}&
36c_3^TW_Jc_3h
+
144c_3^TW_Jc_4h^2\\
&+
\left(
192c_4^TW_Jc_4
+
240c_3^TW_Jc_5
\right)h^3\\
&+
720c_4^TW_Jc_5h^4
+
720c_5^TW_Jc_5h^5.
\end{aligned}
\]

Coefficient gradients:

\[
\boxed{
\frac{\partial E}{\partial c_3}
=
72W_Jc_3h
+
144W_Jc_4h^2
+
240W_Jc_5h^3
}
\]

\[
\boxed{
\frac{\partial E}{\partial c_4}
=
144W_Jc_3h^2
+
384W_Jc_4h^3
+
720W_Jc_5h^4
}
\]

\[
\boxed{
\frac{\partial E}{\partial c_5}
=
240W_Jc_3h^3
+
720W_Jc_4h^4
+
1440W_Jc_5h^5.
}
\]

Direct segment-time gradient:

\[
\boxed{
\begin{aligned}
\frac{\partial E}{\partial h}
={}&
36c_3^TW_Jc_3\\
&+
288c_3^TW_Jc_4h\\
&+
\left(
576c_4^TW_Jc_4
+
720c_3^TW_Jc_5
\right)h^2\\
&+
2880c_4^TW_Jc_5h^3\\
&+
3600c_5^TW_Jc_5h^4.
\end{aligned}
}
\]

Generalize `MINCOQuintic.jerk_energy()` to support a dimension weight vector.

---

# 35. Optional RRT-reference regularizer

To keep early optimization near the safe RRT homotopy, use

\[
\boxed{
J_{\rm ref}
=
\lambda_{\rm ref}
\sum_{k=1}^{M-1}
(P_k-P_k^0)^TW_R(P_k-P_k^0).
}
\]

Gradient:

\[
\boxed{
\frac{\partial J_{\rm ref}}{\partial P_k}
=
2\lambda_{\rm ref}
W_R(P_k-P_k^0).
}
\]

This is optional and should be easy to disable.

---

# 36. Quadrature

Reuse the existing GCOPTER integrated-penalty pattern.

For each segment, use fixed normalized samples

\[
\alpha_j\in[0,1].
\]

Physical local time:

\[
t_j=\alpha_jh.
\]

Use trapezoidal weights or the same quadrature rule already used by GCOPTER.

At each sample evaluate:

\[
\sigma,
\dot\sigma,
\ddot\sigma,
\sigma^{(3)},
\sigma^{(4)}.
\]

Batch evaluations where possible.

---

# 37. Gradient from sampled state to polynomial coefficients

Let

\[
\sigma=C^T\beta_0,
\]

\[
\dot\sigma=C^T\beta_1,
\]

\[
\ddot\sigma=C^T\beta_2,
\]

\[
\sigma^{(3)}=C^T\beta_3.
\]

For a sample cost \(L\), define:

\[
g_0=\frac{\partial L}{\partial\sigma},
\]

\[
g_1=\frac{\partial L}{\partial\dot\sigma},
\]

\[
g_2=\frac{\partial L}{\partial\ddot\sigma},
\]

\[
g_3=\frac{\partial L}{\partial\sigma^{(3)}}.
\]

Then

\[
\boxed{
\frac{\partial L}{\partial C}
=
\beta_0g_0^T
+
\beta_1g_1^T
+
\beta_2g_2^T
+
\beta_3g_3^T.
}
\]

This is the central analytic chain rule.

The integrated quadrature contribution multiplies this by the physical quadrature scale.

---

# 38. Collision gradient chain

For one environment sphere:

\[
g_s
=
r_s+d_{\rm safe}-D(p_s(\sigma)).
\]

Then

\[
\frac{\partial g_s}{\partial\sigma}
=
-J_s^T\nabla D.
\]

Penalty:

\[
L_s
=
w_{\rm obs}\phi_\epsilon(g_s).
\]

Therefore:

\[
\boxed{
\frac{\partial L_s}{\partial\sigma}
=
-w_{\rm obs}
\phi_\epsilon'(g_s)
J_s^T\nabla D.
}
\]

Then:

\[
\boxed{
\frac{\partial L_s}{\partial C}
=
\beta_0
\left(
\frac{\partial L_s}{\partial\sigma}
\right)^T.
}
\]

No finite difference is needed anywhere in this chain.

---

# 39. Sample-time derivative

Because

\[
t_j=\alpha_jh,
\]

sample state also changes when \(h\) changes.

For a sample cost

\[
L(\sigma,\dot\sigma,\ddot\sigma,\sigma^{(3)}),
\]

the state evolution contribution is

\[
\boxed{
\frac{dL}{dt}
=
g_0^T\dot\sigma
+
g_1^T\ddot\sigma
+
g_2^T\sigma^{(3)}
+
g_3^T\sigma^{(4)}.
}
\]

Thus

\[
\boxed{
\frac{\partial L(t_j)}{\partial h}
\supset
\alpha_j\frac{dL}{dt}.
}
\]

The quadrature scale itself also contains \(h\), so include the direct integration-weight derivative exactly as the existing GCOPTER `_integrated_penalty()` does.

Reuse/refactor the current logic rather than approximating this derivative.

---

# 40. MINCO adjoint propagation

The evaluator accumulates:

- gradient w.r.t. polynomial coefficients;
- direct gradient w.r.t. each segment duration.

Call the generalized MINCO adjoint propagation to obtain:

\[
\frac{\partial J}{\partial P_k}
\]

and per-piece

\[
\frac{\partial J}{\partial h_i}.
\]

Since all

\[
h_i=T/M,
\]

combine:

\[
\boxed{
\frac{\partial J}{\partial T}
=
\frac{1}{M}
\sum_i
\frac{\partial J}{\partial h_i}.
}
\]

Then backpropagate through \(T=\mathcal T(\tau)\).

---

# 41. `evaluator.py` responsibility

`planner.py` should remain thin.

`evaluator.py` owns the numerical objective.

Suggested structure:

```python
class AerialManipulatorTrajectoryEvaluator:
    def evaluate(
        self,
        coefficients,
        total_time,
        ...
    ) -> tuple[float, np.ndarray, float]:
        """
        Return:
            cost
            grad_coefficients
            grad_total_time_direct
        """
```

Internally:

1. evaluate all polynomial samples;
2. split base/yaw/arm states;
3. evaluate base dynamic feasibility;
4. evaluate arm limits;
5. evaluate ESDF obstacle penalties;
6. evaluate self-collision penalties;
7. accumulate analytic state gradients;
8. map them to coefficient/time gradients.

Do not place all of this in `planner.py`.

---

# 42. `collision.py` responsibility

`gcopter/aerial_manipulator/collision.py` owns:

- sphere definitions;
- kinematic sphere-center evaluation;
- analytic point Jacobians;
- environment ESDF penalty;
- UAV-arm self-collision penalty;
- payload sphere handling.

It should not know anything about L-BFGS or MINCO coefficients.

Input conceptually:

\[
\sigma
\]

Output:

- collision cost;
- \(\partial L/\partial\sigma\).

---

# 43. `planner.py` responsibility

`planner.py` owns orchestration only:

```text
validate start/goal
        ↓
RRT-Connect
        ↓
shortcut / resample
        ↓
initialize MINCO knots + total time
        ↓
encode nonlinear variables
        ↓
L-BFGS objective callback
        ↓
MINCO solve
        ↓
evaluator
        ↓
adjoint propagation
        ↓
decode optimized trajectory
        ↓
dense validation
```

Suggested public API:

```python
class AerialManipulatorGCOPTER:
    def plan(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        *,
        robot,
        esdf,
        gripper_opening: float,
        payload=None,
        rng=None,
    ) -> AerialManipulatorTrajectory:
        ...
```

---

# 44. Result type

Create a dedicated result type.

Suggested fields:

```python
@dataclass(frozen=True)
class AerialManipulatorTrajectory:
    durations: np.ndarray        # shape (M,), all identical
    coefficients: np.ndarray     # shape (M, 6, 8)
    total_time: float
    rrt_path: np.ndarray         # shape (N, 8)
    cost: float
    iterations: int
    converged: bool
    message: str
```

Useful methods:

```python
sample(t)
sample_many(times)
to_controller_references(...)
```

---

# 45. Convert MINCO sample to full robot reference

For an 8-D sample:

\[
\sigma=[p,\psi,q_a],
\]

\[
\dot\sigma=[\dot p,\dot\psi,\dot q_a],
\]

\[
\ddot\sigma=[\ddot p,\ddot\psi,\ddot q_a].
\]

Recover UAV roll/pitch/quaternion from the quadrotor flatness map.

Construct:

\[
q_{\rm full}
=
[p,\;q_B,\;q_a,\;g].
\]

Construct public velocity:

\[
\nu
=
[v_B,\omega_B,\dot q_a,\dot g].
\]

Set

\[
\dot g=0
\]

during a motion leg.

Construct compatible acceleration for `AerialManipulatorReference`.

Reuse existing coordinate conventions and controller APIs. Do not introduce a second NED/ENU convention.

---

# 46. Pick/place state machine

Create:

```text
uav_ac/tasks/aerial_pick_place.py
```

Keep it simple.

Suggested states:

```python
PLAN_TO_PICK
MOVE_TO_PICK
GRASP
PLAN_TO_PLACE
MOVE_TO_PLACE
RELEASE
DONE
FAILED
```

If both legs are preplanned, `PLAN_TO_PLACE` may only select the second stored trajectory.

---

# 47. Event conditions

At the pick event require:

\[
\|p_G-p_{\rm pick}\|
\le
\epsilon_p
\]

The position here is the world position of the graspable object. It is not the
base position computed by `make_terminal_state()`.

and preferably:

\[
\|v_G\|
\le
\epsilon_v.
\]

The grasp-frame linear velocity is

\[
v_G
=
J_G\nu.
\]

In code, query the world-frame grasp linear velocity as
`robot.jacobian(frame="grasp")[:3] @ robot.velocity`. The public Jacobian has
six spatial rows and eleven tangent-coordinate columns.

When satisfied:

1. command gripper closed;
2. wait a configurable settle interval;
3. set `holding=True`;
4. begin the carry trajectory.

At place:

\[
\|p_G-p_{\rm place}\|
\le
\epsilon_p
\]

and low velocity.

Then:

1. open gripper;
2. wait settle interval;
3. set `holding=False`;
4. enter `DONE`.

Do not trajectory-optimize the closure/opening phase.

---

# 48. Payload attachment in V1

The planning state machine may use a logical payload attachment.

The minimum requirement is:

- before grasp: payload fixed at world pick point;
- after grasp: payload collision envelope moves rigidly with the grasp frame;
- after release: payload fixed at world place point.

For the first demo, it is acceptable for visualization/simulation attachment to use a simple weld/equality or kinematic pose update if that fits the existing MuJoCo architecture.

Do not let payload-attachment engineering block the planner.

---

# 49. Dense final validation

After optimization, sample the complete trajectory more densely than the optimizer quadrature.

Validate:

- MuJoCo exact collision;
- self collision;
- workspace bounds;
- joint position limits;
- joint velocity limits;
- joint acceleration limits;
- base speed/acceleration limits;
- thrust limits;
- tilt limits;
- body-rate proxy;
- terminal grasp-frame error;
- controller reference finiteness.

For the carry leg also validate payload clearance.

A trajectory that optimizes successfully but fails dense validation must not be returned as valid.

---

# 50. Optional full-dynamics post-check

For each dense sample, query:

\[
M(q),h(q,\nu),f_{\rm passive},B(q).
\]

Construct approximate desired public acceleration.

Record:

- required arm torques from the existing inverse-dynamics relationship;
- rotor/control saturation indicators if available;
- maximum base-arm coupling terms;
- tracking feasibility diagnostics.

This is validation only in V1.

Do not add numerical dynamic derivatives to L-BFGS.

---

# 51. Configuration proposal

Create a dedicated config dataclass.

Example structure:

```python
@dataclass(frozen=True)
class AerialManipulatorGCOPTERConfig:
    pieces: int = 8

    time_weight: float = ...
    position_jerk_weight: float = ...
    yaw_jerk_weight: float = ...
    arm_jerk_weights: tuple[float, float, float, float] = ...

    max_velocity: float = ...
    max_acceleration: float = ...
    max_yaw_rate: float = ...
    max_yaw_acceleration: float = ...

    joint_velocity_limits: tuple[...] = ...
    joint_acceleration_limits: tuple[...] = ...

    obstacle_clearance: float = ...
    self_clearance: float = ...
    obstacle_weight: float = ...
    self_collision_weight: float = ...

    integral_resolution: int = 12
    max_iterations: int = ...
    lbfgs_memory: int = ...

    rrt_step_size: float = ...
    rrt_max_iterations: int = ...
    rrt_position_scale: float = ...
    rrt_yaw_scale: float = ...
    rrt_joint_scale: tuple[...] = ...

    edge_position_resolution: float = ...
    edge_yaw_resolution: float = ...
    edge_joint_resolution: float = ...

    reference_weight: float = ...
```

Use robot-model joint bounds as source of truth rather than duplicating them in YAML.

---

# 52. YAML task configuration

Example concept:

```yaml
task: aerial_pick_place
scene: aerial_manipulator_pick_place
planner: aerial_manipulator_gcopter
controller: cascaded
control_dt: 0.01
seed: 7
visualize: true

pick_place:
  pick_position_ned: [ ... ]
  place_position_ned: [ ... ]

  pick_yaw: 0.0
  place_yaw: 0.0

  pick_nominal_joints: [ ... ]
  place_nominal_joints: [ ... ]

  gripper_open: 0.060
  gripper_closed: 0.025

  position_tolerance: 0.02
  velocity_tolerance: 0.05
  settle_time: 0.30

aerial_manipulator_gcopter:
  pieces: 8
  ...
```

Fit this into the repository's existing config-validation style rather than bypassing it.

---

# 53. Testing requirements

## 53.1 MINCO N-D regression

Verify:

- current 3-D GCOPTER still passes;
- 8-D coefficient solve has correct shapes;
- continuity across segment boundaries;
- endpoint PVA constraints;
- weighted jerk gradient against finite difference **in tests only**.

Finite differences are allowed as gradient tests, not runtime implementation.

## 53.2 ESDF

Test trilinear distance and analytic gradient against finite-difference reference.

Use simple synthetic fields where the exact answer is known.

## 53.3 Kinematic sphere Jacobian

For several random valid arm states compare analytic sphere-center Jacobian to finite differences.

Again: tests only.

## 53.4 Collision gradient

For a simple synthetic ESDF compare:

\[
\nabla_\sigma L_{\rm obs}
\]

to finite difference.

## 53.5 RRT-Connect

Test:

- direct connection;
- obstacle detour;
- correct start/goal ordering after tree swaps;
- wrapped yaw interpolation across \(-\pi/\pi\);
- edge collision detection between safe endpoints;
- deterministic result for fixed seed.

## 53.6 Full planner

For a simple scene require:

- RRT returns valid path;
- optimizer decreases or does not materially worsen objective;
- dense exact collision validation passes;
- terminal state is reached;
- grasp-frame terminal position error below tolerance.

## 53.7 Integration task

Require:

- reaches pick;
- closes gripper;
- reaches place;
- opens gripper;
- no exact collision;
- no NaN/Inf;
- no joint-limit violation.

---

# 54. Implementation order for Codex

Follow this order. Do not implement everything simultaneously.

## Phase 1 — MINCO core cleanup

1. Generalize `MINCOQuintic` from 3-D to N-D.
2. Add optional per-dimension jerk weights.
3. Preserve all existing GCOPTER behavior/tests.
4. Add N-D unit tests.

## Phase 2 — generic RRT-Connect

1. Implement callback-based `RRTConnect`.
2. Add wrapped-angle-compatible interpolation through callbacks.
3. Implement shortcut simplification.
4. Add standalone tests.

## Phase 3 — ESDF

1. Add ESDF data structure.
2. Add batched trilinear distance query.
3. Add analytic batched gradient.
4. Add synthetic tests.

## Phase 4 — aerial-manipulator geometry evaluator

1. Define UAV/arm/gripper collision spheres.
2. Implement sphere-center FK.
3. Implement analytic 8-D point Jacobians.
4. Add ESDF obstacle penalty.
5. Add UAV-arm sphere self-collision.
6. Add payload spheres.

## Phase 5 — 8-D trajectory evaluator

1. Sample 8-D MINCO.
2. Add weighted jerk/time.
3. Add joint feasibility.
4. Reuse base flatness penalties.
5. Add collision gradients.
6. Implement coefficient/time accumulation.

## Phase 6 — planner

1. Build fixed terminal state.
2. Run 8-D RRT-Connect.
3. Shortcut and resample.
4. Initialize MINCO.
5. Optimize interior points + one scalar total time.
6. Dense validate.
7. Return trajectory object.

## Phase 7 — task execution

1. Build pick/place state machine.
2. Convert trajectory to `AerialManipulatorReference`.
3. Execute first leg.
4. Trigger grasp.
5. Execute second leg.
6. Trigger release.
7. Add integration test.

---

# 55. Important architectural rules

### Rule 1

Do not put planner-specific penalty weights or MINCO logic into:

```text
robot/aerial_manipulator/
```

### Rule 2

Do not duplicate FK/dynamics.

### Rule 3

Do not add finite-difference gradients to runtime optimization.

### Rule 4

The optimizer uses smooth approximate collision geometry; MuJoCo remains the exact validator.

### Rule 5

The first planner is exactly 8-D:

\[
\boxed{
[p_x,p_y,p_z,\psi,q_1,q_2,q_3,q_4].
}
\]

### Rule 6

Each segment is quintic.

### Rule 7

All segment times are equal:

\[
\boxed{
h=T/M.
}
\]

Only one total-time variable is optimized.

### Rule 8

A→B and B→C are independent MINCO optimization problems.

### Rule 9

Grasp/release is an event/state-machine operation, not part of trajectory optimization.

### Rule 10

Do not over-generalize V1. Get one complete pick-and-place pipeline working first.

---

# 56. Mathematical summary

The continuous trajectory is

\[
\boxed{
\sigma(t)
=
[p(t),\psi(t),q_a(t)]
\in\mathbb R^8.
}
\]

The optimizer is

\[
\boxed{
\min_{\{P_k\},T}
\;
J_{\rm jerk}
+
\lambda_TT
+
J_{\rm dyn}
+
J_{\rm obs}
+
J_{\rm self}
+
J_{\rm ref}
}
\]

subject implicitly/through penalties to:

\[
q_{\min}\le q_a(t)\le q_{\max},
\]

\[
|\dot q_a(t)|\le\dot q_{\max},
\]

\[
|\ddot q_a(t)|\le\ddot q_{\max},
\]

\[
\|v_B(t)\|\le v_{\max},
\]

\[
\|a_B(t)\|\le a_{\max},
\]

\[
F_{\min}\le F(t)\le F_{\max},
\]

\[
\theta(t)\le\theta_{\max},
\]

\[
D(p_s(\sigma(t)))
\ge
r_s+d_{\rm safe},
\]

and

\[
\|p_i(\sigma)-p_j(\sigma)\|
\ge
r_i+r_j+d_{\rm self}
\]

for selected self-collision pairs.

The nonlinear variables are only:

\[
\boxed{
8(M-1)+1.
}
\]

MINCO analytically reconstructs the complete quintic trajectory.

The essential collision gradient chain is

\[
\boxed{
\nabla D
\rightarrow
J_{\rm point}^{T}\nabla D
\rightarrow
\nabla_\sigma L
\rightarrow
\nabla_C L
\rightarrow
\nabla_P L.
}
\]

This chain must remain explicit and testable.

---

# 57. Acceptance criteria

The milestone is complete only if all of the following are true:

1. Existing ordinary GCOPTER tests still pass.
2. A generic 8-D RRT-Connect finds a collision-free A→B and B→C path.
3. The aerial-manipulator MINCO trajectory is 8-D.
4. Every piece is quintic.
5. Every piece has duration \(T/M\).
6. The optimizer has only one total-time scalar per leg.
7. ESDF obstacle gradients are analytic.
8. Robot point Jacobians are analytic.
9. Self-collision gradients are analytic for the envelope model.
10. Runtime optimization uses no finite differences.
11. Dense MuJoCo collision validation passes.
12. The grasp frame reaches B within tolerance.
13. The state machine closes the gripper.
14. The carry trajectory reaches C without collision.
15. The state machine releases the payload.
16. The complete integration test runs deterministically under a fixed seed.

Once this version is stable, future work can add full coupled inverse-dynamics feasibility, optimized end-effector terminal manifolds, contact, and more general whole-body planning without changing the high-level architecture.

---

# 58. V1 implementation notes and corrections

The repository now includes the V1 planning and execution path described above.
Run it with:

```bash
python -m uav_ac.main --config configs/aerial_manipulator_pick_place.yaml
```

The implementation keeps the existing `main.py` entry point and cascaded
flight/arm controller. It preplans both legs before execution, uses an 8-D
RRT-Connect path followed by equal-duration quintic MINCO and L-BFGS, and
returns optimizer convergence, dense trajectory validation, and task execution
status separately. The 8-D MINCO implementation also supports the ordinary
3-D GCOPTER path. `robot.point_positions_and_jacobians(...)` is the shared
batched query for local points on arbitrary links; its Jacobians are expressed
in the planner's eight tangent coordinates. Pure kinematic queries can opt out
of joint-limit rejection with `check_limits=False`; execution and final checks
retain strict limits.

V1 builds a signed-distance grid from static axis-aligned boxes and the ground
plane, then uses trilinear interpolation with analytic gradients. Unknown
out-of-grid space receives an optimization penalty and is rejected during
final validation. Optimizer collision spheres use yaw-only link geometry plus
an explicit tilt allowance, while final checks use the robot's full pose
geometry. The logical payload is a marker in MuJoCo and a carried sphere in
planning; it does not change robot mass or simulate grasp contact. The
specified gripper-payload contact is allowed while payload collisions with the
robot and environment remain checked.

Yaw differences use the shortest angular distance for RRT steering. The path is
unwrapped before MINCO fitting, and optimized yaw values are not wrapped
pointwise. MINCO preserves its internal continuity conditions; zero terminal
velocity and acceleration do not imply zero jerk or zero body angular rate, so
the state machine waits for the measured position, end-effector speed, arm
configuration, and body rate to remain stable before grasping or releasing.
References update at `control_dt`, while the controller runs each physics
step. On arrival, the terminal reference is held while these event conditions
settle. Reference quaternions are signed to remain continuous with the current
robot attitude; linear velocity is in world NED and angular velocity is in
body FRD coordinates.

Dense MuJoCo validation checks piece boundaries and interior samples at the
reported `validation_sample_dt`, and reports minimum clearance and maximum
constraint violation. This is a sampled geometric check, not a continuous-time
collision certificate; the simulated execution also checks collisions at each
physics step.

The deterministic headless integration test exercises the complete A→B→C
logical pick-and-place loop. A successful run requires both plans to pass
validation, the gripper events to complete, and the released marker to end
within 2 cm of C. Optimizer convergence is reported independently because a
nonconverged iterate may still be feasible, while no trajectory that fails
validation is executed.
