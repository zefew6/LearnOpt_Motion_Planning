# Whole-body YOPO staged repair

The three stages are ordered. A trained checkpoint is an experiment artifact;
it is promoted only after its stage gate passes. The forest flight presets keep
their previous checkpoint paths until a replacement qualifies.

The subsequent user-approved redesign supersedes the objective below for new
training: all tracked training presets now use [yopo_minco_v1](GUIDANCE_MINCO.md),
without stopping or gyro-matching losses. The experiments documented here used
the previous objective; their checkpoints/reports retain those semantics.

## 1. Candidate initialization and guidance gradients

The shared prediction head now uses ordinary nonzero convolution initialization.
The 27 candidates retain a calibrated 3×3 primitive grid and three arm modes.
Independent terminal outputs remain learnable; there is no endpoint separation
loss or requirement that all candidates finish in distinct regions.

The audit checks every physical parameter against central finite differences,
reports separate environment/self/physical/jerk/progress/stopping contributions,
and compares the actual neural backward with oracle gradients. It additionally
finite-differences the raw terminal-position channels through the decoder, which
detects an accidentally detached endpoint. For a saved model, the audit reads
the objective settings from its checkpoint metadata.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.audit \
  --dataset runs/yopo_whole_body/staged_repair_v1/curriculum_closed_loop_states/dataset \
  --checkpoint <checkpoint-directory> --samples 3 --output <new-report.json>
```

The old `primitive_v4` terminal is a bounded camera-FOV point. The new,
versioned `primitive_goal_v5` terminal is a bounded body-origin ball, expressed
in each primitive frame. It includes zero displacement and reverse motion,
needed for holonomic eight-dimensional reaching and stopping. Its intermediate
point keeps the 3×3 direction constraint. This extends the YOPO-inspired
parameterization; it is not an exact reproduction of YOPO-MINCO. Schema 1/2
models retain their old decoding; schema 3 records the new profile and bounds.

Near the goal, the guidance objective adds a terminal-rate cost, including arm
rates. The legacy default is zero; curriculum presets set weight 50 and an
activation distance of 1.5 m.

Curriculum training uses `collision_model: exact_geometry`. Authored planning
spheres penalized some poses accepted by actual robot geometry, including the
fully folded witness. Exact geometry reuses the existing actual-geometry
broad/narrow phase and analytic Jacobian derivatives; deep penetrations retain
its ESDF-envelope descent fallback. Old recipes retain their
`planning_spheres` default. Neither collision model is available to deployment
as an online candidate filter.

## 2. Simple-scene curriculum and real tracking

The families are `free`, `obstacle`, `low` and `wall`. The latter two constrain
altitude and require arm clearance along the authored feasibility witness.
Small scenes are constructed by `YOPO/simple_scenes.py`; generated layouts
are scene-disjoint. Feasibility witnesses are separate checksummed privileged
resources, not network inputs or training trajectory labels.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo/dataset/yopo_curriculum.yaml \
  --run-dir runs/yopo_whole_body/<new-dataset-run>

MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.training \
  --config configs/yopo/train/yopo_stage_free.yaml \
  --dataset runs/yopo_whole_body/<new-dataset-run>/dataset \
  --run-dir runs/yopo_whole_body/<new-free-run>

MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.acceptance \
  --dataset runs/yopo_whole_body/<new-dataset-run>/dataset \
  --checkpoint runs/yopo_whole_body/<new-free-run>/best_200 \
  --families free --output <new-gate-report.json>
```

Each sample is one independently sampled current state and one depth frame.
Positions along the route, height, folded arm states, velocity, acceleration,
jerk and angular-rate mismatch vary. There is no recurrent observation or
training-time replanning. Deployment repeatedly plans from the measured state.

Once free passes, warm-start the next preset using `--warm-start <checkpoint>`.
The presets add families cumulatively, preserving previously learned cases.
Warm start resets optimizer/calibration; `--resume` instead preserves optimizer
and RNG and requires matching training contracts.

Acceptance requires at least 90% held-out selected dense feasibility per
requested family, actual MuJoCo/controller reaching without collision in every
canonical scene, and maximum handoff body-rate discrepancy ≤0.5 rad/s. Dense
validation is a sampled test, not a continuous safety certificate. The headless
gate runs actual physics and the controller, not trajectory playback; it does
not verify a graphical viewer. Interactive runs continue through YAML → main.

Earlier experiments under `runs/yopo_whole_body/staged_repair_v1/` are retained.
`free_goal_v5` achieved 16/16 dense-valid selected trajectories but failed the
real tracking test with a workspace violation at 5.85 s. It must not be promoted.
`free_exact` completed 200 steps with corrected collision supervision. Its
training cost fell from 114.18 to 105.39 (7.7%), proxy selected validity reached
100%, and exact selected validation passed 16/16 samples. Actual tracking still
failed at 7.09 s with a workspace violation: position error 1.313 m, maximum
joint error to the goal 1.304 rad. Maximum tracking error was only 6.5 mm and
handoff body-rate discrepancy was 0.323 rad/s. The controller closely followed
a plan that did not converge to the goal. Stage 2 is **not passed**; obstacle,
low and wall training and stage 3 have not been promoted or started.

The recorded `plan_trace.json` shows terminal reach staying around 1.5 m
forward after the actual vehicle passes its goal, along with accumulating
joint offset. The current sampler never places the vehicle beyond the goal
(current x≤2.6, goal x≥2.7), and its neutral/folded joint mixture omits many
opposite-direction joint deviations seen online. The next correction must
cover those states and verify learned target sensitivity and stopping before
training the remaining families. These coverage gaps are confirmed;
inadequate conditional learning remains a separate convergence issue.

For this checkpoint, actual neural backward checks passed in all three audit
samples. The strict full physical finite-difference gate did **not** pass on
every collision-heavy candidate. The step sweep in `fd_step_probe.log` shows
small-step narrow-phase numerical sensitivity; a larger step also crosses
piecewise collision/ESDF branches. Both failed reports are retained as
`gradient_audit.json` and `gradient_audit_step1e4.json`; this is not a claim that
all new collision gradients are globally verified. The isolated nonzero wall
penetration gradient regression passes. Complete the collision-heavy audit
before promoting this new objective to the other narrow-scene families.

A separate controller-isolation probe replanned a goal-directed MINCO from the
same measured state. It reached in 15.48 s with 1.2 mm final position error, but
its maximum handoff body-rate discrepancy was 0.666 rad/s. This is diagnostic
evidence that the controller can reach in this scene; it is neither a learned
model nor a passing promotion result. The corresponding summary is saved as
`runs/yopo_whole_body/staged_repair_v1/controller_isolation.json`.

## 3. Dense forest/factory training

Start only when all four simple families pass together. Expand independent
layouts and current-state coverage, train a balanced dense-scene recipe, and
repeat held-out exact validation plus actual online controller tracking on the
forest/factory benchmark. A successful training loss reduction alone does not
qualify a checkpoint. If stage 2 fails, preserve its reports and diagnose the
specific planning/tracking discrepancy before enlarging the environment.
