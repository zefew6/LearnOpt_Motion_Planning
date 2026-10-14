# Local whole-body YOPO-MINCO guidance

All tracked `configs/yopo/train/*.yaml` now select `guidance_profile:
yopo_minco_v1`. This recipe excludes terminal stopping and initial gyro-matching
costs. Data generation, network parameterization, scene geometry, controllers
and flight checkpoint paths are unchanged. Each training item is one current
state and depth image. Online replanning is a deployment operation.

## Reference and adaptations

The reference is official YOPO-MINCO revision
[`d5e1e60`](https://github.com/TJU-Aerial-Robotics/YOPO/tree/d5e1e60f83cc991602f142d9b6b96204a24d6b9f), especially
[`safety_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/safety_loss.py),
[`guidance_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/guidance_loss.py),
[`feasible_loss.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/loss/feasible_loss.py) and
[`yopo_trainer.py`](https://github.com/TJU-Aerial-Robotics/YOPO/blob/d5e1e60f83cc991602f142d9b6b96204a24d6b9f/YOPO/policy/yopo_trainer.py).

This is a whole-body adaptation, not a literal quadrotor reproduction. Actual
robot geometry supplies privileged signed pair distances and analytic witness
Jacobians, with conservative geometry covers pruning far environment pairs.
All queries use the robot's public scratch-state interfaces. Static planning
spheres do not decide contact. The existing ESDF/context remains available for
offline trajectory validation. No privileged geometry enters network inputs
or deployment candidate selection.

The point-robot free-ball-chain certificate is not applied to articulated body
motion. Contact classification uses 17 samples per piece including endpoints;
it is a discrete whole-body check, not a continuous safety certificate. Safe
corridor prediction, geodesic guidance, heading-gated speed shaping and the
optional inner-radius regularizer are outside this four-component baseline.
The upstream near-goal stopping term is explicitly disabled by user request.

## Trajectory cost

The cost is safety + smoothness/acceleration energy + time + feasibility +
goal progress. No `goal_stop` or `continuity_and_peak` component exists in this
recipe. Every weight and threshold is configurable in the `guidance` YAML
block, validated strictly, and expanded into saved checkpoint metadata.

Safety uses shifted exponential barriers on the minimum environment and valid
self-pair distances at each sample. Environment defaults are activation gap
0.15 m, exponential origin 0.05 m and decay 0.05 m. Self activation/decay are
0.015 m. These are explicit whole-body baseline parameters, not the upstream
6 m/s point-robot clearances. An exponent cap of 50 bounds already-invalid
initial-contact costs. Contact means signed distance <=0, independently of
barrier activation or its weight.

For collision-free trajectories, the aggregate is
`(logsumexp(beta*c)-log(M))/beta`, beta=4. For a contacting trajectory it is the
cost at the last sample before first contact (sample zero if already in
contact). There is one focused safety cost; no duplicate safety integral or
additional duration-scaled peak is included. Safety diagnostics allocate the
aggregate to environment/self contributions without changing the total.

Smoothness is exact eight-dimensional minimum-jerk energy using the existing
per-axis weights, multiplied by 0.002. Acceleration energy is an exact
polynomial integral, weighted by 0.002 after position/yaw/joint scale
normalization. Time weight is 1.5. Gradients pass through both waypoint and
free-terminal PVA, and both segment durations.

Physical feasibility sums each constraint's peak positive normalized
overshoot: base speed/acceleration/jerk, yaw rate/acceleration, joint
position/rate/acceleration, thrust, tilt, flatness body rate, and workspace
bounds. Joint constraints and thrust/attitude constraints are whole-body
extensions; this does not certify full coupled actuator-torque feasibility.
Weight is 1.5. The base curvature cap follows the upstream speed-only gradient:
curvature determines a detached speed limit, so its gradient slows motion
without changing path curvature. There is no target-distance speed penalty.

Goal position cost uses parallel/perpendicular vector Smooth-L1 relative to
the start-to-goal ray; perpendicular weight is 0.3 and total goal weight 0.6.
Wrapped yaw and four joint Smooth-L1 preferences extend the target to eight
variables (scales 0.7 rad and 0.8 rad). These are soft preferences, not equality
constraints on every local endpoint. For a contacting candidate, the goal
value remains available for ranking but its entire goal gradient is stopped.

## Generation, selection and gradient audits

For 27-candidate recipes, the positive score `s=softplus(raw_score)` is used
consistently for cross-entropy and detached generation weights. Each image's
candidate costs are normalized by their detached mean. The classification
label is the least-cost collision-free candidate, or the least-cost candidate
when all contact. Physical limits affect the total cost, not the collision
label. Deployment's argmax of raw scores is equivalent to argmax of positive
scores, so weights/network architecture and inference contracts need no
migration. The single-candidate P3 ablation retains cost regression.

Validity metrics use raw physical violations and actual contact flags;
zeroing a cost weight cannot make an infeasible trajectory appear valid.
Margin-only barriers are not contacts. Legacy metric behavior is preserved
for legacy objectives.

The generated backward is intentionally a surrogate when goal gradients or
curvature geometry are detached. `GuidanceBatch.detached_gradients` records
those excluded derivatives. The audit compares full-cost FD with
`guidance_gradient + detached_gradient`; its neural audit separately verifies
the actual surrogate backward and subtracts the excluded contribution from
raw-decoder FD. It still detects accidentally detached terminal outputs.

## Compatibility and reproduction

`TrainingConfig` defaults to `legacy_direct_v3` only for historical API and
checkpoint compatibility. Legacy `DirectEvaluator`/rolling/static objectives
remain unchanged. The new profile requires `collision_model: exact_geometry`
and rejects nonzero legacy `goal_stop_weight` or `peak_weight`. Resume requires
matching recipe/settings; warm-start transfers compatible network weights
and resets optimizer/calibration.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.training.runner \
  --config configs/yopo/train/yopo_stage_free.yaml \
  --dataset runs/yopo_whole_body/staged_repair_v1/curriculum_closed_loop_states/dataset \
  --run-dir runs/yopo_whole_body/<new-run>

MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.validation.audit \
  --dataset runs/yopo_whole_body/staged_repair_v1/curriculum_closed_loop_states/dataset \
  --checkpoint runs/yopo_whole_body/<new-run>/last \
  --samples 3 --epsilon 1e-5 --output <new-audit.json>
```

Use `YOPO.evaluation --guidance-profile yopo_minco_v1` for an explicitly common
new objective when comparing checkpoints. The existing acceptance tool runs
selected dense validation plus actual MuJoCo/controller tracking. Training or
audit success does not promote a checkpoint to the live benchmark.

The `local_motion_v1` acceptance gate requires >=90% selected dense validity
per family and actual reaching without collision. The former 0.5 rad/s gyro
handoff gate has been removed; that quantity remains an execution diagnostic.
Historical acceptance reports retain their original rules and results.

## Bounded verification run

`runs/yopo_whole_body/guidance_minco_v1/smoke_exact_acceleration` contains the
16-step run on the unchanged current-state dataset. Training cost decreased
from 5.47960 to 5.21962 (4.74%). The saved checkpoint's three-scene/nine-candidate
full 34-variable and neural-backward audits passed (maximum relative FD error
1.21e-5). Selected dense feasibility passed 4/4 observations. Actual controller
tracking failed with a workspace violation at 5.34 s, position error 1.484 m
and joint error 1.045 rad; maximum base tracking error was 6.1 mm. It is an
interface/gradient smoke checkpoint, not a qualified navigation policy.

`gate_local_motion_v1.json` reclassifies those same actual measurements under
the updated gate and still fails. Flight checkpoint presets are unchanged.
