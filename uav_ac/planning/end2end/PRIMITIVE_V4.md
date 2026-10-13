# YOPO whole-body primitive_v4

This profile implements the user's selected YOPO-style 3×3 motion-primitive
partition and primitive-frame input/output alignment. It preserves independent
single-frame training and actual-state online replanning. ESDF remains privileged
training/evaluation information; deployment does not use it to filter candidates.

## Representation

The complete calibrated 96×160 depth/mask image enters one CNN. The nine primitive
cells cover left/centre/right and up/centre/down camera angles. Backbone features
are sampled at the projected centres of these cells, then each cell uses the same
1×1 convolutional head. CNN feature-map dimensions are not the primitive count:
the existing stride-32 backbone still produces 3×5 spatial features. We retain
camera intrinsics/extrinsics and match nine prediction sites to the nine angular
cells without cropping independent images or changing the depth dataset.

`primitive.geometry` constructs a right-handed orthonormal primitive frame for
each camera direction in FRD. The camera-frame ray is proportional to
`[tan(horizontal angle), tan(vertical angle), 1]`. This respects the rectangular
pinhole camera projection; it is not an uncalibrated spherical XYZ offset.

For each cell, current base PVA spatial vectors, linear velocity and goal direction
are rotated from the gravity-local frame through body FRD into the primitive
frame. Body angular velocity and gravity are rotated from FRD directly. Actual
attitude metadata, scalar yaw semantics and arm joints/rates are retained. Joint
angles are not spatial vectors. The goal position feature is clipped as
`goal_delta / max(goal_clip, norm(goal_delta))`, with goal_clip=3 m. The actual
8-D final goal remains unchanged in the observation and training goal cost.

Each cell has three conditioned arm modes. A learned mode embedding feeds the
same prediction head; initial arm configurations use direct/folded/extended
priors. Mode order is **cell-major, arm-minor**: `candidate = cell*3 + arm_mode`,
with cell order top-to-bottom and left-to-right. All 27 candidates get independent
raw predictions and scores. This is an initialization prior, not hard collision
certification or a requirement that three modes remain distinct after learning.

The raw 34 channels predict:

| Channels | Meaning |
| --- | --- |
| 0–2 | Inner camera angular offsets and radial distance |
| 3–7 | Inner yaw and four arm angles |
| 8–10 | Independent terminal camera direction and radial distance |
| 11–15 | Terminal yaw and four arm angles |
| 16–23 | Terminal base velocity in the primitive frame, yaw rate and arm rates |
| 24–31 | Terminal base acceleration in the primitive frame, yaw acceleration and arm accelerations |
| 32–33 | Two positive segment durations |

The inner angular half-range is 1.1 times the cell half-width, allowing modest
overlap; camera-edge angles are clipped to the full FOV. Inner radial range is
0.2–2.5 m. The endpoint is independently predicted across the full camera FOV,
with radius 0.2–3.0 m, rather than restricted to a ±0.25 m residual about a shared
endpoint. Its range is not tied to the inner cell. Joint outputs can explore the
full joint bounds through a tanh map around mode priors. Derivative components
are bounded, while full trajectory feasibility remains a training/evaluation
constraint. Segment durations lie in 0.1–3 s.

Spatial derivatives are rotated primitive→FRD→gravity-local. The differentiable
decoder then emits the existing **physical local MINCO34** layout (inner8,
terminal PVA24, durations2). Existing DirectEvaluator gradients and MINCO solver
are reused; actual current head PVA is preserved exactly. This does not enforce
exact actual attitude/body-rate consistency or certify the full trajectory.

## Contracts and commands

Network and geometric encoding live in `planning/end2end/primitive*.py`.
Dataset generation, oracle guidance and training stay under `uav_ac/YOPO`.
Training selects `profile: primitive_v4` and requires 27 candidates.

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.training \
  --config configs/yopo/train/yopo_primitive_pilot.yaml \
  --dataset runs/yopo_whole_body/wall_portals_forest_v4/dataset \
  --run-dir runs/yopo_whole_body/my_primitive_pilot
```

Saved checkpoints use schema 2, profile primitive_v4, a distinct 58-value state
layout contract and serialized PrimitiveSpec. Geometry/normalization buffers are
verified against metadata, including all spec values. Legacy schema-1 direct_v3
checkpoints still load through their original network/encoding; cross-profile
training resume is rejected. Both evaluation and deployment use the saved spec.
Existing depth/current-state datasets can be reused; old network weights cannot
be treated as trained primitive_v4 weights.

Experimental live entry point:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/yopo_primitive_wbp.yaml
```

It uses the local 32-step pilot checkpoint; edit the YAML checkpoint path when
training a different run. The old `yopo_like_wbp.yaml` and its weights are retained.
Ordinary viewer runs write no artifacts. The new entry point uses the existing
controller, scene and live overlays, without a classical map-based fallback.

## Verification and limits

Regression tests cover calibrated cell/feature correspondence, right-handed
frames, unchanged joint features, primitive derivative rotations, independent
endpoint prediction, inner angular bounds, zero-head initialization/calibration
agreement, MINCO moving-head continuity, finite-difference guidance gradients,
legacy/new checkpoint round trips, geometry metadata rejection, train/save/load/
deploy and resume normalization. A final independent review found a non-default
resume spec issue; it was reproduced and fixed with a regression test.

The engineering pilot used 24 independent training samples, 6 held-out samples,
32 optimizer steps, batch size 2, width16 and learning rate 0.0003. Training
mean cost fell from 307211 to 262542 (14.5%). Held-out selected/candidate constraint
validity was **0%**. This is evidence that the chain trains and serializes, not
that it has learned obstacle avoidance. The real-physics diagnostic accepted
9 plans and collided at 1.79 s; maximum base tracking error was 0.0251 m, joint
error 0.0865 rad, and handoff body-rate discrepancy 5.26 rad/s. Further data,
training and constraint/handoff evaluation are required before treating this
checkpoint as a usable navigation policy. No success claim is made.

The small initial pilot's samples all have zero relative goal yaw. A later
calibration regression also verifies nonzero relative goal yaw; its anchor yaw
initialization was corrected without changing the reported pilot's encoding.

References: original YOPO II-A (https://tju-air-lab.github.io/projects/YOPO/YOPO.pdf),
YOPO-MINCO II-A–C and official branch
(https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-MINCO).

The desktop YAML/main entry point also ran the new profile, showing selected
candidate changes (12→3) and reporting collision at 2.71 s. An isolated depth
sensitivity probe changed the initial selection from 12 to 4 when replacing
actual depth by all no-return pixels. These establish that the learned output
and selection are not simply fixed by cell ID; they do not establish good
avoidance. Different asynchronous runs collided at different times, so the
single-run 1.79 s diagnostic is not a stable performance estimate.

Final relevant checks: 81 passed, with two unrelated witness tests excluded
(the scene was unchanged). `git diff --check` passed.
