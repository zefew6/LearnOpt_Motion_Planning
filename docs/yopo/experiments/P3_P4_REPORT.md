# P3/P4 implementation and acceptance — 2026-10-13

Implementation is maintained on `main`. Core network, anchors, decoding,
checkpoint loading and map-free prediction belong to `uav_ac/yopo/model`;
dataset generation, privileged whole-body MINCO costs, training and offline
comparison belong to `uav_ac/yopo`. [Commands and interfaces](../../../uav_ac/yopo/README.md) and
[paper/source mapping](../SOURCE.md) describe reproduction.

## Decisions and their limits

- Main learned profile `direct_v3` uses 34 fixed-semantic outputs. The older
  constrained 32-variable rolling profile remains a baseline. This removes exact
  measured-pose/gyro admission; actual attitude mismatch remains a physical
  deployment question, and angular-rate mismatch receives a soft cost.
- A reduced-width ResNet18-style backbone retains spatial features, actual-camera
  anchor projection and shared state/anchor conditioning. This adapts official
  YOPO ideas to arm states; it is not an architecture-equivalent reproduction.
- Explicit CPU smoke settings reflect unavailable CUDA. Small offline samples
  prove engineering behavior, not dense-environment generalization.
- Zero extra self-clearance follows the existing workcell convention for tangent
  authored sphere proxies. Geometry and collision masks are unchanged. This
  removes an impossible supervision margin and provides no added safety buffer.
- P4 first used eight training samples/60 steps; the bounded final attempt uses
  32 samples/100 steps. No physical flight or larger training stage is implied.

## P3 evidence

Dataset: `runs/yopo_whole_body/p2_rolling_v2_verified/dataset` (384 independent
reference snapshots, 12 scene-disjoint XML scenes, 8/2/2 split). This is not
policy rollout data. Run: `runs/yopo_whole_body/p3_final`.

Fixed 32 training samples, 100 AdamW steps, batch four, width 16:

| Metric | Value |
| --- | ---: |
| Initial mean guidance cost | 2029.4125 |
| Final mean guidance cost | 490.6041 |
| Reduction | 75.8253% |
| Validation selected cost (8 frames) | 6520.7874 |
| Validation sampled proxy validity | 4/8 |

The ≥10% optimization smoke gate passed. Gradients, save/load and exact CPU
continuation passed. P3 `last` is used for comparison; earlier pilot checkpoints
that predate the fixed-gripper contract are diagnostic artifacts.

## P4 evidence

The shared head emits 27 candidates (nine spatial directions × three arm
priors), all trained with analytic guidance. Ranking uses safety-prioritized
cross entropy; deployment uses highest learned logit without querying a map.
Final bounded run: `runs/yopo_whole_body/p4_32_samples`, 32 training samples,
100 steps, batch one. Mean training cost fell 37155.45→8647.59 (**76.73%**).
Validation safety-priority selection chose `best_100`.

The corrected 8-sample/60-step pilot `p4_final` reduced training mean cost
28233.25→8613.03 (69.49%). On eight validation frames, common peak-weight-zero
objective selected cost was 15760.11 versus random expected cost 16728.85
(5.79% lower), oracle safe-priority cost 6552.11, sampled selected validity 1/8.
P3 selected cost was 6520.79 and validity 4/8. Thus that pilot does **not**
establish a multi-candidate advantage over P3. Safety labels and arm/spatial
coverage remain meaningful failure diagnostics.

## Verification

- Focused end2end/YOPO plus real-depth/trajectory/train integration: **46 passed**,
  scoped coverage **94.52%** (80% threshold).
- Full integration: **20 passed** in 309.71 s, using `MUJOCO_GL=egl`.
- Full units: **690 passed, 2 skipped, 3 pre-existing failures**: gate racing
  configuration timing and two fake-viewer initialization/reset timing checks.
  Their unrelated behavior was preserved.
- Peak/time/boundary analytic gradients checked by directional finite differences.
- Candidate permutation, bounds, detached safe ranking targets, image-preserving
  augmentation and fresh-process deployment imports checked.
- Checkpoint metadata/geometry-buffer consistency, robot/camera mismatch, fixed
  gripper mismatch, incompatible resume settings and backward-step resume checked.
- Interrupted CPU training reproduces continuous-training weights exactly.
- Final read-only review identified camera/CNN offset, gripper contract, resume
  rewind and safety-priority checkpoint selection; all have regression coverage.

These are offline planning tests. There is no learned physics closed-loop flight,
continuous collision certificate or proof from limited field-of-view depth.
P5 should begin only after reviewing P4 ranking/coverage diagnostics.

## Final common-objective comparison

Both models were evaluated with the same peak-weight-zero objective and zero
extra self-clearance. The P4 training objective includes peak weight one, so
its training costs should not be directly compared to P3 training costs. P3
batch four/P4 batch one and different candidate budgets are also disclosed.

| Split (8 frames each) | P3 selected | P4 selected | P4 random expectation | P4 safe-priority oracle | P3/P4 selected proxy validity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 6520.79 | 13287.79 | 7068.65 | 4182.44 | 4/8, 4/8 |
| Test | 2326.62 | 1337.30 | 2291.54 | 415.34 | 2/8, 3/8 |

On the untouched test scenes P4 selection costs **41.64% less than random** and
42.52% less than P3. Validation selection remains worse than random and P3.
This passes the limited candidate-value/selection smoke on the fixed test
subset, but the conflicting splits and low proxy validity rule out a claim of
robust narrow/dense-environment performance. Only two scenes per held-out split
and eight diagnostic frames were evaluated; these are descriptive numbers,
not statistical evidence of generalization. Selected test candidates still have
environment/self/physical proxy violations in 50%/25%/37.5% of frames, respectively
(these overlap).

Raw reports: `p4_32_samples/best_test_comparison.json` and
`p4_32_samples/validation_comparison.json`. No oracle selection is used by the
planner. The oracle gap suggests remaining room in learned ranking; the candidate
validity suggests that ranking alone cannot solve all failures.

CPU prediction timing (20 warmed calls, two torch threads): median **3.73 ms**,
p95 **4.96 ms** for observation encoding, network, learned selection and MINCO
decoding. Depth capture, scheduling, controller and physics are excluded.
This timing does not prove a physical replanning frequency.

P3/P4 implementation and bounded offline acceptance are complete. Stop here
before P5. Next investigation should separate arm/spatial anchor coverage,
whole-trajectory constraint penalties and score ranking with larger scene-disjoint
diagnostics, then assess actual-state physical rolling execution.
