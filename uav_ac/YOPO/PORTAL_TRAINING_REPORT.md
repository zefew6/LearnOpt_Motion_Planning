# First portal/forest training — 2026-10-13

Run: `runs/yopo_whole_body/p4_portal_forest_seed23_v1`.
Dataset: `wall_portals_forest_v4/dataset` (9 layouts,216 snapshots,3 layouts per
split). Config: `configs/yopo_portal_train.yaml`. No source/network/loss code
changed; fresh width16 network,27 candidates,200 AdamW steps,batch1,CPU,seed23,
peak weight1,self clearance0,no augmentation. All72 train frames were eligible;
200 random draws visited71 of them. The 12-frame validation subset selects the
checkpoint; the separate24-frame test subset was not used for fitting/selection.

## Recommended checkpoint

`runs/yopo_whole_body/p4_portal_forest_seed23_v1/best_100`

Training completed200steps, but safety-prioritized validation retained step100:
proxy-valid selections were2/12 at50,3/12 at100,0/12 at150 and1/12 at200. Lower
validation cost at later steps did not replace the more valid checkpoint.
`last/` retains the200step weights and optimizer/RNG for reproducibility.

Initial gradient preflight on12 observations/324 candidates was finite but had
zero proxy-valid candidates. Model/contract tests passed12/12. Checkpoints were
fresh-loaded through the actual map-free planner for evaluation, including robot,
camera,normalization and fixed-gripper contracts. No deployment oracle was added.

## Common held-out test comparison

Both models receive exactly the same24 test observations with identical oracle
settings (peak1,self clearance0). The old model is
`runs/yopo_whole_body/p4_32_samples/best_100`.

| Metric | Old model | New best100 |
| --- | ---: | ---: |
| Learned selected mean cost | 85347.27 | 22710.93 |
| Proxy-valid selected candidates | 0/24 | 6/24 |
| Independently dense-validated selections, strict ≥.05m world gap | 0/24 | 12/24 |
| Selected trajectories with physical geometry collision | 18/24 | 3/24 |

New-model random expected cost is51459.38; safe-priority oracle cost11222.87.
Learned selection costs73.39% less than the old model and55.87% less than random.
Oracle is offline only. The new model's remaining12 exact-check failures comprise
**9 altitude-band violations** (`workspace_upper`,NEDz: planned base below the
allowed minimum height) and**3 world-collision/clearance failures**. Figures are
in `learning_curves.png` and detailed records in `new_model_exact.json`.

Proxy validity and exact validation are different metrics: the training oracle
uses conservative collision spheres/ESDF and integrated penalties, while the
independent audit checks exact geometry, polynomial joint extrema and physical
constraints at dense/refined times. Neither is continuous certification or a
physical flight result. Their disagreement must not be hidden by changing masks,
robot geometry or scene bounds.

## Optimization limits

On the fixed72-frame train set, the **last200 model's all-candidate mean cost**
fell86577.06→79647.40, only**8.00%**. This is not the test-selected cost and is not
a measurement of the recommended100-step checkpoint's training cost. Best100 was
chosen solely using validation. The later validity regression and limited mean
cost reduction show this bounded run is not converged.

The saved report timing610.64s covers the optimization loop and post-evaluations;
initial cost calibration/initial fixed-set evaluation are additional. CUDA was
unavailable andCPU was explicit. Old runs were preserved and no physical rollout,
controller integration,longer training,architecture or anchor modifications were
performed. Nine related layouts/24 test frames provide a prototype comparison,
not broad generalization evidence. Next investigation should prioritize the
height-bound violations and candidate expression/constraint balance before
expanding training budget or claiming reliable whole-body passage.

Artifacts: `report.json`, `experiment_summary.json`, `metrics.jsonl`,
`new_model_test.json`, `old_model_test.json`, `new_model_exact.json`,
`old_model_exact.json`, `learning_curves.png`, and reproducible evaluation/plot
scripts alongside the saved weights. Initial preflight is stored in
`wall_portals_forest_v4/training_preflight.json`.
