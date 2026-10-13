# Corridor/bamboo coupled-data prototype — 2026-10-13

User goal: motion-only whole-body planning to an eight-dimensional configuration,
using corridors and bamboo-like cylinder scenes with varying lower-obstacle
heights and varying front-gap widths. Gripper is fixed at .06 m. No grasp,
payload or end-effector task was added. Existing robot and camera models and
observation/deployment contracts remain unchanged.

## Delivered data

- Config: `configs/yopo_coupled_dataset.yaml`; profile `coupled_v3`, seed31.
- Run: `runs/yopo_whole_body/coupled_corridor_bamboo_v2`.
- Six physical scene layouts, 144 snapshots; 48 per train/validation/test split.
- Each split contains one corridor and one bamboo scene. Each layout contains
  two authored source modes: vertical hanging and lateral extension.
- Twelve six-piece MINCO sources; snapshots cover approach/traverse/restore.
- Source coefficients/validation/counterfactuals remain privileged resources;
  observations retain schema2. Source resources are checksummed by the loader.

## Independent evidence

All twelve source paths pass the existing full-body dense exact-geometry and
physical validator at nominal .025 s resolution with refinement and analytic
joint-extrema checks. Required environment clearance is .05 m; smallest observed
source world clearance is **.06190 m**, smallest exact self-clearance **.06000 m**.
This is sampled/refined validation, not a continuous safety certificate.

All twelve fixed-arm/same-base counterfactuals physically collide. Base-only
geometry remains clear. The paired modes establish collisions against both
lower obstacles and front-gap pillars/trunks; initial lower-only prototype v1
was retained as a diagnostic and corrected to v2 after review. Maximum frozen
world penetration is approximately **.11705 m**. These are route-specific
witnesses, not proof that every possible fixed-arm route is infeasible.

All 144 observations reconstruct from stored source PVA and jerk and reproduce
stored depth/masks. A geometry-ID segmentation audit finds both lower and front
obstacles in **48/48 approach frames** with the original camera mounting. Later
hidden obstacles remain hidden; ESDF does not enter observations.

Figures and exact numeric audit are in the run's `inspection/` folder:
`coupling_comparison.png`, `depth_gallery.png`, `inspection_report.json`.
Explanatory RGB overviews hide overhead/side-wall geometry to reveal the route;
actual sensor renders retain it. Orange dashed proxy envelopes show frozen arms,
blue filled envelopes show coordinated arms; collision evidence uses exact
geometry rather than the plotted proxies.

## Verification and decisions

Relevant YOPO/end2end/depth/dynamic/trajectory integration checks: **51 passed**,
scoped coverage **96.74%**. Full integration: **21 passed** in 242.51 s (EGL). Full units: **694 passed,2 skipped,3 known baseline
failures** (gate config timing and two fake-viewer timing/reset checks). Existing
user README edits were preserved. Final reviewer identified missing front-width
arm coupling; a test failed on both families before adding lateral-start sources,
then passed along with the full relevant suite.

Ruling: authored validated MINCO sources rather than a large RRT/L-BFGS generation
batch keep this first-stage runtime/coverage inspectable. They are not optimized
planning demonstrations; speed, endpoint and path diversity remain limited.
A single reference family does not establish broad generalization. The network
was not enlarged or retrained; candidate/speed expression remains a separate
acceptance gate. Old datasets/runs were retained.

Stop at this dataset prototype. Next step is reviewing visual coverage and
checking whether the current two-piece/27-anchor output family can express local
fold-and-slow behavior before beginning a new training experiment.
