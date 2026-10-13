# Wall portals + interior forest prototype — 2026-10-13

User approved closed-room rectangular, irregular stepped and consecutive offset
wall apertures, then added mixed-height bamboo/tree obstacles between walls.
This is motion planning to a specified eight-dimensional state only. Gripper
is fixed; no grasp, payload or contact objective was introduced.

## Deliverable

Config `configs/yopo_portal_dataset.yaml`, profile `portal_v4`, seed61.
Acceptance run `runs/yopo_whole_body/wall_portals_forest_v4`:9 physical layouts,
216 independent source snapshots, 72 per train/validation/test split. Each split
has rectangle, stepped and two-offset-wall families. Each layout has down-hanging
and laterally extended initial-arm source modes, all using the existing MINCO/
flatness/robot interfaces. Generated XML owns aperture and source-route metadata.

The closed physical room contains finite-depth box-union aperture walls plus
thin bamboo cylinders, larger tree trunks and solid sphere crowns with varied
heights. The two-wall family also has a low central tree obstacle within the
inter-wall route. Botanical realism is outside this primitive collision model.
Source paths are authored, not optimized; they do not serve as imitation labels.

## Evidence

All **18 source paths** pass independent dense exact-geometry/physical validation,
including polynomial joint-extrema checks. Production additionally requires a
strict environmental minimum distance of .05 m. Observed minimum is **.06843 m**.
All **18 same-base frozen-arm counterfactuals physically collide**, with base
geometry clear. Failures include lower wall edges and aperture sides; **3 sources**
also collide with the central tree between offset walls. This demonstrates
route-specific whole-body coupling, not infeasibility of every fixed-arm detour.

Every wall has an explicitly sampled center-crossing state per source mode; data
therefore includes passage through its finite thickness rather than only entrance
and exit poses. All **216 frames** reproduce their stored depth and masks. Tree/
bamboo/crown geometry is visible in **132/216** actual camera frames; full-world
geometry and ESDF remain privileged and invisible surfaces remain unobserved.

Figures and numeric audit are in `inspection/`: `portal_forest_gallery.png`,
`inspection_report.json`, and reproducible `visualize_data.py`. Only explanatory
RGB overviews remove room shell geometry and use translucent aperture walls;
stored/verified depth is rendered with the complete opaque physical scene.

## Verification and rulings

Relevant YOPO/end2end plus source/data/depth/trajectory integrations: **57 passed**,
scoped coverage **96.89%** (80% gate), using EGL rendering. Full units: **699 passed,
2 skipped,3 known baseline failures** in gate configuration and fake-viewer
initialization/reset timing. User's unrelated root README edit is preserved.

Final reviewer found an accepted .04906 m distance at the second wall's lower
edge because the existing validator tolerates3 mm constraint violation. A strict
regression failed; generation now rejects that condition and the offending lower
lip was adjusted. The same clearance/coupling tests now cover internal trees.
Intermediate wall-only and forest-only trial data were retained as diagnostics.

Ruling: box unions express stepped apertures while preserving the current exact
collision/conservative ESDF infrastructure. Arbitrary curved/rotated mesh holes
are not implemented. Ruling: retain bounded authored sources and small layout
variation for inspectable runtime; endpoint/speed/path diversity remains limited.
Ruling: interior vegetation uses physical cylinders/spheres, and global fixed-arm
infeasibility is not claimed. No neural retraining, physical flight or continuous
collision certification is asserted. Current 27 anchors/speed range still need
expression-range assessment before this data becomes a new training experiment.
