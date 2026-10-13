# Configuration index

Run commands from the repository root. Configurations are grouped by workflow;
flight, RL, planner and YOPO schemas remain independent.

| Directory | Purpose | Examples |
| --- | --- | --- |
| [flight/](flight/) | Interactive flight and trained-policy viewing | `flight_quickstart.yaml`, `flight.yaml`, `aerial_manipulator_workcell.yaml`, `yopo_like_wbp.yaml` |
| [planning/](planning/) | Standalone planner experiments | `bmtp.yaml` |
| [rl/](rl/) | PPO/ACMPC training and trajectory-bank settings | `ppo_trajectory.yaml`, `acmpc_gate_racing.yaml` |
| [yopo/dataset/](yopo/dataset/) | Whole-body depth dataset generation | `yopo_whole_body.yaml`, `yopo_coupled_dataset.yaml`, `yopo_portal_dataset.yaml` |
| [yopo/train/](yopo/train/) | Guidance-learning training | `yopo_p3.yaml`, `yopo_p4.yaml`, `yopo_portal_train.yaml` |

Quickstart flight:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/flight_quickstart.yaml
```

Current portal/forest dataset and first training recipe:

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.YOPO.generate \
  --config configs/yopo/dataset/yopo_portal_dataset.yaml \
  --run-dir runs/yopo_whole_body/my_portal_data
.venv/bin/python -m uav_ac.YOPO.training \
  --config configs/yopo/train/yopo_portal_train.yaml \
  --dataset runs/yopo_whole_body/my_portal_data/dataset \
  --run-dir runs/yopo_whole_body/my_portal_training
```

All existing file names are retained. Previous flat paths such as
`configs/flight_quickstart.yaml` now include the corresponding directory above;
update local shell commands/scripts to those paths. Root-level duplicate copies
are not maintained. Existing run artifacts/checkpoints were not rewritten.

Configuration parameter values are unchanged. Flight checkpoint paths are
relative to the YAML file, so moved flight presets use `../../runs/...` to resolve
to the same checkpoints. RL trajectory-bank paths retain their existing
repository-root interpretation. XML scenes remain under
`uav_ac/simulation/models/`; these folders do not introduce another scene schema.
