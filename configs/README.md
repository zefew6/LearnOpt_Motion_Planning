# Configuration index

Run commands from the repository root. Configurations are grouped by workflow;
flight, RL, planner and YOPO schemas remain independent.

| Directory | Purpose | Examples |
| --- | --- | --- |
| [flight/](flight/) | Interactive flight and trained-policy viewing | `flight_quickstart.yaml`, `flight.yaml`, `aerial_manipulator_workcell.yaml`, `yopo.yaml` |
| [planning/](planning/) | Standalone planner experiments | `bmtp.yaml` |
| [rl/](rl/) | PPO/ACMPC training and trajectory-bank settings | `ppo_trajectory.yaml`, `acmpc_gate_racing.yaml` |
| [yopo/dataset/](yopo/dataset/) | Whole-body depth dataset generation | `yopo_fixed_dense.yaml`; earlier stage presets are also available |
| [yopo/train/](yopo/train/) | Guidance-learning training | `yopo_cuda_large.yaml`; earlier stage presets are also available |

Quickstart flight:

```bash
.venv/bin/python -m uav_ac.main --config configs/flight/flight_quickstart.yaml
```

Bamboo dataset and CUDA training recipe:

```bash
MUJOCO_GL=egl .venv/bin/python -m uav_ac.yopo.data.generate \
  --config configs/yopo/dataset/yopo_fixed_dense.yaml \
  --run-dir runs/yopo_whole_body/my_bamboo_data
.venv/bin/python -m uav_ac.yopo.training \
  --config configs/yopo/train/yopo_cuda_large.yaml \
  --dataset runs/yopo_whole_body/my_bamboo_data/dataset \
  --run-dir runs/yopo_whole_body/my_bamboo_training
```

Flight checkpoint paths resolve relative to their YAML. Training datasets and
run directories are supplied explicitly. XML scenes remain under
`uav_ac/simulation/models/`; generated dataset XML belongs in `runs/`.
