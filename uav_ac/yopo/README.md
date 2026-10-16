# YOPO planning

YOPO has two independent workflows: official pure-UAV planning and articulated
whole-body planning. Online physics and task orchestration remain in repository
`runners/` and `tasks/`; robot interfaces remain in `robot/`.

| Directory | Responsibility |
| --- | --- |
| `uav/` | Official UAV collection, training, checkpoints and deployment |
| `whole_body/data/` | Whole-body observation records and collection adapters |
| `whole_body/model/` | Full-body network, primitives, encoding, FK and checkpoints |
| `whole_body/trajectory/` | Eight-dimensional MINCO and decoding |
| `whole_body/training/` | Guidance losses, training and CPU/CUDA backends |
| `whole_body/validation/` | Gradient audits, evaluation and controller acceptance |
| `common/` | Shared depth/YAML contracts and collection output handling |
| `vendor/yopo_minco/` | Pinned official network, MINCO, dataset and training losses |
| `vendor/yopo_simulator/` | Shared official random maps and depth collector |
| `_compat.py` | Lazy aliases for historical whole-body imports |

Third-party directories retain licenses and separate per-file provenance.
The simulator bundle contains the sources required for collection; ROS viewer
nodes, RViz files and screenshots are excluded. Executables, generated scenes,
datasets and checkpoints belong under ignored `runs/` directories.

- [Pure-UAV model, data collection and deployment](uav/README.md)
- [Whole-body models, training and validation](whole_body/README.md)
- [Official simulator build and source boundary](vendor/yopo_simulator/SOURCE.md)
- [Official training source and whole-body extensions](vendor/yopo_minco/SOURCE.md)

Existing checkpoint schemas and deployment imports are unchanged. The pure-UAV
collector remains `python -m uav_ac.yopo.uav.dataset`; whole-body numeric datasets
remain readable through `whole_body.data.dataset`. Removed historical samplers
are not restored as alternate workflows.

<a id="online"></a>
## Live whole-body planning

See the [online benchmark guide](whole_body/README.md#online). Launch from the
repository root using `configs/flight/yopo.yaml`; use `yopo_uav.yaml` for the
independent pure-UAV task.

<a id="primitives"></a>
[Whole-body primitive contracts](whole_body/README.md#primitives)

<a id="guidance"></a>
[Guidance learning](whole_body/README.md#guidance)

<a id="cuda-training"></a>
[Training and recovery](whole_body/README.md#cuda-training)

<a id="references"></a>
[Paper and code references](whole_body/README.md#references)
