"""Single-owner configuration, resolved before any training or planning starts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path

import yaml


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of accepting the last spelling silently."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.load(stream, Loader=UniqueLoader)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    return data


def keys(value, allowed, path, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    unknown = value.keys() - set(allowed)
    missing = set(required) - value.keys()
    if unknown:
        raise ValueError(f"{path}.{sorted(unknown)[0]}: unknown or inactive option")
    if missing:
        raise ValueError(f"{path}.{sorted(missing)[0]}: required")


def positive(value, path, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: expected a positive finite number")
    if integer and (not isinstance(value, int)):
        raise ValueError(f"{path}: expected an integer")


def _path(value, base):
    if not isinstance(value, str) or not value:
        raise ValueError("file path must be a nonempty string")
    return str((base / value).resolve())


def load_config(path: str | Path) -> dict:
    """Load one experiment, with paths relative to the YAML that defines them.

    The only file inclusion is ``scene.config``. No INI, environment-variable,
    planner-specific scene override, or training YAML participates in resolution.
    """
    path = Path(path).resolve()
    data = read_yaml(path)
    keys(data, {"mode", "scene", "task", "planner", "agent", "disturbance", "execution", "training", "output", "replay"},
         "config", {"mode", "scene", "execution", "output"})
    data = deepcopy(data)
    mode = data["mode"]
    if mode not in {"plan", "train", "evaluate", "deploy", "replay"}:
        raise ValueError("mode: expected plan/train/evaluate/deploy/replay")
    scene = data["scene"]
    if "config" in scene:
        keys(scene, {"config"}, "scene")
        scene_path = Path(_path(scene["config"], path.parent))
        scene = read_yaml(scene_path)
        base = scene_path.parent
    else:
        base = path.parent
    keys(scene, {"xml"}, "scene", {"xml"})
    data["scene"] = {"xml": _path(scene["xml"], base)}
    if not Path(data["scene"]["xml"]).is_file():
        raise ValueError(f"scene.xml: file not found: {data['scene']['xml']}")
    execution = data["execution"]
    keys(execution, {"action_dt", "seed", "episodes", "max_steps"}, "execution", {"action_dt"})
    execution.setdefault("seed", 42)
    execution.setdefault("episodes", 1)
    execution.setdefault("max_steps", 100000)
    for name in ("episodes", "max_steps"):
        positive(execution[name], f"execution.{name}", integer=True)
    if isinstance(execution["seed"], bool) or not isinstance(execution["seed"], int) or execution["seed"] < 0:
        raise ValueError("execution.seed: expected a nonnegative integer")
    output = data["output"]
    keys(output, {"directory", "viewer", "record", "planning_visualization"}, "output", {"directory"})
    output["directory"] = _path(output["directory"], path.parent)
    for name in ("viewer", "record", "planning_visualization"):
        output.setdefault(name, False)
        if not isinstance(output[name], bool):
            raise ValueError(f"output.{name}: expected boolean")
    if mode == "train" and any(output[k] for k in ("viewer", "record", "planning_visualization")):
        raise ValueError("output: rendering is not supported during training")
    if mode == "replay":
        if any(k in data for k in ("task", "agent", "planner", "training", "disturbance")):
            raise ValueError("replay: task/agent/planner/training/disturbance are inactive")
        keys(data.get("replay"), {"path"}, "replay", {"path"})
        data["replay"]["path"] = _path(data["replay"]["path"], path.parent)
    else:
        if "replay" in data:
            raise ValueError("replay: only valid in replay mode")
        _task(data, path.parent)
        _agent(data, path.parent)
        _wind(data)
        _training(data)
    action_dt = execution["action_dt"]
    if action_dt == "from_model":
        if mode not in {"deploy", "evaluate"} or data["agent"]["type"] != "rl":
            raise ValueError("execution.action_dt: from_model requires a deployed RL checkpoint")
    else:
        positive(action_dt, "execution.action_dt")
    return data


def _task(data, base):
    task = data.get("task")
    keys(task, {"name", "reference", "initialization", "success"}, "task", {"name", "reference"})
    if task["name"] != "trajectory_tracking":
        raise ValueError("task.name: only trajectory_tracking is registered in the CLI; custom tasks use envs.MujocoTaskEnv")
    init = task.setdefault("initialization", {"source": "scene"})
    keys(init, {"source", "perturb"}, "task.initialization", {"source"})
    if init["source"] not in {"scene", "reference", "random_reference"}:
        raise ValueError("task.initialization.source: expected scene/reference/random_reference")
    init.setdefault("perturb", False)
    if not isinstance(init["perturb"], bool):
        raise ValueError("task.initialization.perturb: expected boolean")
    success = task.setdefault("success", {"position_tolerance_m": 0.5})
    keys(success, {"position_tolerance_m"}, "task.success", {"position_tolerance_m"})
    positive(success["position_tolerance_m"], "task.success.position_tolerance_m")
    ref = task["reference"]
    source = ref.get("source") if isinstance(ref, dict) else None
    if source == "planner":
        keys(ref, {"source"}, "task.reference")
        from uav_ac.planning.trajectory.bmtp import BMTPConfig, BMTPLimits
        planner = data.get("planner")
        keys(planner, {"name", "limits", "options", "initial_path"}, "planner", {"name", "limits"})
        if planner["name"] not in {"mini_snap", "gcopter", "gcs", "bmtp"}:
            raise ValueError("planner.name: unsupported planner")
        limits = planner["limits"]
        allowed = {f.name for f in fields(BMTPLimits)} if planner["name"] == "bmtp" else {"velocity"}
        keys(limits, allowed, "planner.limits", {"velocity"})
        for key, value in limits.items():
            positive(value, f"planner.limits.{key}")
        options = planner.setdefault("options", {})
        if planner["name"] == "bmtp":
            keys(options, {f.name for f in fields(BMTPConfig)}, "planner.options")
            BMTPConfig(**options)
            BMTPLimits(**limits)
        else:
            keys(options, set(), "planner.options")
        initial = planner.setdefault("initial_path", {"source": "scene_waypoints"})
        if planner["name"] == "bmtp":
            keys(initial, {"source", "index", "segments", "clearance"}, "planner.initial_path", {"source", "index", "segments", "clearance"})
            if initial["source"] != "scene_route":
                raise ValueError("planner.initial_path.source: BMTP requires scene_route")
            positive(initial["segments"], "planner.initial_path.segments", integer=True)
            if not isinstance(initial["index"], int) or isinstance(initial["index"], bool) or initial["index"] < 0:
                raise ValueError("planner.initial_path.index: expected nonnegative integer")
            if not isinstance(initial["clearance"], (int, float)) or not math.isfinite(initial["clearance"]) or initial["clearance"] < 0:
                raise ValueError("planner.initial_path.clearance: expected finite nonnegative number")
        else:
            keys(initial, {"source"}, "planner.initial_path", {"source"})
            if initial["source"] not in {"scene_waypoints", "random_open_field"}:
                raise ValueError("planner.initial_path.source: expected scene_waypoints/random_open_field")
    elif source in {"saved_trajectory", "trajectory_bank"}:
        allowed = {"source", "path"} | ({"split"} if source == "trajectory_bank" else set())
        keys(ref, allowed, "task.reference", {"path"})
        ref["path"] = _path(ref["path"], base)
        if source == "trajectory_bank":
            ref.setdefault("split", "train" if data["mode"] == "train" else "test")
            if ref["split"] not in {"train", "validation", "test"}:
                raise ValueError("task.reference.split: expected train/validation/test")
        if "planner" in data:
            raise ValueError("planner: inactive with saved reference; remove it")
    else:
        raise ValueError("task.reference.source: expected planner/saved_trajectory/trajectory_bank")
    if data["mode"] == "plan" and source != "planner":
        raise ValueError("mode plan requires task.reference.source: planner")


def _agent(data, base):
    if data["mode"] == "plan":
        if "agent" in data:
            raise ValueError("agent: inactive in plan mode")
        return
    agent = data.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("agent: required mapping")
    if agent.get("type") == "controller":
        keys(agent, {"type", "name", "options"}, "agent", {"name"})
        if data["mode"] == "train":
            raise ValueError("agent.type: training requires rl")
        options = agent.setdefault("options", {})
        if agent["name"] == "cascaded":
            keys(options, set(), "agent.options")
        elif agent["name"] == "mpc":
            from uav_ac.control.mpc_controller import MPCConfig
            keys(options, {f.name for f in fields(MPCConfig)} - {"dt"}, "agent.options")
        else:
            raise ValueError("agent.name: expected cascaded/mpc")
    elif agent.get("type") == "rl":
        if data["mode"] == "train":
            keys(agent, {"type", "policy", "device", "mpc"}, "agent", {"policy"})
            if agent["policy"] not in {"mlp", "acmpc"}:
                raise ValueError("agent.policy: expected mlp/acmpc")
            if agent["policy"] == "mlp" and "mpc" in agent:
                raise ValueError("agent.mpc: inactive for MLP")
            if agent["policy"] == "acmpc":
                from uav_ac.rl.acmpc.solver import MPCSettings
                keys(agent.setdefault("mpc", {}), {f.name for f in fields(MPCSettings)} - {"dt"}, "agent.mpc")
        else:
            keys(agent, {"type", "checkpoint", "device"}, "agent", {"checkpoint"})
            agent["checkpoint"] = _path(agent["checkpoint"], base)
            if not Path(agent["checkpoint"]).is_file() or Path(agent["checkpoint"]).suffix != ".zip":
                raise ValueError("agent.checkpoint: must name an existing .zip file, not a run directory")
        agent.setdefault("device", "cpu")
    else:
        raise ValueError("agent.type: expected controller/rl")


def _wind(data):
    disturbance = data.setdefault("disturbance", {"wind": {"type": "none"}})
    keys(disturbance, {"wind"}, "disturbance", {"wind"})
    wind = disturbance["wind"]
    kind = wind.get("type") if isinstance(wind, dict) else None
    from uav_ac.simulation.wind_disturb import GustingCrosswind, RandomWindConfig
    cls = {"fixed_gust": GustingCrosswind, "random_gust": RandomWindConfig}.get(kind)
    if kind == "none":
        keys(wind, {"type"}, "disturbance.wind")
    elif cls:
        keys(wind, {"type", "parameters"}, "disturbance.wind")
        parameters = wind.setdefault("parameters", {})
        keys(parameters, {f.name for f in fields(cls)} - {"curriculum_fraction"}, "disturbance.wind.parameters")
        cls(**parameters)
    else:
        raise ValueError("disturbance.wind.type: expected none/fixed_gust/random_gust (forces in N)")
    if data["mode"] == "plan" and kind != "none":
        raise ValueError("disturbance.wind: inactive in plan mode")


def _training(data):
    if data["mode"] != "train":
        if "training" in data:
            raise ValueError("training: only valid in train mode")
        return
    training = data.get("training")
    keys(training, {"total_timesteps", "n_envs", "torch_threads", "ppo", "evaluation_interval", "checkpoint_interval", "evaluation", "curriculum"},
         "training", {"total_timesteps", "n_envs", "ppo"})
    for key in ("total_timesteps", "n_envs", "torch_threads", "evaluation_interval", "checkpoint_interval"):
        if key in training:
            positive(training[key], f"training.{key}", integer=True)
    keys(training["ppo"], {"net_arch", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "learning_rate_start", "learning_rate_end", "clip_range", "ent_coef", "max_grad_norm"}, "training.ppo")
    if "net_arch" in training["ppo"]:
        keys(training["ppo"]["net_arch"], {"pi", "vf"}, "training.ppo.net_arch", {"pi", "vf"})
    keys(training.setdefault("evaluation", {}), {"panel_size", "perturb_initial_state"}, "training.evaluation")
    curriculum = training.setdefault("curriculum", {"wind_fraction": 0.3})
    keys(curriculum, {"wind_fraction"}, "training.curriculum")
    fraction = curriculum.setdefault("wind_fraction", 0.3)
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("training.curriculum.wind_fraction: expected [0, 1]")
    if data["task"]["reference"]["source"] != "trajectory_bank":
        raise ValueError("training currently requires task.reference.source: trajectory_bank")
    if data["task"]["reference"]["split"] != "train":
        raise ValueError("training requires task.reference.split: train")


def resolve_runtime(config, simulation):
    """Resolve model timing, physics stride and wind compatibility without mutation."""
    data = deepcopy(config)
    dt = data["execution"]["action_dt"]
    agent = data.get("agent", {})
    if agent.get("type") == "rl" and "checkpoint" in agent:
        from uav_ac.control.rl_controller import _validate_run_config
        checkpoint = Path(agent["checkpoint"])
        metadata_path = checkpoint.parent / "rl_config.json"
        if not metadata_path.is_file() and checkpoint.parent.name == "checkpoints":
            metadata_path = checkpoint.parent.parent / "rl_config.json"
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        _validate_run_config(metadata, simulation.quad)
        if metadata.get("task", "trajectory_tracking") != data["task"]["name"]:
            raise ValueError("agent.checkpoint: task contract mismatch")
        model_dt = float(metadata["control_dt"])
        if dt != "from_model" and not math.isclose(dt, model_dt, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("execution.action_dt: does not match agent.checkpoint control_dt")
        dt = model_dt
    stride = float(dt) / simulation.quad.dt
    if not math.isclose(stride, round(stride), abs_tol=1e-8) or round(stride) < 1:
        raise ValueError("execution.action_dt: must be an integer multiple of XML physics timestep")
    if data.get("disturbance", {}).get("wind", {}).get("type", "none") != "none":
        if any(simulation.model.opt.wind):
            raise ValueError("disturbance.wind: conflicts with nonzero XML option.wind")
    data["execution"]["action_dt"] = float(dt)
    data["execution"]["physics_dt"] = float(simulation.quad.dt)
    data["execution"]["steps_per_action"] = int(round(stride))
    return data
