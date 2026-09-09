"""Planning, training and deployment composition with explicit ownership."""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from .config import load_config, resolve_runtime


def _config(config):
    return load_config(config) if isinstance(config, (str, Path)) else deepcopy(config)


def _simulation(config):
    from uav_ac.simulation.mujoco_sim import MujocoSimulation
    return MujocoSimulation(config["scene"]["xml"], record_actual_trajectory=False,
                            planning_path_capacity=8 if config.get("planner", {}).get("name") == "bmtp" else 0)


def _seeds(config):
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(
        config["execution"]["seed"]).spawn(3)]


def _metadata(agent):
    checkpoint = Path(agent["checkpoint"])
    path = checkpoint.parent / "rl_config.json"
    if not path.is_file() and checkpoint.parent.name == "checkpoints":
        path = checkpoint.parent.parent / "rl_config.json"
    return json.loads(path.read_text())


def _reference(config, simulation):
    from uav_ac.planning.api import plan
    from uav_ac.rl.common.trajectory_bank import TrajectoryBank
    ref = config["task"]["reference"]
    dt = config["execution"]["action_dt"]
    if ref["source"] == "planner":
        return plan(config["planner"], simulation, dt, seed=_seeds(config)[0],
                    visualize=config["output"]["planning_visualization"])
    if ref["source"] == "trajectory_bank":
        bank = TrajectoryBank.load(ref["path"])
        # Old banks have duration per entry; verify their sampling clock rather
        # than silently retiming previously generated data.
        for index in bank.indices(ref["split"]):
            entry = bank.entry(int(index))
            if "duration" in entry:
                stored_dt = entry["duration"] / max(len(bank.trajectory(int(index))) - 1, 1)
                if not np.isclose(stored_dt, dt, rtol=1e-5, atol=1e-8):
                    raise ValueError("task.reference: bank sampling dt does not match execution.action_dt")
        return bank
    with np.load(ref["path"], allow_pickle=False) as saved:
        if not np.isclose(float(saved["dt"]), dt, rtol=1e-8, atol=1e-12):
            raise ValueError("task.reference: saved dt does not match execution.action_dt")
        return saved["trajectory"].copy()


def make_env(config, *, simulation=None, reference=None):
    """Build the existing tracking task with explicit scene, reference and wind.

    Custom non-tracking tasks use ``envs.MujocoTaskEnv`` and the Task protocol;
    they do not need a trajectory or planner.
    """
    from uav_ac.tasks.trajectory_tracking import MujocoTrajectoryTrackingEnv
    from uav_ac.simulation.wind_disturb import GustingCrosswind, RandomWindConfig
    from uav_ac.rl.common.assets import InitializationLibrary, ideal_initialization_library
    from uav_ac.rl.common.trajectory_bank import TrajectoryBank
    config = _config(config)
    simulation = _simulation(config) if simulation is None else simulation
    config = resolve_runtime(config, simulation)
    reference = _reference(config, simulation) if reference is None else reference
    task = config["task"]
    init = task["initialization"]
    library = None
    if init["source"] == "scene":
        if isinstance(reference, TrajectoryBank):
            raise ValueError("task.initialization.source: scene is incompatible with bank reset snapshots; use reference/random_reference")
        library = ideal_initialization_library(reference, simulation.quad)
        states, motors = library.states.copy(), library.motor_speeds.copy()
        states[0], motors[0] = simulation.quad.X.copy(), simulation.quad.omega.copy()
        library = InitializationLibrary(states, motors)
    agent = config.get("agent", {})
    metadata = _metadata(agent) if "checkpoint" in agent else {}
    policy = metadata.get("policy_type", agent.get("policy", "mlp"))
    wind = config["disturbance"]["wind"]
    random_wind = None
    fixed_wind = None
    if wind["type"] == "random_gust":
        fraction = config.get("training", {}).get("curriculum", {}).get("wind_fraction", 0.0)
        random_wind = RandomWindConfig(**wind["parameters"], curriculum_fraction=fraction)
    elif wind["type"] == "fixed_gust":
        fixed_wind = GustingCrosswind(**wind["parameters"])
    env = MujocoTrajectoryTrackingEnv(
        reference, library, model_path=config["scene"]["xml"],
        steps_per_action=config["execution"]["steps_per_action"],
        random_start=init["source"] == "random_reference",
        perturb_initial_state=init["perturb"], curriculum_progress=1.0 if config["mode"] != "train" else 0.0,
        split=task["reference"].get("split", "train"),
        wind_config=random_wind, fixed_wind=fixed_wind, wind_seed=_seeds(config)[2],
        success_position_error=task["success"]["position_tolerance_m"],
        observation_mode=policy,
        mpc_horizon_steps=metadata.get("mpc", agent.get("mpc", {})).get("horizon_steps", 20))
    env.experiment_config = config
    return env


def _save_config(config, simulation):
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    # Never replace an earlier experiment's evidence inadvertently.
    target = output / "resolved_config.yaml"
    if target.exists():
        raise FileExistsError(f"output.directory already contains an experiment: {output}; select a new directory")
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    from uav_ac.control.rl_controller import quad_parameters
    metadata = {"scene_xml": config["scene"]["xml"],
                "scene_sha256": hashlib.sha256(Path(config["scene"]["xml"]).read_bytes()).hexdigest(),
                "quad_parameters": quad_parameters(simulation.quad), "seeds": _seeds(config)}
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(yaml.safe_dump(config, sort_keys=False))
    return output


def train(config):
    """Translate the unified configuration once into the shared PPO trainer."""
    from uav_ac.rl.training import DEFAULT_TRAINING_CONFIG, _deep_merge, train as train_ppo
    from uav_ac.simulation.wind_disturb import RandomWindConfig
    config = _config(config)
    if config["mode"] != "train":
        raise ValueError("train requires mode: train")
    simulation = _simulation(config)
    config = resolve_runtime(config, simulation)
    _reference(config, simulation)  # Validate bank clock before writing anything.
    training = config["training"]
    wind = config["disturbance"]["wind"]
    if wind["type"] == "fixed_gust":
        raise ValueError("training currently supports none/random_gust wind; fixed_gust is deployment-only")
    wind_settings = asdict(RandomWindConfig(
        **(wind.get("parameters", {}) if wind["type"] == "random_gust" else {"probability": 0.0}),
        curriculum_fraction=training["curriculum"]["wind_fraction"]))
    settings = _deep_merge(deepcopy(DEFAULT_TRAINING_CONFIG), {
        **{key: value for key, value in training.items() if key not in {"curriculum"}},
        "policy_type": config["agent"]["policy"], "device": config["agent"]["device"],
        "trajectory_bank_path": config["task"]["reference"]["path"],
        "seed": config["execution"]["seed"],
        "steps_per_action": config["execution"]["steps_per_action"], "wind": wind_settings})
    if settings["policy_type"] == "acmpc":
        settings["mpc"] = {**config["agent"]["mpc"], "dt": config["execution"]["action_dt"]}
    output = _save_config(config, simulation)
    return train_ppo(output, settings=settings, model_path=config["scene"]["xml"])


def run(config):
    """Execute a validated experiment. Display and recording only observe state."""
    config = _config(config)
    if config["mode"] == "train":
        return train(config)
    simulation = _simulation(config)
    config = resolve_runtime(config, simulation)
    if config["mode"] == "replay":
        return replay(config, simulation)
    reference = _reference(config, simulation)
    if config["mode"] == "plan":
        output = _save_config(config, simulation)
        np.savez_compressed(output / "trajectory.npz", trajectory=reference, dt=config["execution"]["action_dt"])
        if config["output"]["record"]:
            raise ValueError("output.record: use deploy or replay to record flight states")
        if config["output"]["viewer"]:
            simulation.set_trajectory_visualization(reference[:, :3])
            import mujoco.viewer
            with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
        return output
    env = make_env(config, simulation=simulation, reference=reference)
    from uav_ac.deployment.agents import make_agent
    agent = make_agent(config, env)
    output = _save_config(config, simulation)
    if isinstance(reference, np.ndarray):
        np.savez_compressed(output / "trajectory.npz", trajectory=reference, dt=env.control_dt)
    try:
        for episode in range(config["execution"]["episodes"]):
            _episode(config, env, agent, output, episode)
    finally:
        env.close()
    return output


class Presentation:
    """Observers never advance physics or consume random numbers."""
    def __init__(self, config, simulation, output, episode):
        self.config, self.simulation = config, simulation
        self.output, self.episode = output, episode
        self.stack = ExitStack()
        self.viewer = self.renderer = self.recorder = None
        self.next_frame = 0.0

    def __enter__(self):
        import mujoco
        if self.config["output"]["viewer"]:
            import mujoco.viewer
            self.viewer = self.stack.enter_context(mujoco.viewer.launch_passive(
                self.simulation.model, self.simulation.data))
        if self.config["output"]["record"]:
            from uav_ac.simulation.recording import Mp4Recorder, default_camera
            self.simulation.model.vis.global_.offwidth = max(640, self.simulation.model.vis.global_.offwidth)
            self.simulation.model.vis.global_.offheight = max(480, self.simulation.model.vis.global_.offheight)
            self.renderer = mujoco.Renderer(self.simulation.model, height=480, width=640)
            self.stack.callback(self.renderer.close)
            self.recorder = self.stack.enter_context(Mp4Recorder(
                self.output / f"episode_{self.episode:03d}.mp4", 640, 480, 30))
            self.camera = default_camera(self.simulation.model)
        return self

    def update(self):
        if self.viewer:
            if not self.viewer.is_running():
                return False
            self.viewer.sync()
        if self.renderer:
            self.renderer.update_scene(self.simulation.data, camera=self.camera)
            while self.simulation.data.time + 1e-12 >= self.next_frame:
                self.recorder.write(self.renderer.render())
                self.next_frame += 1 / 30
        return True

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def _episode(config, env, agent, output, episode):
    observation, info = env.reset(seed=_seeds(config)[1] + episode)
    agent.reset()
    env.simulation.set_trajectory_visualization(env.trajectory[:, :3])
    states = [env.quad.X.copy()]
    motors = [env.quad.omega.copy()]
    timestamps = [float(env.simulation.data.time)]
    reward_sum = 0.0
    terminated = truncated = False
    with Presentation(config, env.simulation, output, episode) as presentation:
        presentation.update()
        for _ in range(config["execution"]["max_steps"]):
            start = time.monotonic()
            observation, reward, terminated, truncated, info = agent.step(env, observation)
            reward_sum += reward
            states.append(env.quad.X.copy())
            motors.append(env.quad.omega.copy())
            timestamps.append(float(env.simulation.data.time))
            if not presentation.update():
                truncated = True
                info["stop_reason"] = "viewer_closed"
            if terminated or truncated:
                break
            if config["output"]["viewer"]:
                time.sleep(max(0.0, env.control_dt - (time.monotonic() - start)))
        else:
            truncated = True
            info["stop_reason"] = "max_steps"
    np.savez_compressed(output / f"episode_{episode:03d}.npz", states=states,
                        motor_speeds=motors, times=timestamps, trajectory=env.trajectory,
                        dt=env.control_dt, scene_sha256=hashlib.sha256(Path(config["scene"]["xml"]).read_bytes()).hexdigest())
    summary = {"episode": episode, "reward": reward_sum, "terminated": terminated,
               "truncated": truncated, "steps": len(states) - 1, "metrics": info}
    (output / f"episode_{episode:03d}.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()), encoding="utf-8")
    print(f"Episode {episode}: reward={reward_sum:.3f}, steps={len(states)-1}, success={info.get('is_success', info.get('success', False))}")


def replay(config, simulation):
    """Display saved states without replanning or rerunning any control policy."""
    with np.load(config["replay"]["path"], allow_pickle=False) as saved:
        states, motors, times = saved["states"], saved["motor_speeds"], saved["times"]
        scene_hash = hashlib.sha256(Path(config["scene"]["xml"]).read_bytes()).hexdigest()
        if str(saved["scene_sha256"]) != scene_hash:
            raise ValueError("scene.xml: does not match recorded scene hash")
        trajectory = saved["trajectory"].copy()
    output = _save_config(config, simulation)
    simulation.set_trajectory_visualization(trajectory[:, :3])
    with Presentation(config, simulation, output, 0) as presentation:
        for index, (state, motor, stamp) in enumerate(zip(states, motors, times)):
            simulation.reset(state, motor)
            simulation.data.time = stamp
            if not presentation.update():
                break
            if config["output"]["viewer"] and index + 1 < len(times):
                time.sleep(max(0.0, float(times[index + 1] - stamp)))
    return output
