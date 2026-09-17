"""Metrics and replay for trained gate-racing policies."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from .config import read_run
from .environment import make_environment


def evaluate_model(model, settings, seeds, *, perturb=True, frame=None,
                   continue_after_terminal=False):
    if not seeds:
        raise ValueError("evaluation requires at least one seed")
    if frame is not None and len(seeds) != 1:
        raise ValueError("replay requires exactly one episode")
    environments = []
    results = [None] * len(seeds)
    strict = getattr(model.policy, "solver_strict", None)
    if strict is not None:
        model.policy.solver_strict = False
    try:
        observations = []
        for seed in seeds:
            env = make_environment(settings, perturb=perturb, record_actual_trajectory=frame is not None)
            env.unwrapped.continue_after_terminal = bool(continue_after_terminal)
            environments.append(env)
            observations.append(env.reset(seed=int(seed))[0])
            if frame is not None:
                frame(env, False)
        totals = np.zeros(len(seeds))
        calls = np.zeros(len(seeds), dtype=int)
        failures = np.zeros(len(seeds), dtype=int)
        seconds = np.zeros(len(seeds))
        single_latencies = []
        maximum_residual = 0.
        active = list(range(len(seeds)))
        while active:
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            control_start = time.perf_counter()
            selected = [observations[i] for i in active]
            obs = ({key: np.stack([o[key] for o in selected]) for key in selected[0]}
                   if isinstance(selected[0], dict) else np.stack(selected))
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            actions, _ = model.predict(obs, deterministic=True)
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            duration = time.perf_counter()-start
            if len(seeds) == 1:
                single_latencies.append(time.perf_counter()-control_start)
            diagnostic = getattr(getattr(model.policy, "mpc", None), "last_diagnostics", {})
            maximum_residual = max(maximum_residual, diagnostic.get("max_residual", 0.))
            batch_failures = diagnostic.get("sample_failures", [0] * len(active))
            remaining = []
            for offset, index in enumerate(active):
                env = environments[index]
                observations[index], reward, terminated, truncated, info = env.step(actions[offset])
                totals[index] += reward
                calls[index] += 1
                failures[index] += batch_failures[offset]
                seconds[index] += duration / len(active)
                keep_running = True
                if frame is not None:
                    keep_running = frame(env, terminated or truncated) is not False
                if terminated or truncated:
                    if truncated and not terminated:
                        info["termination_reason"] = "timeout"
                    results[index] = {**info, "seed": int(seeds[index]),
                                      "course": env.unwrapped.task.course_description,
                                      "timeout": bool(truncated and not terminated),
                                      "return": float(totals[index]), "solver_failures": int(failures[index]),
                                      "policy_calls": int(calls[index]), "inference_seconds": float(seconds[index])}
                    if continue_after_terminal and keep_running:
                        remaining.append(index)
                elif keep_running:
                    remaining.append(index)
            active = remaining
    finally:
        if strict is not None:
            model.policy.solver_strict = strict
        for env in environments:
            env.close()
    successes = [r for r in results if r["success"]]
    calls = sum(r["policy_calls"] for r in results)
    gate_speeds = [speed for result in results for speed in result["gate_speeds"]]
    straight_seconds = sum(r["straight_seconds"] for r in results)
    latency = None
    if single_latencies:
        latency = {f"p{q}_ms": float(np.percentile(single_latencies, q)*1000) for q in (50, 95, 99)}
        latency.update(over_10ms_rate=float(np.mean(np.array(single_latencies) > .01)),
                       samples=len(single_latencies),
                       definition="single-environment observation batching through returned action; excludes physics and rendering; includes cold first call")
    return {"episodes": results, "success_rate": len(successes)/len(results),
            "single_environment_latency": latency, "max_solver_residual": maximum_residual,
            "mean_gate_speed": float(np.mean(gate_speeds)) if gate_speeds else None,
            "mean_straight_speed": sum((r["mean_straight_speed"] or 0)*r["straight_seconds"] for r in results)/straight_seconds if straight_seconds else None,
            "solver_residual_is_convergence_certificate": False,
            "mean_gates_passed": float(np.mean([r["gates_passed"] for r in results])),
            "collision_rate": float(np.mean([r["collision"] for r in results])),
            "missed_gate_rate": float(np.mean([r["termination_reason"] == "missed_gate" for r in results])),
            "out_of_bounds_rate": float(np.mean([r["termination_reason"] == "out_of_bounds" for r in results])),
            "timeout_rate": float(np.mean([r["timeout"] for r in results])),
            "mean_success_seconds": float(np.mean([r["elapsed_seconds"] for r in successes])) if successes else None,
            "mean_speed": float(np.mean([r["mean_speed"] for r in results])),
            "peak_speed": max(r["peak_speed"] for r in results),
            "solver_failure_rate": sum(r["solver_failures"] for r in results)/calls,
            "mean_inference_seconds": sum(r["inference_seconds"] for r in results)/calls,
            "timing_definition": "amortized batched policy seconds per action, excluding physics",
            "maximum_inference_batch": len(seeds)}


def load_model(run_dir, device="cpu", *, checkpoint=None):
    run_dir = Path(run_dir)
    if not (run_dir / "rl_config.json").is_file() and (run_dir.parent / "rl_config.json").is_file():
        run_dir = run_dir.parent
    _, settings = read_run(run_dir)
    torch.set_num_threads(settings["torch_threads"])
    env = make_environment(settings)
    try:
        model = PPO.load(Path(run_dir) / "best_model.zip" if checkpoint is None else Path(checkpoint), env=env, device=device)
    finally:
        env.close()
    return model, settings


def evaluate_metrics(run_dir, *, seeds=tuple(range(20)), perturb_initial_state=True, device="cpu"):
    model, settings = load_model(run_dir, device)
    result = evaluate_model(model, settings, seeds, perturb=perturb_initial_state)
    (Path(run_dir) / "evaluation_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def replay(run_dir, *, device="cpu", seed=0, output=None, fps=30., width=1280, height=720,
           checkpoint=None, follow_camera=False):
    import mujoco
    from uav_ac.simulation.recording import Mp4Recorder, default_camera, update_chase_camera

    if not np.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
        raise ValueError("positive video dimensions and FPS are required")
    if output is not None and Path(output).exists():
        raise FileExistsError(f"video already exists: {output}")
    model, settings = load_model(run_dir, device, checkpoint=checkpoint)
    renderer = recorder = viewer = None
    next_frame = 0.
    last_frame = time.perf_counter()
    try:
        def frame(env, done):
            nonlocal renderer, recorder, viewer, next_frame, last_frame
            sim = env.unwrapped.simulation
            if output is None:
                if viewer is None:
                    from mujoco import viewer as mj_viewer
                    viewer = mj_viewer.launch_passive(sim.model, sim.data)
                    camera = default_camera(sim.model)
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
                if not viewer.is_running():
                    return False
                if follow_camera:
                    with viewer.lock():
                        update_chase_camera(viewer.cam, sim)
                viewer.sync()
                remaining = env.unwrapped.control_dt-(time.perf_counter()-last_frame)
                if remaining > 0:
                    time.sleep(remaining)
                last_frame = time.perf_counter()
            elif sim.data.time + 1e-9 >= next_frame or done:
                if renderer is None:
                    renderer = mujoco.Renderer(sim.model, width=width, height=height)
                    recorder = Mp4Recorder(output, width, height, fps).__enter__()
                camera = default_camera(sim.model)
                if follow_camera:
                    update_chase_camera(camera, sim)
                renderer.update_scene(sim.data, camera=camera)
                pixels = renderer.render()
                wrote = False
                while sim.data.time + 1e-9 >= next_frame:
                    recorder.write(pixels)
                    next_frame += 1 / fps
                    wrote = True
                if done and not wrote:
                    recorder.write(pixels)
        return evaluate_model(model, settings, (seed,), frame=frame,
                              continue_after_terminal=output is None)
    finally:
        if viewer is not None:
            viewer.close()
        if recorder is not None:
            recorder.close()
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a gate-racing checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate_metrics(args.run_dir, seeds=(args.seed,), device=args.device), indent=2))
