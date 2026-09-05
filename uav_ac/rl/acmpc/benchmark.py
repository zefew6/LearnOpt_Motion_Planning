"""Reproducible validation and paired controller comparisons.

Run: python -m uav_ac.rl.acmpc.benchmark --run-dir RUN --mode preflight
or:  python -m uav_ac.rl.acmpc.benchmark --run-dir RUN --mode compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from uav_ac.control.mpc_controller import MPCController, MPCConfig
from uav_ac.control.rl_controller import quad_parameters
from uav_ac.rl.common.environment import MujocoTrajectoryTrackingEnv
from uav_ac.rl.mlp_baseline.training import load_training_config, prepare_assets
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH
from uav_ac.simulation.wind_disturb import RandomWindConfig
from .policy import ACMPCPolicy
from .solver import MPCSettings, SOLVER_VERSION
from dataclasses import asdict


def make_policy(env, settings, device):
    return PPO(ACMPCPolicy, env, device=device, n_steps=8, batch_size=8, seed=42,
        policy_kwargs={"quad_parameters":quad_parameters(env.quad),
            "mpc_settings":asdict(MPCSettings(**settings.get("mpc", {}))),
            "net_arch":settings["ppo"]["net_arch"]})


def collect_panel(policy, environments, cases, *, native_mpc=False, progress=False):
    """Batch simultaneous policy calls; each case owns its plant and RNG."""
    observations, rows = [], []
    controllers = []
    for env, case in zip(environments, cases):
        obs, _ = env.reset(seed=case["seed"], options={
            "trajectory_id":case["trajectory_id"], "start_index":case.get("start_index",0),
            "perturbation_scale":case.get("perturbation_scale",1.),
            "wind_enabled":case["wind_enabled"], "wind_scale":1.})
        observations.append(obs)
        rows.append(dict(case, steps=0, return_=0., latencies=[], allocation_scaled_steps=0,
            solver_failures=0, solver_retries=0, max_position_error=0.))
        if native_mpc:
            controllers.append(MPCController(env.quad, MPCConfig(dt=env.control_dt)))
    active = list(range(len(environments)))
    iteration = 0
    while active:
        if not native_mpc:
            batch = ({key:np.stack([observations[i][key] for i in active]) for key in observations[active[0]]}
                     if isinstance(observations[active[0]],dict) else np.stack([observations[i] for i in active]))
            if policy.device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            actions, _ = policy.predict(batch, deterministic=True)
            if policy.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            diagnostic = getattr(getattr(policy.policy, "mpc", None), "last_diagnostics", {})
        completed = []
        for j,i in enumerate(active):
            env, row = environments[i], rows[i]
            if native_mpc:
                started = time.perf_counter()
                command = controllers[i].step(env.quad, env._reference())
                elapsed = time.perf_counter()-started
                observations[i], reward, terminated, truncated, info = env.step_wrench(command)
                row["solver_failures"] += int(controllers[i].last_status != 0)
            else:
                observations[i], reward, terminated, truncated, info = env.step(actions[j])
                row["solver_failures"] += diagnostic.get("sample_failures", [0]*len(active))[j]
                row["solver_retries"] += diagnostic.get("sample_retries", [0]*len(active))[j]
            row["steps"] += 1
            row["return_"] += reward
            row["latencies"].append(elapsed)
            row["allocation_scaled_steps"] += int(info["allocation_scale"] < 1-1e-8)
            row["max_position_error"] = max(row["max_position_error"],info["position_error"])
            limit = cases[i].get("max_steps")
            budget_done = limit is not None and row["steps"] >= limit
            if terminated or truncated or budget_done:
                row.update(success=bool(info["success"]), collision=bool(info["collision"]),
                    failure_reason=info["failure_reason"], position_rmse=info["position_rmse"],
                    final_position_error=info["position_error"], budget_truncated=bool(budget_done and not terminated and not truncated))
                completed.append(i)
        active = [i for i in active if i not in completed]
        iteration += 1
        if progress and iteration % 250 == 0:
            print(f"panel step={iteration} active={len(active)}",flush=True)
    for row in rows:
        row["return"] = row.pop("return_")
        latencies = row.pop("latencies")
        row["batch_call_p50_ms"] = float(np.percentile(latencies,50)*1000)
        row["batch_call_p95_ms"] = float(np.percentile(latencies,95)*1000)
        row["allocation_scaled_fraction"] = row["allocation_scaled_steps"]/row["steps"]
    return rows


def preflight(settings, run_dir, device):
    paths = []
    for kind in ("hover", "short_track"):
        trajectory = np.zeros((201,10))
        trajectory[:,2] = -1.5
        if kind == "short_track":
            t = np.arange(201)*.01
            # Smooth minimum-jerk 20 cm move over two seconds.
            s = t/2
            trajectory[:,0] = .2*(10*s**3-15*s**4+6*s**5)
            trajectory[:,3] = .1*(30*s**2-60*s**3+30*s**4)
            trajectory[:,6] = .05*(60*s-180*s**2+120*s**3)
        env = MujocoTrajectoryTrackingEnv(trajectory,model_path=OPEN_FIELD_SCENE_PATH,
            observation_mode="acmpc",mpc_horizon_steps=settings["mpc"]["horizon_steps"],
            random_start=False,perturb_initial_state=False)
        try:
            policy = make_policy(env,settings,device)
            policy.policy.solver_strict = True
            rows = collect_panel(policy,[env],[dict(trajectory_id=0,seed=42,wind_enabled=False,perturbation_scale=0.)])
            rows[0]["case"] = kind
            paths.extend(rows)
        finally:
            env.close()
    result = {"device":device,"solver_version":SOLVER_VERSION,"episodes":paths,
        "passed":all(r["success"] and r["position_rmse"] < .25 and r["solver_failures"] == 0 for r in paths)}
    (run_dir/"preflight.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("fixed MPC closed-loop preflight failed; see preflight.json")
    return result


def compare(settings, run_dir, device, baseline_run, count=5, methods=("acmpc","fixed_mpc","ppo","acados")):
    bank, _ = prepare_assets(run_dir,settings=settings)
    cases = [dict(trajectory_id=int(i),seed=int(100000*wind+j),wind_enabled=bool(wind))
        for j,i in enumerate(bank.indices("test")[:count]) for wind in (0,1)]
    output = run_dir/"comparison.json"
    results = json.loads(output.read_text())["results"] if output.exists() else {}
    for method in methods:
        envs = [MujocoTrajectoryTrackingEnv(bank,model_path=OPEN_FIELD_SCENE_PATH,split="test",
            observation_mode="acmpc" if method in {"acmpc","fixed_mpc"} else "mlp",
            mpc_horizon_steps=settings["mpc"]["horizon_steps"], random_start=False,
            curriculum_progress=1.,wind_config=RandomWindConfig(**settings["wind"])) for _ in cases]
        try:
            model = (make_policy(envs[0],settings,device) if method == "fixed_mpc" else
                PPO.load((baseline_run if method == "ppo" else run_dir)/"best_model.zip",device=device)
                if method != "acados" else None)
            print(f"Comparing {method}",flush=True)
            episodes = collect_panel(model,envs,cases,native_mpc=method=="acados",progress=True)
            results[method] = {"episodes":episodes,"success_rate":float(np.mean([r["success"] for r in episodes])),
                "mean_position_rmse":float(np.mean([r["position_rmse"] for r in episodes]))}
            (run_dir/"comparison.json").write_text(json.dumps({"device":device,"results":results,
                "note":"PPO checkpoint is a long-trained performance reference, not a matched sample-budget baseline. Native acados commands retain rotor allocation; reward action penalties use bounded normalization."},indent=2),encoding="utf-8")
        finally:
            for env in envs:
                env.close()
    return results


def timing(settings, run_dir):
    from .solver import DifferentiableMPC
    from .dynamics import QuadrotorDynamics
    trajectory = np.zeros((25,10)); trajectory[:,2] = -1.5
    env = MujocoTrajectoryTrackingEnv(trajectory,model_path=OPEN_FIELD_SCENE_PATH,
        observation_mode="acmpc",random_start=False,perturb_initial_state=False)
    obs,_ = env.reset()
    records = []
    for device in (["cpu","cuda"] if torch.cuda.is_available() else ["cpu"]):
        for batch in (1,4,32):
            layer = DifferentiableMPC(quad_parameters(env.quad)).to(device)
            x = torch.tensor(obs["mpc_state"],device=device).repeat(batch,1); x[:,0]+=.1
            ref = torch.tensor(obs["reference"],device=device).repeat(batch,1,1)
            previous = torch.zeros(batch,4,device=device)
            cost = torch.zeros(batch,layer.cost_size,device=device)
            elapsed = []
            for index in range(4):
                if device == "cuda": torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    layer(x,ref,previous,cost,strict=True)
                if device == "cuda": torch.cuda.synchronize()
                if index: elapsed.append(time.perf_counter()-start)
            records.append(dict(device=device,batch=batch,p50_ms=float(np.percentile(elapsed,50)*1000),
                p95_ms=float(np.percentile(elapsed,95)*1000)))
    # Nonlinear sensitivity is only an iLQR approximation; report its discrepancy.
    layer = DifferentiableMPC(quad_parameters(env.quad),{"horizon_steps":5})
    x = torch.tensor(obs["mpc_state"])[None]; x[:,0]+=.1
    ref = torch.tensor(obs["reference"][:6])[None]
    previous = torch.zeros(1,4)
    cost = torch.zeros(1,layer.cost_size,dtype=torch.float64,requires_grad=True)
    direction = torch.zeros_like(cost); direction[0,17*5+13+17+2] = 1
    action = layer(x,ref,previous,cost,strict=True)[0,0]
    grad, = torch.autograd.grad(action,cost)
    with torch.no_grad():
        epsilon = .001
        numeric = ((layer(x,ref,previous,cost+epsilon*direction,strict=True)[0,0]-
                    layer(x,ref,previous,cost-epsilon*direction,strict=True)[0,0])/(2*epsilon)).item()
    result = dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, records=records,
        nonlinear_directional_derivative=dict(implicit=float((grad*direction).sum()),finite_difference=numeric),
        single_sample_realtime_100hz=any(r["batch"]==1 and r["p95_ms"]<=10 for r in records))
    (run_dir/"timing.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config",default="configs/acmpc_trajectory.yaml")
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=("preflight","compare","timing"),default="preflight")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--baseline-run",type=Path,default=Path("runs/ppo_trajectory/multitraj01"))
    parser.add_argument("--count",type=int,default=5)
    parser.add_argument("--methods",nargs="+",choices=("acmpc","fixed_mpc","ppo","acados"),default=["acmpc","fixed_mpc","ppo","acados"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    settings = load_training_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(settings,args.run_dir,args.device),indent=2))
    elif args.mode == "timing":
        print(json.dumps(timing(settings,args.run_dir),indent=2))
    else:
        compare(settings,args.run_dir,args.device,args.baseline_run,args.count,args.methods)


if __name__ == "__main__":
    main()
