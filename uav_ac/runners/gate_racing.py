"""Replay a reference-free gate-racing policy in the native viewer."""

from pathlib import Path


def run_gate_racing(config: dict):
    """Replay a reference-free gate-racing policy in the native viewer."""
    from uav_ac.rl.tasks.gate_racing.evaluation import replay

    checkpoint = Path(config["rl"]["checkpoint"])
    result = replay(
        checkpoint.parent,
        device=config["rl"]["device"],
        seed=config["seed"],
        checkpoint=checkpoint,
        follow_camera=config.get("follow_camera", True),
    )
    print(f"Finished gate racing: success={result['success_rate']:.3f} | "
          f"gates={result['mean_gates_passed']:.2f} | "
          f"collision={result['collision_rate']:.3f}")
    return result

