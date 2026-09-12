"""Trajectory-bank ownership for trajectory tracking."""

from pathlib import Path
from typing import Any

from uav_ac.control.rl_controller import quad_parameters
from uav_ac.simulation.mujoco_sim import OPEN_FIELD_SCENE_PATH, MujocoSimulation
from ...common.trajectory_bank import (
    TRAJECTORY_BANK_DIRECTORY,
    TrajectoryBank,
    generate_trajectory_bank,
)
from .config import DEFAULT_STEPS_PER_ACTION, DEFAULT_SEED, DEFAULT_TRAINING_CONFIG

def prepare_assets(
        run_dir: Path,
        *,
        settings: dict[str, Any] | None = None,
        steps_per_action: int = DEFAULT_STEPS_PER_ACTION,
        seed: int = DEFAULT_SEED,
        model_path: str | Path = OPEN_FIELD_SCENE_PATH,
) -> tuple[TrajectoryBank, dict[str, Any]]:
    """Generate or load the strict MPC-validated trajectory bank."""
    settings = DEFAULT_TRAINING_CONFIG if settings is None else settings
    external = settings.get("trajectory_bank_path")
    if external:
        from ...common.trajectory_bank import trajectory_bank_fingerprint
        bank = TrajectoryBank.load(Path(external))
        expected = trajectory_bank_fingerprint(
            settings["trajectory_bank"], scene_path=model_path,
            steps_per_action=steps_per_action)
        if bank.metadata.get("fingerprint") != expected:
            raise ValueError("external trajectory bank fingerprint does not match configuration")
    else:
        bank = generate_trajectory_bank(
            run_dir / TRAJECTORY_BANK_DIRECTORY,
            settings["trajectory_bank"], steps_per_action=steps_per_action, seed=seed,
            scene_path=model_path)
    simulation = MujocoSimulation(model_path, record_actual_trajectory=False)
    return bank, quad_parameters(simulation.quad)
