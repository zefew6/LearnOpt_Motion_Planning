"""Train the configured RL policy."""

from .training import load_training_config, main, prepare_assets, train

__all__ = ["load_training_config", "main", "prepare_assets", "train"]

if __name__ == "__main__":
    main()
