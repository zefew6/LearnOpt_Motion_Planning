"""Policy and controller deployment through the same task environment as training."""


def deploy(config):
    from uav_ac.experiments.runner import run
    return run(config)


__all__ = ["deploy"]
