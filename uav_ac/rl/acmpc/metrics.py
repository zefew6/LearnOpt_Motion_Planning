"""Shared physical diagnostics; no Torch or solver dependency."""
import numpy as np


def command_to_action(command, quad):
    hover = quad.m * quad.g
    delta = command.thrust - hover
    authority = hover - 4*quad.min_thrust if delta <= 0 else 4*quad.max_thrust-hover
    margin = min(hover/4-quad.min_thrust, quad.max_thrust-hover/4)
    scale = 4*margin*np.array([quad.l, quad.l, quad.kappa])
    return np.clip(np.r_[delta/authority, command.moment/scale], -1, 1).astype(np.float32)


def allocation_scale(command, quad):
    c = np.clip(command.thrust,4*quad.min_thrust,4*quad.max_thrust)/4
    moment = quad.propeller_coeffs() @ np.r_[command.moment[:2]/quad.l, -command.moment[2]/quad.kappa, 0.] / 4
    positive = moment > 0
    negative = moment < 0
    limits = np.ones(4)
    limits[positive] = (quad.max_thrust-c)/moment[positive]
    limits[negative] = (quad.min_thrust-c)/moment[negative]
    return float(np.clip(limits.min(),0,1))
