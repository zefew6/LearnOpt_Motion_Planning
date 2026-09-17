"""Online gate geometry targets; independent of trajectory-tracking references."""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RacingSettings:
    cruise_speed: float = 7.
    minimum_speed: float = 2.
    lateral_acceleration: float = 5.
    braking_acceleration: float = 3.
    clearance_margin: float = .02
    vehicle_diameter: float = .45

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
               for v in vars(self).values()):
            raise ValueError("racing settings must be positive finite numbers")
        if not self.minimum_speed <= self.cruise_speed <= 8:
            raise ValueError("racing speed must satisfy minimum_speed <= cruise_speed <= 8")


def body_rotation(q):
    w, x, y, z = torch.nn.functional.normalize(q, dim=-1).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def gate_targets(state, features, steps, dt, settings):
    """Decode relative corners and sample an ordered entry/center/exit route.

    Targets are recomputed on every control tick, not accumulated episode state.
    Actual gate switching is owned exclusively by the physical task.
    """
    cfg = settings
    rotation = body_rotation(state[:, 3:7])
    corners = features[:, 14:38].double().reshape(-1, 2, 4, 3) * 10
    corners = corners @ rotation[:, None].transpose(-1, -2)
    centers = corners.mean(-2)
    ys = corners[:, :, 1]-corners[:, :, 0]
    zs = corners[:, :, 3]-corners[:, :, 0]
    sizes = torch.stack((ys.norm(dim=-1), zs.norm(dim=-1)), -1)/2
    y = torch.nn.functional.normalize(ys, dim=-1)
    z = torch.nn.functional.normalize(zs, dim=-1)
    normal = torch.linalg.cross(y, z)
    frame = torch.stack((normal[:, 0], y[:, 0], z[:, 0]), -1)
    active = features[:, -2] > .5
    has_next = features[:, -1] > .5
    identity = torch.eye(3, dtype=state.dtype, device=state.device).expand(len(state), -1, -1)
    frame = torch.where(active[:, None, None], frame, identity)
    n0, n1 = normal.unbind(1)
    c0, c1 = centers.unbind(1)
    d = cfg.vehicle_diameter
    signed = -(c0*n0).sum(-1)
    entry = torch.where((signed < -d)[:, None], c0-d*n0, c0)
    exit0 = c0+d*n0
    next_entry = torch.where(has_next[:, None], c1-d*n1, exit0+20*n0)
    next_center = torch.where(has_next[:, None], c1, next_entry+20*n0)
    next_exit = torch.where(has_next[:, None], c1+d*n1, next_center+20*n0)
    nodes = torch.stack((torch.zeros_like(c0), entry, c0, exit0, next_entry,
                         next_center, next_exit, next_exit+20*torch.where(has_next[:, None], n1, n0)), 1)
    segments = nodes[:, 1:]-nodes[:, :-1]
    lengths = segments.norm(dim=-1).clamp_min(1e-8)
    directions = segments/lengths[..., None]
    cumulative = torch.cat((torch.zeros_like(lengths[:, :1]), lengths.cumsum(-1)), -1)
    aperture_ratio = 2*sizes[:, 0].amin(-1)/d
    aperture_speed = cfg.minimum_speed + (cfg.cruise_speed-cfg.minimum_speed)*((aperture_ratio-1.3)/2.7).clamp(0, 1)
    outgoing = torch.nn.functional.normalize(next_entry-exit0, dim=-1)
    cos_turn = (n0*outgoing).sum(-1).clamp(-1, 1)
    radius = (c1-c0).norm(dim=-1) / (2*((1-cos_turn)/2).clamp_min(1e-6).sqrt())
    turn_speed = (cfg.lateral_acceleration*radius).sqrt()
    gate_speed = torch.minimum(aperture_speed, torch.where(has_next, turn_speed, aperture_speed))
    clearance = (sizes[:, 0]-d/2-cfg.clearance_margin).clamp_min(.02)
    transverse = torch.stack(((c0*y[:, 0]).sum(-1), (c0*z[:, 0]).sum(-1)), -1)
    alignment_speed = cfg.cruise_speed / (1+(transverse/clearance).norm(dim=-1)*.15)
    gate_speed = gate_speed.clamp(min=cfg.minimum_speed, max=cfg.cruise_speed)
    # Braking envelope is reevaluated at every predicted route position.
    position = torch.zeros(len(state), dtype=state.dtype, device=state.device)
    velocity = state[:, 7:10].norm(dim=-1).clamp(max=cfg.cruise_speed)
    positions, velocities = [], []
    for k in range(steps+1):
        remaining = (cumulative[:, 2]-position-d).clamp_min(0)
        cap = torch.minimum((gate_speed.square()+2*cfg.braking_acceleration*remaining).sqrt(), alignment_speed)
        cap = cap.clamp(min=cfg.minimum_speed, max=cfg.cruise_speed)
        if k:
            velocity = torch.maximum(torch.minimum(cap, velocity+cfg.braking_acceleration*dt),
                                     velocity-cfg.braking_acceleration*dt)
            position = position+velocity*dt
        segment = (position[:, None] >= cumulative[:, 1:]).sum(-1).clamp(max=segments.shape[1]-1)
        index = torch.arange(len(state), device=state.device)
        direction = directions[index, segment]
        target = nodes[index, segment]+direction*(position-cumulative[index, segment])[:, None]
        positions.append(target)
        velocities.append(direction*velocity[:, None])
    target_position = torch.stack(positions, 1)
    target_velocity = torch.stack(velocities, 1)
    target_position = torch.where(active[:, None, None], target_position, 0.)
    target_velocity = torch.where(active[:, None, None], target_velocity, 0.)
    return target_position, target_velocity, frame, clearance, gate_speed
