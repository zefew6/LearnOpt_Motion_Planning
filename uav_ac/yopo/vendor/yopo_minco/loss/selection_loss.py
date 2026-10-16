"""Extracted from pinned YopoTrainer.loss_aggregation and _radius_loss."""
import torch
from torch.nn import functional as F

def candidate_loss(total_cost, score, safety_collided, inner_radio_offset, normalized_goal_norm, *, rank_weight=1., inner_reg_weight=3.):
    """Generation + Selection + inner-waypoint regularizer. Returns (loss, rank_loss, inner_reg_loss)."""
    B, N = score.shape
    score_flat = score.reshape(B * N)
    collided = safety_collided.view(B, N)                                # (B, N) bool

    # Generation: score-weighted cost, score DETACHED (weight only, no grad to score). Normalize
    # per image (over its N trajs) so obstacle-dense and open scenes contribute equally.
    cost_2d = total_cost.view(B, N)
    cost_mean = cost_2d.detach().mean(dim=1, keepdim=True).clamp(min=1e-3)
    gen_loss = ((cost_2d / cost_mean).reshape(-1) * score_flat.detach().clamp(min=1e-3)).mean()

    # Selection: top-1 CE over FEASIBLE trajs. Offset colliding trajs above every feasible one so
    # argmin picks the best feasible; a fully colliding image falls back to the least-bad traj.
    cost_det = total_cost.detach().view(B, N)
    rank_loss = F.cross_entropy(score_flat.view(B, N), (cost_det + (cost_det.max() + 1.0) * collided.float()).argmin(dim=1))

    # Inner-waypoint regularizer (centring, feasible trajs only): pull raw Δr toward a goal-adaptive target.
    dr_target = (normalized_goal_norm - 1.0).clamp(-1.0, 0.0)       # (B,) from normalized goal dist
    inner_reg_sq = (inner_radio_offset - dr_target[:, None]).pow(2)       # (B, N)
    inner_reg_loss = inner_reg_sq[~collided].mean() if (~collided).any() else inner_reg_sq.new_zeros(())

    loss = gen_loss + rank_weight * rank_loss + inner_reg_weight * inner_reg_loss
    return loss, rank_loss, inner_reg_loss

def radius_loss(radius_pred, radius_dist, *, warp_lambda=2., over_weight=3.):
    """Safety-corridor NLL: warp y = 1-exp(-d/λ), L = w·|y-μ|/b + log b (radius_pred = [μ; b]).
    b absorbs unpredictable samples; μ > y (over-estimating clearance) weighs radius_over_weight."""
    nr = radius_dist.shape[1]
    mu, b = radius_pred[:, :nr], radius_pred[:, nr:]
    y = 1.0 - torch.exp(-radius_dist.clamp(min=0.0) / warp_lambda)
    err = mu - y
    w = 1.0 + (over_weight - 1.0) * (err > 0).detach().float()   # heavier when μ > y
    return (w * err.abs() / b + b.log()).mean()
