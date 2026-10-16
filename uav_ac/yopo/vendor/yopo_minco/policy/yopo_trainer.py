"""Pinned YopoTrainer numerical path; explicit IO/orchestration lives in uav.training.

The body/world transform, complete trajectory costs and diagnostics are upstream
methods. Selection/radius methods delegate to extracted upstream functions shared
with the whole-body extension.
"""
import torch
import numpy as np
from uav_ac.yopo.vendor.yopo_minco.config.config import cfg
from uav_ac.yopo.vendor.yopo_minco.loss.loss_function import YOPOLoss
from uav_ac.yopo.vendor.yopo_minco.loss.selection_loss import candidate_loss, radius_loss
from uav_ac.yopo.vendor.yopo_minco.policy.yopo_network import YopoNetwork
from uav_ac.yopo.vendor.yopo_minco.policy.state_transform import *

class YopoTrainer:
    def __init__(self, batch_size):
        self.device = cfg.device
        self.batch_size = batch_size
        self.traj_num = cfg['traj_num']
        self.rank_weight = float(cfg['rank_weight'])
        self.inner_reg_weight = float(cfg['w_inner_reg'])
        self.radius_weight = float(cfg['w_radius'])
        self.radius_lambda = float(cfg['radius_warp_lambda'])
        self.radius_over_weight = float(cfg['radius_over_weight'])
        self.policy = YopoNetwork().to(self.device)
        self.yopo_loss = YOPOLoss()

    def loss_aggregation(self, total_cost, score, safety_collided, inner_radio_offset, obs_b):
        return candidate_loss(total_cost.view(self.batch_size,self.traj_num),
            score.reshape(self.batch_size,self.traj_num),safety_collided.view(self.batch_size,self.traj_num),
            inner_radio_offset,obs_b[:,6:9].norm(dim=-1),
            rank_weight=self.rank_weight,inner_reg_weight=self.inner_reg_weight)

    def _radius_loss(self, radius_pred, radius_dist):
        return radius_loss(radius_pred,radius_dist,warp_lambda=self.radius_lambda,over_weight=self.radius_over_weight)

    def forward_and_compute_loss(self, depth, pos, rot, obs_b, map_id):
        depth, pos, rot, obs_b, map_id = [x.to(self.device) for x in [depth, pos, rot, obs_b, map_id]]

        # 1. network forward + body→world lift
        traj = self._predict_world(depth, pos, rot, obs_b)

        # 2. weighted trajectory costs
        costs, aux = self.yopo_loss(traj["head_w"], traj["tail_w"], traj["inner_w"], traj["goal_w"], map_id,
                                    durations=traj["durations"])
        total_cost = sum(costs.values())

        # 3. training loss: generation + selection (rank CE) + inner-waypoint reg + safety-corridor NLL
        loss, rank_loss, inner_reg_loss = self.loss_aggregation(
            total_cost, traj["score"], aux["collided"], traj["inner_dr"], obs_b)
        radius_loss = self._radius_loss(traj["radius"], aux["radius_dist"])
        loss = loss + self.radius_weight * radius_loss

        # 4. tensorboard diagnostics
        metrics, pred_rank = self._diagnostics(
            costs, total_cost, aux["stats"], traj["score"], rank_loss, inner_reg_loss, radius_loss)
        return loss, metrics, pred_rank

    def _predict_world(self, depth, pos, rot, obs_b):
        """Network forward + body→world lift. Returns per-trajectory (B*N) world-frame
        head/tail/inner PVA and goal, plus durations, the raw score grid, and the raw inner Δr.
        NOTE: policy.inference normalizes obs_b in place."""
        BN = self.batch_size * self.traj_num
        # MUST read the raw obs_b BEFORE inference — policy.inference normalizes obs_b in place.
        goal_w, start_vel_w, start_acc_w = state_body2world(pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
        inner_b, tail_pva_b, durations, score, inner_dr, radius = self.policy.inference(depth, obs_b)

        inner_b = inner_b.reshape(BN, 3)
        tail_b = tail_pva_b.reshape(BN, 3, 3)                            # rows: pos, vel, acc
        # expand per-image pos/rot to per-trajectory: (B, ...) → (B*N, ...)
        pos_ex = pos.repeat_interleave(self.traj_num, dim=0)
        rot_ex = rot.repeat_interleave(self.traj_num, dim=0)

        inner_w = transform_body2world(rot_ex, pos_ex, inner_b)
        tail_w = torch.stack([transform_body2world(rot_ex, pos_ex, tail_b[:, 0]),
                              rotate_body2world(rot_ex, tail_b[:, 1]),
                              rotate_body2world(rot_ex, tail_b[:, 2])], dim=1)   # [B*N, 3, 3]
        head_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1).repeat_interleave(self.traj_num, dim=0)
        return {"head_w": head_w, "tail_w": tail_w, "inner_w": inner_w,
                "goal_w": goal_w.repeat_interleave(self.traj_num, dim=0),
                "durations": durations.reshape(BN, 2),
                "score": score, "inner_dr": inner_dr, "radius": radius.reshape(BN, -1)}

    def _diagnostics(self, costs, total_cost, traj_stats, score, rank_loss, inner_reg_loss, radius_loss):
        """Detached metrics for tensorboard (internal keys mapped to tags by _train_tag / eval_one_epoch).
        best/* = the selected (max-score) traj per image; mean/* = over all grids; perf/* = best-traj
        performance. Returns (metrics, pred_rank); pred_rank = rank of the selected traj in the true cost order."""
        B, N = self.batch_size, self.traj_num
        best_idx = score.detach().reshape(B, N).argmax(dim=1)                 # (B,)
        def _best(c):  # (B*N,) → per-image best-scoring grid, averaged over batch
            return c.detach().view(B, N).gather(1, best_idx[:, None]).squeeze(1).mean()

        metrics = {}
        for name, c in costs.items():                       # per-component loss: best traj vs mean over all
            metrics[f"best/{name}"] = _best(c)
            metrics[f"mean/{name}"] = c.detach().mean()
        for name, v in traj_stats.items():                  # best-traj performance (Duration/PathLength/AvgSpeed/MaxSpeed/MinDist)
            metrics[f"perf/{name}"] = _best(v)
        metrics["mean_traj_loss"] = total_cost.detach().mean()
        metrics["rank_loss"] = rank_loss.detach()
        metrics["inner_reg_loss"] = inner_reg_loss.detach()
        metrics["radius_loss"] = radius_loss.detach()

        # rank of the predicted-best traj in the true cost order (0 = true optimum)
        cost_2d = total_cost.detach().view(B, N)
        selected = cost_2d.gather(1, best_idx[:, None])
        pred_rank = (cost_2d < selected).sum(dim=1)                           # (B,) 0-based
        return metrics, pred_rank
