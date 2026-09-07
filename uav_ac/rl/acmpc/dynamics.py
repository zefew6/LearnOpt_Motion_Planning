"""Batched NED/FRD rigid-body prediction with normalized wrench inputs."""
from __future__ import annotations

import torch
from torch import nn


class QuadrotorDynamics(nn.Module):
    def __init__(self, parameters: dict, dt: float, *, internal_controls=False):
        super().__init__()
        self.dt = float(dt)
        self.mass = float(parameters["mass"])
        self.gravity = float(parameters["gravity"])
        low, high = parameters["thrust_limits"]
        hover = self.mass * self.gravity
        if not 4 * low < hover < 4 * high:
            raise ValueError("hover must lie strictly inside thrust limits")
        self.hover = hover
        self.thrust_down = hover - 4 * low
        self.thrust_up = 4 * high - hover
        self.thrust_mid = 2 * (low + high)
        self.thrust_half = 2 * (high - low)
        self.internal_controls = internal_controls
        margin = min(hover / 4 - low, high - hover / 4)
        self.register_buffer("inertia", torch.tensor(parameters["inertia"], dtype=torch.float64))
        self.register_buffer("moment_scale", torch.tensor([
            4 * parameters["arm_length"] * margin,
            4 * parameters["arm_length"] * margin,
            4 * parameters["drag_to_thrust"] * margin,
        ], dtype=torch.float64))

    def _thrust(self, action):
        if self.internal_controls:
            thrust = self.thrust_mid + self.thrust_half * action[..., 0]
            slope = torch.full_like(action[..., 0], self.thrust_half)
        else:
            slope = torch.where(action[..., 0] <= 0, self.thrust_down, self.thrust_up)
            thrust = self.hover + action[..., 0] * slope
        return thrust, slope

    def wrench(self, action):
        thrust, _ = self._thrust(action)
        return torch.cat((thrust.unsqueeze(-1), action[..., 1:] * self.moment_scale), -1)

    def to_internal(self, action):
        thrust = self.hover + action[..., 0] * torch.where(action[..., 0] <= 0, self.thrust_down, self.thrust_up)
        return torch.cat((((thrust-self.thrust_mid)/self.thrust_half).unsqueeze(-1), action[..., 1:]), -1)

    def to_external(self, control):
        delta = self.thrust_mid + self.thrust_half * control[..., 0] - self.hover
        normalized = delta / torch.where(delta <= 0, self.thrust_down, self.thrust_up)
        return torch.cat((normalized.unsqueeze(-1), control[..., 1:]), -1)

    def derivative(self, state, action):
        q = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        w, x, y, z = q.unbind(-1)
        omega = state[..., 10:13]
        p, r, s = omega.unbind(-1)
        qdot = 0.5 * torch.stack((
            -x*p-y*r-z*s, w*p+y*s-z*r, w*r+z*p-x*s, w*s+x*r-y*p), -1)
        b3 = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
        wrench = self.wrench(action)
        gravity = torch.zeros_like(b3)
        gravity[..., 2] = self.gravity
        acceleration = gravity - wrench[..., :1] / self.mass * b3
        angular = (wrench[..., 1:] - torch.linalg.cross(omega, self.inertia * omega)) / self.inertia
        return torch.cat((state[..., 7:10], qdot, acceleration, angular), -1)

    @staticmethod
    def _normalize_with_jacobian(vector):
        """Match ``torch.nn.functional.normalize`` and return d(unit)/d(vector)."""
        eps = 1e-12
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        denominator = norm.clamp_min(eps)
        normalized = vector / denominator
        size = vector.shape[-1]
        identity = torch.eye(size, dtype=vector.dtype, device=vector.device)
        identity = identity.expand(*vector.shape[:-1], size, size)
        active = (norm > eps).to(vector.dtype)[..., None]
        projection = normalized[..., :, None] * normalized[..., None, :]
        jacobian = (identity - active * projection) / denominator[..., None]
        return normalized, jacobian

    def _derivative_with_jacobians(self, state, action):
        """Continuous rigid-body dynamics and analytic state/control Jacobians."""
        q, q_normalization = self._normalize_with_jacobian(state[..., 3:7])
        w, x, y, z = q.unbind(-1)
        omega = state[..., 10:13]
        p, r, s = omega.unbind(-1)

        qdot = 0.5 * torch.stack((
            -x*p-y*r-z*s, w*p+y*s-z*r, w*r+z*p-x*s, w*s+x*r-y*p), -1)
        b3 = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
        thrust, thrust_slope = self._thrust(action)
        moments = action[..., 1:] * self.moment_scale
        gravity = torch.zeros_like(b3)
        gravity[..., 2] = self.gravity
        acceleration = gravity - thrust[..., None] / self.mass * b3
        angular = (moments - torch.linalg.cross(omega, self.inertia * omega)) / self.inertia
        derivative = torch.cat((state[..., 7:10], qdot, acceleration, angular), -1)

        batch_shape = state.shape[:-1]
        a = torch.zeros(*batch_shape, 13, 13, dtype=state.dtype, device=state.device)
        b = torch.zeros(*batch_shape, 13, 4, dtype=state.dtype, device=state.device)
        identity3 = torch.eye(3, dtype=state.dtype, device=state.device)
        a[..., 0:3, 7:10] = identity3

        dqdot_dq = 0.5 * torch.stack((
            torch.stack((torch.zeros_like(p), -p, -r, -s), -1),
            torch.stack((p, torch.zeros_like(p), s, -r), -1),
            torch.stack((r, -s, torch.zeros_like(p), p), -1),
            torch.stack((s, r, -p, torch.zeros_like(p)), -1),
        ), -2)
        dqdot_domega = 0.5 * torch.stack((
            torch.stack((-x, -y, -z), -1),
            torch.stack((w, -z, y), -1),
            torch.stack((z, w, -x), -1),
            torch.stack((-y, x, w), -1),
        ), -2)
        a[..., 3:7, 3:7] = dqdot_dq @ q_normalization
        a[..., 3:7, 10:13] = dqdot_domega

        db3_dq = 2 * torch.stack((
            torch.stack((y, z, w, x), -1),
            torch.stack((-x, -w, z, y), -1),
            torch.stack((torch.zeros_like(w), -2*x, -2*y, torch.zeros_like(w)), -1),
        ), -2)
        a[..., 7:10, 3:7] = -(thrust / self.mass)[..., None, None] * (db3_dq @ q_normalization)
        b[..., 7:10, 0] = -(thrust_slope / self.mass)[..., None] * b3

        # d(omega x I*omega)/domega = -skew(I*omega) + skew(omega)*I.
        ip, ir, is_ = (self.inertia * omega).unbind(-1)
        zeros = torch.zeros_like(p)
        skew_iomega = torch.stack((
            torch.stack((zeros, -is_, ir), -1),
            torch.stack((is_, zeros, -ip), -1),
            torch.stack((-ir, ip, zeros), -1),
        ), -2)
        skew_omega = torch.stack((
            torch.stack((zeros, -s, r), -1),
            torch.stack((s, zeros, -p), -1),
            torch.stack((-r, p, zeros), -1),
        ), -2)
        cross_jacobian = -skew_iomega + skew_omega * self.inertia
        a[..., 10:13, 10:13] = -cross_jacobian / self.inertia[..., :, None]
        b[..., 10:13, 1:4] = torch.diag(self.moment_scale / self.inertia)
        return derivative, a, b

    def forward(self, state, action):
        h = self.dt
        k1 = self.derivative(state, action)
        k2 = self.derivative(state + h/2*k1, action)
        k3 = self.derivative(state + h/2*k2, action)
        k4 = self.derivative(state + h*k3, action)
        result = state + h/6*(k1 + 2*k2 + 2*k3 + k4)
        return torch.cat((result[..., :3], torch.nn.functional.normalize(
            result[..., 3:7], dim=-1), result[..., 7:]), -1)

    def grad_input(self, state, action):
        """Exact RK4 discrete-time Jacobians without constructing autograd graphs."""
        with torch.no_grad():
            h = self.dt
            identity = torch.eye(13, dtype=state.dtype, device=state.device)
            identity = identity.expand(*state.shape[:-1], 13, 13)

            k1, k1_x, k1_u = self._derivative_with_jacobians(state, action)
            stage2_x = identity + h/2*k1_x
            stage2_u = h/2*k1_u
            k2, f2_x, f2_u = self._derivative_with_jacobians(state + h/2*k1, action)
            k2_x = f2_x @ stage2_x
            k2_u = f2_x @ stage2_u + f2_u

            stage3_x = identity + h/2*k2_x
            stage3_u = h/2*k2_u
            k3, f3_x, f3_u = self._derivative_with_jacobians(state + h/2*k2, action)
            k3_x = f3_x @ stage3_x
            k3_u = f3_x @ stage3_u + f3_u

            stage4_x = identity + h*k3_x
            stage4_u = h*k3_u
            k4, f4_x, f4_u = self._derivative_with_jacobians(state + h*k3, action)
            k4_x = f4_x @ stage4_x
            k4_u = f4_x @ stage4_u + f4_u

            integrated = state + h/6*(k1 + 2*k2 + 2*k3 + k4)
            a = identity + h/6*(k1_x + 2*k2_x + 2*k3_x + k4_x)
            b = h/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)

            _, final_q_jacobian = self._normalize_with_jacobian(integrated[..., 3:7])
            a[..., 3:7, :] = final_q_jacobian @ a[..., 3:7, :]
            b[..., 3:7, :] = final_q_jacobian @ b[..., 3:7, :]
        return a, b
