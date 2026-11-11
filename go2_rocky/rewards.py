"""Tensor operations for training; actor observations match run_go2.observe."""

import torch


def rotate(q, vector):
    xyz = q[..., :3]
    cross = 2 * torch.cross(xyz, vector, dim=-1)
    return vector + q[..., 3:] * cross + torch.cross(xyz, cross, dim=-1)


def rotate_inverse(q, vector):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), vector)


def actor_observation(q, qd, rest, command, previous_action):
    gravity = torch.zeros_like(qd[:, :3])
    gravity[:, 2] = -1
    return torch.cat((rotate_inverse(q[:, 3:7], qd[:, :3]),
                      rotate_inverse(q[:, 3:7], qd[:, 3:6]),
                      rotate_inverse(q[:, 3:7], gravity), command,
                      q[:, 7:] - rest, qd[:, 6:], previous_action), -1)


def grid_height(heights, xy, size=12.0):
    """Piecewise triangular grid query matching native HFIELD (h00–h11)."""
    n = heights.shape[0]
    coordinates = torch.nan_to_num((xy+size/2)*(n-1)/size, nan=0).clamp(0, n-1-1e-5)
    ij = coordinates.long()
    fraction = coordinates-ij
    i, j = ij[:, 0], ij[:, 1]
    x, y = fraction[:, 0], fraction[:, 1]
    a, b = heights[j, i], heights[j, i+1]
    c, d = heights[j+1, i], heights[j+1, i+1]
    return torch.where(x >= y, a+x*(b-a)+y*(d-b), a+x*(d-c)+y*(c-a))


def reward_terms(q, qd, actions, previous_actions, target, rest, ke, kd, effort,
                 tangent, across, normal, speed, command, previous_qd, clearance, dt=0.02):
    """Slope-relative stability and progress, plus bounded effort regularizers.

    Torque is a PD estimate, not a measurement. No foot-slip reward is used
    until reliable stance/contact-force information is available in this env.
    """
    forward_v = (qd[:, :3] * tangent).sum(-1)
    lateral_v = (qd[:, :3] * across).sum(-1)
    yaw_v = (qd[:, 3:6] * normal).sum(-1)
    up = torch.zeros_like(normal)
    up[:, 2] = 1
    alignment = (rotate(q[:, 3:7], up) * normal).sum(-1)
    torque = (ke * (target - q[:, 7:]) - kd * qd[:, 6:]).clamp(-effort, effort)
    terms = {
        "velocity": 2 * torch.exp(-((forward_v-speed).square()+lateral_v.square()) / 0.25),
        "yaw": 0.5 * torch.exp(-(command[:, 2]-yaw_v).square() / 0.25),
        "progress": 0.5 * forward_v.clamp(-2, 2),
        "normal_velocity": -1.0 * (qd[:, :3] * normal).sum(-1).square(),
        "posture": -1.0 * (1-alignment).square(),
        "height": -2 * (clearance-0.34).square(),
        "angular_motion": -0.05 * ((qd[:, 3:6]*across).sum(-1).square()
                                     + (qd[:, 3:6]*tangent).sum(-1).square()),
        "effort_proxy": -2e-4 * torque.square().sum(-1),
        "joint_acceleration": -2.5e-7 * ((qd[:, 6:]-previous_qd)/dt).square().sum(-1),
        "action_rate": -0.01 * (actions-previous_actions).square().sum(-1),
    }
    return {name: value * dt for name, value in terms.items()}
