"""Optional path tracking around the pretrained velocity-command policy.

This outer controller uses simulator pose and a known reference path. It is
not a learned terrain-aware policy; it only supplies the actor's existing
forward/lateral/yaw command inputs. It never applies forces or changes poses.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathControllerConfig:
    heading_gain: float = 2.0
    yaw_damping: float = 0.2
    lateral_gain: float = 0.8
    max_lateral_speed: float = 0.25
    max_yaw_rate: float = 0.8
    max_body_lateral_speed: float = 0.25


def path_command(terrain, q, qd, speed, config=PathControllerConfig()):
    """Track terrain-local u=0 and point along +v, with bounded commands.

    The target velocity lies in the bed plane. Transform it into the current
    body frame before sending it to the policy. Heading feedback uses the
    projection of body +X onto that plane; yaw damping uses actual angular
    velocity about the bed normal.
    """
    x, y, z, w = q[3:7]
    forward = np.array((1 - 2 * (y*y + z*z), 2 * (x*y + z*w), 2 * (x*z - y*w)))
    left = np.array((2 * (x*y - z*w), 1 - 2 * (x*x + z*z), 2 * (y*z + x*w)))
    rotation = np.asarray(terrain.frame_at(q[:3]))
    cross_track = terrain.path_state(q[:3])[0]
    local_forward = rotation.T @ forward
    heading_error = math.atan2(local_forward[0], local_forward[1])
    lateral = float(np.clip(-config.lateral_gain * cross_track,
                            -config.max_lateral_speed, config.max_lateral_speed))
    desired_world = rotation @ np.array((lateral, speed, 0.0))
    yaw_velocity = float(np.dot(qd[3:6], rotation[:, 2]))
    yaw_command = float(np.clip(config.heading_gain * heading_error - config.yaw_damping * yaw_velocity,
                                -config.max_yaw_rate, config.max_yaw_rate))
    body_lateral = float(np.clip(np.dot(desired_world, left),
                                  -config.max_body_lateral_speed, config.max_body_lateral_speed))
    return (float(np.dot(desired_world, forward)), body_lateral, yaw_command)
