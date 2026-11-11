"""Check our policy adapter against the installed Newton 1.6 reference."""

import math

import numpy as np
import pytest
import warp as wp

from newton.examples.robot.example_robot_policy import _compute_obs_kernel

from go2_rocky.run_go2 import joint_targets, observe
from go2_rocky.terrain import SlopeTerrain


def test_observations_match_upstream_contract():
    rest = wp.array(np.linspace(-0.5, 0.5, 12), dtype=wp.float32, device="cpu")
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.4)
    q = wp.array([1.0, 2.0, 3.0, *rotation, *np.linspace(-1, 1, 12)], dtype=wp.float32, device="cpu")
    qd = wp.array(np.linspace(-2, 2, 18), dtype=wp.float32, device="cpu")
    previous = wp.array(np.linspace(-0.3, 0.7, 12).reshape(1, 12), dtype=wp.float32, device="cpu")
    command = wp.vec3(0.3, 0.0, 0.0)
    ours = wp.zeros((1, 48), dtype=wp.float32, device="cpu")
    reference = wp.zeros_like(ours)
    identity_order = wp.array(np.arange(12), dtype=wp.int32, device="cpu")
    wp.launch(observe, dim=1, inputs=[q, qd, rest, previous, command, ours], device="cpu")
    wp.launch(_compute_obs_kernel, dim=1, inputs=[
        q, qd, rest, identity_order, wp.vec3(0.0, 0.0, -1.0), command,
        previous, 12, reference,
    ], device="cpu")
    np.testing.assert_allclose(ours.numpy(), reference.numpy(), atol=1e-7)
    np.testing.assert_allclose(ours.numpy()[0, 9:12], [0.3, 0.0, 0.0], atol=1e-7)


def test_actions_are_joint_offsets_not_base_motion():
    rest = np.linspace(-0.5, 0.5, 12).astype(np.float32)
    action = np.linspace(-1, 1, 12).astype(np.float32)
    target = wp.zeros(19, dtype=wp.float32, device="cpu")
    wp.launch(joint_targets, dim=12, inputs=[
        wp.array(action.reshape(1, 12), dtype=wp.float32, device="cpu"),
        wp.array(rest, dtype=wp.float32, device="cpu"), 0.25, target,
    ], device="cpu")
    np.testing.assert_allclose(target.numpy()[:7], 0.0)
    np.testing.assert_allclose(target.numpy()[7:], rest + 0.25 * action)


@pytest.mark.parametrize("direction", ("uphill", "downhill", "cross"))
def test_spawn_heading_points_along_path(direction):
    terrain = SlopeTerrain(5, direction)
    rotation = wp.quat_from_matrix(wp.mat33(*(
        value for row in terrain.rotation_matrix for value in row
    )))
    heading = wp.mul(rotation, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2))
    forward = wp.quat_rotate(heading, wp.vec3(1.0, 0.0, 0.0))
    np.testing.assert_allclose(tuple(forward), terrain.path_tangent, atol=1e-6)
