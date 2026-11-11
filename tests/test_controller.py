"""Command-frame and steering-sign checks for the outer path controller."""

import math

import numpy as np
import pytest
import warp as wp

from go2_rocky.controller import path_command
from go2_rocky.terrain import SlopeTerrain


def state(terrain, u=0.0, heading_offset=0.0):
    bed_rotation = wp.quat_from_matrix(wp.mat33(*(
        v for row in terrain.rotation_matrix for v in row
    )))
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2 + heading_offset)
    orientation = wp.mul(bed_rotation, yaw)
    return np.array([*terrain.local_to_world((u, -3.0, 0.35)), *orientation, *([0.0]*12)])


@pytest.mark.parametrize("direction", ("uphill", "downhill", "cross"))
def test_aligned_robot_gets_forward_command_on_any_slope(direction):
    terrain = SlopeTerrain(15, direction)
    command = path_command(terrain, state(terrain), np.zeros(18), 0.6)
    np.testing.assert_allclose(command, (0.6, 0, 0), atol=1e-6)


@pytest.mark.parametrize("u", (-0.2, 0.2))
def test_lateral_command_points_back_toward_centre(u):
    terrain = SlopeTerrain(15)
    # Facing +v means body +Y (left) points toward terrain -u.
    command = path_command(terrain, state(terrain, u=u), np.zeros(18), 0.6)
    assert command[1] * u > 0
    assert command[0] == pytest.approx(0.6, abs=1e-6)


@pytest.mark.parametrize("offset", (-0.4, 0.4, -3.1, 3.1))
def test_yaw_correction_turns_back_toward_reference_heading(offset):
    terrain = SlopeTerrain(15)
    command = path_command(terrain, state(terrain, heading_offset=offset), np.zeros(18), 0.6)
    assert command[2] * offset < 0
    assert abs(command[2]) <= 0.8


def test_lateral_command_is_bounded_and_yaw_damps_rotation():
    terrain = SlopeTerrain(15)
    qd = np.zeros(18)
    qd[3:6] = np.asarray(terrain.normal) * 2.0
    command = path_command(terrain, state(terrain, u=10), qd, 0.6)
    assert command[1] == pytest.approx(0.25, abs=1e-6)
    assert command[2] == pytest.approx(-0.4, abs=1e-6)


@pytest.mark.parametrize("offset", (-0.8, 0.8))
def test_body_frame_lateral_command_is_bounded_even_when_heading_is_wrong(offset):
    terrain = SlopeTerrain(15)
    command = path_command(terrain, state(terrain, heading_offset=offset), np.zeros(18), 1.0)
    assert abs(command[1]) <= 0.25

