"""Geometry checks need neither CUDA nor Newton."""

import math

import pytest

from go2_rocky.terrain import SlopeTerrain


DIRECTIONS = ("uphill", "downhill", "cross")


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_zero_slope_is_identity(direction):
    terrain = SlopeTerrain(direction=direction)
    point = (1.2, -3.4, 0.5)
    assert terrain.local_to_world(point) == pytest.approx(point)
    assert terrain.normal == pytest.approx((0, 0, 1))
    assert terrain.path_tangent == pytest.approx((0, 1, 0))
    assert terrain.height_at(2, 3) == pytest.approx(0)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_analytic_slope_height_and_normal(direction):
    terrain = SlopeTerrain(slope_deg=5, direction=direction, z0=0.4)
    theta = math.radians(5)
    c, s, gradient = math.cos(theta), math.sin(theta), math.tan(theta)
    if direction == "uphill":
        expected_height = 0.4 + 3 * gradient
        expected_normal = (0, -s, c)
    elif direction == "downhill":
        expected_height = 0.4 - 3 * gradient
        expected_normal = (0, s, c)
    else:
        expected_height = 0.4 + 2 * gradient
        expected_normal = (-s, 0, c)
    assert terrain.height_at(2, 3) == pytest.approx(expected_height)
    assert terrain.normal == pytest.approx(expected_normal)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("angle", (0, 5, 15))
def test_transformed_surface_matches_height_equation(direction, angle):
    terrain = SlopeTerrain(slope_deg=angle, direction=direction, z0=0.7)
    # Local w=0 means the point lies on the top surface.
    for u, v in ((0, 0), (1, 2), (-2, 3), (2, -1)):
        x, y, z = terrain.local_to_world((u, v, 0))
        assert z == pytest.approx(terrain.height_at(x, y))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_rotation_preserves_a_right_handed_orthonormal_frame(direction):
    terrain = SlopeTerrain(slope_deg=15, direction=direction)
    r = terrain.rotation_matrix
    across = tuple(row[0] for row in r)
    along = terrain.path_tangent
    normal = terrain.normal
    axes = (across, along, normal)
    for i, a in enumerate(axes):
        for j, b in enumerate(axes):
            assert sum(x * y for x, y in zip(a, b)) == pytest.approx(
                1 if i == j else 0, abs=1e-12
            )
    cross = (
        across[1] * along[2] - across[2] * along[1],
        across[2] * along[0] - across[0] * along[2],
        across[0] * along[1] - across[1] * along[0],
    )
    assert cross == pytest.approx(normal)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_transform_round_trip(direction):
    terrain = SlopeTerrain(slope_deg=15, direction=direction, z0=1.2)
    point = (1.3, -2.7, 0.6)
    assert terrain.world_to_local(terrain.local_to_world(point)) == pytest.approx(point)


@pytest.mark.parametrize("angle", (-5, 90, 100, math.nan, math.inf))
def test_invalid_slope_is_rejected(angle):
    with pytest.raises(ValueError, match="slope_deg"):
        SlopeTerrain(slope_deg=angle)


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction"):
        SlopeTerrain(direction="sideways")


@pytest.mark.parametrize("z0", (math.nan, math.inf, -math.inf))
def test_invalid_origin_height_is_rejected(z0):
    with pytest.raises(ValueError, match="z0"):
        SlopeTerrain(z0=z0)
