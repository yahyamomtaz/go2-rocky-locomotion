"""Check diagnostic clearances against known sphere/plane distances."""

import numpy as np
import pytest

from go2_rocky.analyze_trajectory import point_gap, recorded_surface, support_normal


@pytest.mark.parametrize("gradient", ((0., 0.), (0.3, -0.4)))
def test_recorded_grid_sphere_gap_and_support_normal(gradient):
    axis = np.linspace(-2, 2, 5)
    x, y = np.meshgrid(axis, axis)
    a, b = gradient
    terrain = recorded_surface({"mountain_elevation_m": a*x+b*y,
                                "mountain_xy_axis_m": axis,
                                "route_world_m": np.array(((0., -1., -b), (0., 1., b)))})
    normal = np.array((-a, -b, 1.)) / np.sqrt(1+a*a+b*b)
    fitted, _ = support_normal(terrain, (0, 0, 0.3))
    np.testing.assert_allclose(fitted, normal, atol=1e-7)
    for signed_distance in (-0.01, 0.05, 0.15):
        assert point_gap(terrain, signed_distance * normal, 0.1) == pytest.approx(
            signed_distance - 0.1, abs=1e-7)
