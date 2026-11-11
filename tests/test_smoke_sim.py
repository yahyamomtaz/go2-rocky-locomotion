"""Primitive checks use MuJoCo C on CPU; verify Warp separately with the CLI."""

import numpy as np
import pytest
import warp as wp
import newton

from go2_rocky.drop_sphere import SphereDrop, add_rigid_bed
from go2_rocky.terrain import SlopeTerrain


@pytest.mark.parametrize("direction", ("uphill", "downhill", "cross"))
def test_collision_bed_top_matches_analytic_surface(direction):
    terrain = SlopeTerrain(5, direction, z0=0.3)
    builder = newton.ModelBuilder()
    shape = add_rigid_bed(builder, terrain)
    model = builder.finalize(device="cpu")
    assert model.shape_body.numpy()[shape] == -1
    flags = int(model.shape_flags.numpy()[shape])
    assert flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert flags & int(newton.ShapeFlags.VISIBLE)
    transform = wp.transform(*model.shape_transform.numpy()[shape])
    # Box has half extents (2, 4, 0.1). Its top corners must match z(x,y).
    for u in (-2, 2):
        for v in (-4, 4):
            x, y, z = wp.transform_point(transform, wp.vec3(u, v, 0.1))
            assert z == pytest.approx(terrain.height_at(x, y), abs=1e-6)


@pytest.mark.parametrize("angle,direction", ((0, "uphill"), (5, "uphill"), (5, "downhill"), (5, "cross")))
def test_sphere_contacts_and_stays_on_bed(angle, direction):
    sim = SphereDrop(SlopeTerrain(angle, direction), device="cpu")
    for _ in range(round(2.0 / sim.dt)):
        sim.step()
    sim.check()
    local = sim.arrays()["sphere_position_local_m"][-1]
    if angle == 0:
        np.testing.assert_allclose(local[:2], (0, 0), atol=1e-5)
    elif direction == "uphill":
        assert local[1] < -0.1
    elif direction == "downhill":
        assert local[1] > 0.1
    else:
        assert local[0] < -0.1
