"""Rock coverage must change collision elevations across the mountain."""

import newton
import numpy as np
import pytest

from go2_rocky.mountain import MountainConfig, MountainTerrain
from go2_rocky.rocks import MOUNTAIN_COUNTS, RockConfig, generate_mountain_rocks
from go2_rocky.surface import RockSurface


@pytest.fixture(scope="module")
def dense_mountain():
    return MountainTerrain(MountainConfig(resolution_m=0.4))


def test_embedded_rocks_cover_the_landscape_and_settling_patch_stays_clear(dense_mountain):
    terrain = dense_mountain
    x, y = np.meshgrid(terrain.axis, terrain.axis)
    interior = (np.abs(x) < 32) & (np.abs(y) < 32)
    assert np.mean(terrain.surface_relief_m[interior] > 0.002) > 0.95
    start = terrain.route[0]
    assert terrain.height_at(*start[:2]) == pytest.approx(start[2], abs=1e-5)
    pad = np.hypot(x - start[0], y - start[1]) < 0.5
    assert np.max(terrain.surface_relief_m[pad]) == 0
    assert terrain.surface.cell_count > 10000


def test_surface_multiplier_changes_geometry_not_just_material(dense_mountain):
    smooth = MountainTerrain(MountainConfig(resolution_m=0.4, surface_rockiness=0))
    assert np.count_nonzero(smooth.surface_relief_m) == 0
    np.testing.assert_allclose(dense_mountain.heights - smooth.heights,
                               dense_mountain.surface_relief_m, atol=4e-6)


def test_display_triangles_are_exactly_the_physical_surface(dense_mountain):
    terrain = dense_mountain
    points, indices, normals, uvs = terrain.visual_arrays()
    np.testing.assert_allclose(points[:, 2], terrain.height_at(points[:, 0], points[:, 1]), atol=4e-5)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    assert np.all(normals[:, 2] > 0)
    assert len(indices) == 6 * (terrain.n - 1)**2
    assert np.all((uvs >= 0) & (uvs <= 1))
    builder = newton.ModelBuilder()
    shape = terrain.add_collision(builder, textured=True)
    assert builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert not builder.shape_flags[shape] & int(newton.ShapeFlags.VISIBLE)


def test_material_is_seeded_and_has_real_color_variation():
    surface = RockSurface(1, 80, "medium")
    image = surface.material(96)
    assert image.dtype == np.uint8
    assert image.shape == (96, 96, 3)
    assert np.std(image) > 8
    np.testing.assert_array_equal(image, RockSurface(1, 80, "medium").material(96))
    assert not np.array_equal(image, RockSurface(2, 80, "medium").material(96))


def test_render_material_explicitly_enables_its_texture(dense_mountain):
    class Viewer:
        def log_mesh(self, name, **kwargs):
            self.prototype = name
            assert kwargs["hidden"]
            assert kwargs["texture"] is not None

        def log_instances(self, name, **kwargs):
            assert kwargs["mesh"] == self.prototype
            np.testing.assert_allclose(kwargs["materials"].numpy(), [(1, 0, 0, 1)])
            np.testing.assert_allclose(kwargs["colors"].numpy(), [(1, 1, 1)])

    dense_mountain.log_visual(Viewer(), "cpu")


def test_mountain_scatter_includes_foreground_and_background_boulders(dense_mountain):
    config = RockConfig(width_m=80, length_m=80, distribution="mountain", start_clearance_m=0.9)
    rocks = generate_mountain_rocks(dense_mountain, config)
    assert len(rocks) == 1400
    background = [r for r in rocks if r.coordinate_frame == "world_xy"]
    assert len(background) == 1200
    assert any(r.coordinate_frame == "route" for r in rocks)
    centres = np.asarray([r.centre_local_m[:2] for r in background])
    assert np.ptp(centres[:, 0]) > 40
    assert np.ptp(centres[:, 1]) > 40
    assert all(np.linalg.norm(dense_mountain.route[:, :2] - p, axis=1).min() > 0.35 + r.bounding_radius_m
               for p, r in zip(centres, background))
    assert any(np.linalg.norm(dense_mountain.route[:, :2] - p, axis=1).min() - r.bounding_radius_m < 1.0
               for p, r in zip(centres, background))
    route = [r for r in rocks if r.coordinate_frame == "route"]
    assert min(r.centre_local_m[1] - r.bounding_radius_m for r in route) < 1.3
    assert all(r.embedded_fraction > 0 for r in rocks)
    assert rocks == generate_mountain_rocks(dense_mountain, config)


@pytest.mark.parametrize("roughness", MOUNTAIN_COUNTS)
def test_default_counts_are_landscape_counts(roughness):
    assert RockConfig(roughness=roughness, distribution="mountain").count == MOUNTAIN_COUNTS[roughness]
