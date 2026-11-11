"""Reproducibility, terrain safety, and real collision checks for fixed rocks."""

from collections import Counter
import math

import newton
import numpy as np
import pytest
import warp as wp

from go2_rocky.drop_sphere import GRAVITY, add_rigid_bed
from go2_rocky.rocks import PRESETS, RockConfig, add_static_rocks, generate_rocks
from go2_rocky.terrain import SlopeTerrain


def test_same_seed_reproduces_all_geometry_and_materials():
    config = RockConfig(rock_count=12, seed=7, friction_range=(0.6, 0.9))
    assert generate_rocks(config) == generate_rocks(config)
    assert generate_rocks(config) != generate_rocks(RockConfig(rock_count=12, seed=8))


@pytest.mark.parametrize("roughness", PRESETS)
def test_presets_have_safe_nonoverlapping_supported_rocks(roughness):
    config = RockConfig(roughness=roughness)
    rocks = generate_rocks(config)
    assert len(rocks) == config.count
    for i, rock in enumerate(rocks):
        u, v, w = rock.centre_local_m
        radius = rock.bounding_radius_m
        assert w == 0
        assert abs(u) + radius < config.width_m / 2
        assert v - radius >= -config.length_m / 2 + config.start_clearance_m
        assert v + radius <= config.length_m / 2 - config.target_clearance_m
        assert abs(u - config.bypass_centre_u) >= 0.35 + radius
        vertices = np.asarray(rock.vertices_m)
        assert vertices[:, 2].min() == 0
        assert vertices[:, 2].max() == pytest.approx(rock.height_m)
        assert np.linalg.norm(vertices[:, :2], axis=1).max() <= radius
        for other in rocks[:i]:
            assert math.dist((u, v), other.centre_local_m[:2]) >= radius + other.bounding_radius_m + 0.025
        # A closed convex hull: each edge is shared by exactly two triangles,
        # and every face points away from a point inside the volume.
        edges = Counter()
        interior = vertices.mean(axis=0)
        for a, b, c in rock.triangles:
            for p, q in ((a, b), (b, c), (c, a)):
                edges[tuple(sorted((p, q)))] += 1
            normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
            assert np.dot(normal, vertices[a] - interior) > 0
        assert set(edges.values()) == {2}


@pytest.mark.parametrize("direction", ("uphill", "downhill", "cross"))
def test_render_and_collision_shapes_share_supported_slope_transform(direction):
    terrain = SlopeTerrain(20, direction, z0=0.8)
    rocks = generate_rocks(RockConfig(rock_count=4))
    builder = newton.ModelBuilder()
    ids = add_static_rocks(builder, terrain, rocks)
    model = builder.finalize(device="cpu")
    transforms = model.shape_transform.numpy()
    for shape, rock in zip(ids, rocks):
        assert model.shape_body.numpy()[shape] == -1
        assert model.shape_type.numpy()[shape] == int(newton.GeoType.MESH)
        flags = int(model.shape_flags.numpy()[shape])
        assert flags & int(newton.ShapeFlags.COLLIDE_SHAPES)
        assert flags & int(newton.ShapeFlags.VISIBLE)
        transform = wp.transform(*transforms[shape])
        for vertex in rock.vertices_m:
            world = tuple(wp.transform_point(transform, wp.vec3(*vertex)))
            assert terrain.world_to_local(world)[2] == pytest.approx(vertex[2], abs=1e-6)


def test_sphere_collides_with_rock_above_supporting_bed():
    terrain = SlopeTerrain(15)
    rock = generate_rocks(RockConfig(rock_count=1))[0]
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    add_rigid_bed(builder, terrain, length_m=12)
    add_static_rocks(builder, terrain, (rock,))
    u, v, _ = rock.centre_local_m
    radius = 0.025
    body = builder.add_body(xform=wp.transform(
        wp.vec3(*terrain.local_to_world((u, v, rock.height_m + radius + 0.05))), wp.quat_identity()
    ))
    builder.add_shape_sphere(body, radius=radius)
    model = builder.finalize(device="cpu")
    model.set_gravity(GRAVITY)
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    control = model.control()
    for _ in range(160):
        state.clear_forces()
        solver.step(state, next_state, control, None, 1 / 480)
        state, next_state = next_state, state
        if solver.mj_data.ncon:
            centre_height = terrain.world_to_local(state.body_q.numpy()[body, :3])[2]
            assert centre_height > radius + 0.01
            break
    else:
        pytest.fail("sphere did not hit the rock")


@pytest.mark.parametrize("kwargs", (
    {"roughness": "unknown"}, {"rock_count": 0}, {"rock_count": -1},
    {"width_m": 1}, {"length_m": 2}, {"friction_range": (0.9, 0.3)},
    {"friction_range": (0.1, math.nan)}, {"seed": 1.2},
))
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RockConfig(**kwargs)


def test_impossible_layout_is_rejected_instead_of_silently_overlapping():
    with pytest.raises(ValueError, match="reduce --rock-count"):
        generate_rocks(RockConfig(width_m=2, length_m=4, rock_count=200))
