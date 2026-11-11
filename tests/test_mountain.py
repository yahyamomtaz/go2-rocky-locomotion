"""Mountain geometry, local frames, and native heightfield collision checks."""

import math
from dataclasses import replace

import mujoco
import newton
import numpy as np
import pytest
import warp as wp

from go2_rocky.controller import path_command
from go2_rocky.mountain import MountainConfig, MountainTerrain, add_mountain_rocks
from go2_rocky.rocks import RockConfig, generate_rocks


@pytest.fixture(scope="module")
def mountain():
    return MountainTerrain(MountainConfig(resolution_m=0.4))


def test_seeded_mountain_is_reproducible_and_has_relief(mountain):
    other = MountainTerrain(mountain.config)
    np.testing.assert_array_equal(mountain.heights, other.heights)
    np.testing.assert_array_equal(mountain.route, other.route)
    different = MountainTerrain(MountainConfig(seed=2, resolution_m=0.4))
    assert not np.array_equal(mountain.heights, different.heights)
    assert np.ptp(mountain.heights) > 20
    assert mountain.width_m == 80
    assert mountain.route_point(10)[2] > mountain.route[0, 2] + 1


def test_surface_query_matches_asymmetric_grid_nodes(mountain):
    for row, col in ((0, 0), (17, 41), (50, 75), (121, 100), (200, 200)):
        assert mountain.height_at(mountain.axis[col], mountain.axis[row]) == pytest.approx(
            float(mountain.heights[row, col]), abs=1e-5
        )


def test_triangle_height_and_normal_have_known_values():
    terrain = MountainTerrain.__new__(MountainTerrain)
    terrain.n = 2
    terrain.spacing = terrain.width_m = terrain.length_m = 1.0
    terrain.heights = np.array(((0., 1.), (2., 4.)))
    assert terrain.height_at(0.25, -0.25) == pytest.approx(1.5)
    np.testing.assert_allclose(terrain.normal_at(0.25, -0.25), np.array((-1, -3, 1)) / math.sqrt(11))
    assert terrain.height_at(-0.25, 0.25) == pytest.approx(2.0)


def test_route_progress_and_frames_follow_the_surface(mountain):
    for point in mountain.route[::15]:
        frame = mountain.frame_at(point)
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-7)
        assert np.linalg.det(frame) == pytest.approx(1)
        np.testing.assert_allclose(frame[:, 2], mountain.normal_at(*point[:2]))
        assert abs(mountain.path_state(point)[0]) < 1e-5
    assert mountain.progress(mountain.route[0], mountain.route_point(10)) == pytest.approx(10, abs=0.01)
    assert np.linalg.norm(mountain.frame_at(mountain.route[0]) - mountain.frame_at(mountain.route_point(10))) > 0.1


def test_path_controller_uses_current_route_tangent(mountain):
    point = mountain.route_point(7)
    frame = mountain.frame_at(point)
    rotation = np.column_stack((frame[:, 1], -frame[:, 0], frame[:, 2]))
    q = np.array([*point, *wp.quat_from_matrix(wp.mat33(*rotation.ravel())), *([0.0] * 12)])
    np.testing.assert_allclose(path_command(mountain, q, np.zeros(18), 0.6), (0.6, 0, 0), atol=1e-5)


def test_rocks_are_placed_on_local_mountain_surface(mountain):
    config = RockConfig(rock_count=6, length_m=24)
    rocks = generate_rocks(config)
    builder = newton.ModelBuilder()
    ids, transforms = add_mountain_rocks(builder, mountain, rocks, config.length_m)
    assert len(ids) == 6
    for rock, transform in zip(rocks, transforms):
        pose = wp.transform(*transform)
        vertices = np.asarray([tuple(wp.transform_point(pose, wp.vec3(*v))) for v in rock.vertices_m])
        assert np.min(vertices[:, 2] - mountain.height_at(vertices[:, 0], vertices[:, 1])) > -0.003


def test_embedded_outcrop_has_buried_base_and_exposed_top():
    terrain = MountainTerrain.__new__(MountainTerrain)
    terrain.n = 2
    terrain.spacing = terrain.width_m = terrain.length_m = 80.0
    terrain.heights = np.zeros((2, 2))
    terrain.route = np.array(((0., -20., 0.), (0., 20., 0.)))
    terrain._cache_route()
    source = generate_rocks(RockConfig(rock_count=1))[0]
    rock = replace(source, centre_local_m=(0, 0, 0), coordinate_frame="world_xy",
                   height_m=0.2, embedded_fraction=0.55,
                   vertices_m=tuple((x, y, z * 0.2 / source.height_m) for x, y, z in source.vertices_m))
    builder = newton.ModelBuilder()
    ids, transforms = add_mountain_rocks(builder, terrain, (rock,), 24)
    vertices = np.asarray([tuple(wp.transform_point(wp.transform(*transforms[0]), wp.vec3(*v)))
                           for v in rock.vertices_m])
    assert builder.shape_body[ids[0]] == -1
    np.testing.assert_allclose(vertices[:6, 2], -0.11, atol=1e-6)
    assert vertices[:, 2].max() == pytest.approx(0.09, abs=1e-6)


def test_native_mujoco_heightfield_support_and_ray_heights(mountain):
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.ke = 5e4
    builder.default_shape_cfg.kd = 500
    builder.rigid_gap = 0.0
    shape = mountain.add_collision(builder)
    bodies = []
    for distance in (0, 4, 8):
        p = mountain.route_point(distance)
        body = builder.add_body(xform=wp.transform(wp.vec3(p[0], p[1], p[2] + 0.4), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        bodies.append(body)
    model = builder.finalize(device="cpu")
    model.set_gravity((0, 0, -9.81))
    assert model.shape_type.numpy()[shape] == int(newton.GeoType.HFIELD)
    assert model.shape_body.numpy()[shape] == -1
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True)
    for x, y in ((-12.15, -14.25), (0.13, -10.27), (7.13, 3.17), (-8.23, 10.17)):
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(solver.mj_model, solver.mj_data,
            np.array((x, y, 80.)), np.array((0., 0., -1.)), None, True, -1, geom_id)
        assert distance > 0
        assert 80 - distance == pytest.approx(mountain.height_at(x, y), abs=3e-5)
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    saw_contact = False
    for _ in range(480):
        state.clear_forces()
        solver.step(state, next_state, control, None, 1 / 480)
        state, next_state = next_state, state
        saw_contact |= solver.mj_data.ncon > 0
    assert saw_contact
    for body in bodies:
        position = state.body_q.numpy()[body, :3]
        assert np.isfinite(position).all()
        clearance = mountain.path_state(position)[2] - 0.1
        assert abs(clearance) < 0.03


@pytest.mark.parametrize("kwargs", (
    {"size_m": 10}, {"height_m": -1}, {"resolution_m": 1},
    {"resolution_m": 0.001}, {"trail_grade_deg": 0}, {"trail_grade_deg": math.nan},
))
def test_invalid_mountain_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MountainConfig(**kwargs)
