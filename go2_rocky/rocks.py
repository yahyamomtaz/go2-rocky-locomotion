"""Seeded, fixed convex rocks supported by the local slope plane.

Generation is independent of Newton. Collision construction is kept at the
bottom of this module so the exact same geometry can be inspected in tests.
"""

from dataclasses import dataclass, replace
import math
import random

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from go2_rocky.terrain import Vector3


# Count, footprint radius range, and height above the bed, all in metres.
PRESETS = {
    "low": (120, (0.07, 0.15), (0.015, 0.045)),
    "medium": (180, (0.09, 0.18), (0.03, 0.075)),
    "high": (160, (0.11, 0.21), (0.06, 0.12)),
}
MOUNTAIN_COUNTS = {"low": 800, "medium": 1400, "high": 2000}


@dataclass(frozen=True)
class RockConfig:
    roughness: str = "medium"
    seed: int = 1
    rock_count: int | None = None
    width_m: float = 4.0
    length_m: float = 12.0
    start_clearance_m: float = 2.4
    target_clearance_m: float = 1.0
    friction_range: tuple[float, float] = (0.75, 0.75)
    distribution: str = "trail"

    def __post_init__(self):
        if self.roughness not in PRESETS:
            raise ValueError("roughness must be low, medium, or high")
        if self.distribution not in ("trail", "mountain"):
            raise ValueError("distribution must be trail or mountain")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.rock_count is not None and (
            not isinstance(self.rock_count, int) or self.rock_count <= 0
        ):
            raise ValueError("rock_count must be a positive integer")
        if not math.isfinite(self.width_m) or self.width_m < 2:
            raise ValueError("width_m must be finite and at least 2 m")
        if not math.isfinite(self.length_m) or self.length_m <= 0:
            raise ValueError("length_m must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.start_clearance_m, self.target_clearance_m)):
            raise ValueError("start and target clearances must be finite and nonnegative")
        if self.start_clearance_m + self.target_clearance_m >= self.length_m:
            raise ValueError("start/target clearances leave no rock placement area")
        if len(self.friction_range) != 2 or any(not math.isfinite(f) or f < 0 for f in self.friction_range):
            raise ValueError("friction_range must contain two finite nonnegative values")
        if self.friction_range[0] > self.friction_range[1]:
            raise ValueError("friction_range must be ordered low to high")

    @property
    def count(self):
        default = MOUNTAIN_COUNTS[self.roughness] if self.distribution == "mountain" else PRESETS[self.roughness][0]
        return self.rock_count if self.rock_count is not None else default

    @property
    def bypass_centre_u(self):
        # A 0.7 m unobstructed lane on one side provides a geometric detour.
        # The robot is commanded straight up the rocky centre, not this lane.
        return self.width_m / 2 - 0.5


@dataclass(frozen=True)
class Rock:
    centre_local_m: Vector3
    yaw_rad: float
    bounding_radius_m: float
    height_m: float
    friction: float
    color: Vector3
    vertices_m: tuple[Vector3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    coordinate_frame: str = "trail"
    embedded_fraction: float = 0.0


def generate_rocks(config: RockConfig) -> tuple[Rock, ...]:
    """Reject overlapping footprints and layouts that exceed finite bounds.

    All bottom vertices have local w=0. Conservative bounding circles ensure
    non-overlap even after each rock is rotated about the surface normal.
    """
    if config.distribution != "trail":
        raise ValueError("use generate_mountain_rocks for mountain-wide distribution")
    rng = random.Random(config.seed)
    _, radius_range, height_range = PRESETS[config.roughness]
    lo_v = -config.length_m / 2 + config.start_clearance_m
    hi_v = config.length_m / 2 - config.target_clearance_m
    rocks = []
    for _ in range(max(1000, config.count * 500)):
        radius = rng.uniform(*radius_range)
        if hi_v - lo_v <= 2 * radius:
            continue
        u = rng.uniform(-config.width_m / 2 + radius + 0.05, config.width_m / 2 - radius - 0.05)
        v = rng.uniform(lo_v + radius, hi_v - radius)
        if abs(u - config.bypass_centre_u) < 0.35 + radius:
            continue
        if any(math.hypot(u - r.centre_local_m[0], v - r.centre_local_m[1])
               < radius + r.bounding_radius_m + 0.025 for r in rocks):
            continue

        height = rng.uniform(*height_range)
        vertices = []
        # Uneven rings and an offset summit give an angular, convex stone.
        angles = [2 * math.pi * j / 6 + rng.uniform(-0.15, 0.15) for j in range(6)]
        for top in (False, True):
            for angle in angles:
                distance = radius * rng.uniform(0.5, 0.75) if top else radius * rng.uniform(0.75, 1.0)
                vertices.append((distance * math.cos(angle), distance * math.sin(angle),
                                 height * rng.uniform(0.45, 0.8) if top else 0.0))
        vertices.append((rng.uniform(-0.15, 0.15) * radius, rng.uniform(-0.15, 0.15) * radius, height))
        points = np.asarray(vertices)
        hull = ConvexHull(points)
        triangles = []
        for face, equation in zip(hull.simplices, hull.equations):
            a, b, c = map(int, face)
            # Newton requires outward CCW winding for collision normals.
            if np.dot(np.cross(points[b] - points[a], points[c] - points[a]), equation[:3]) < 0:
                b, c = c, b
            triangles.append((a, b, c))
        shade = rng.uniform(0.34, 0.65)
        rocks.append(Rock(
            centre_local_m=(u, v, 0.0), yaw_rad=rng.uniform(-math.pi, math.pi),
            bounding_radius_m=radius, height_m=height, friction=rng.uniform(*config.friction_range),
            color=(shade, shade * 0.94, shade * 0.85), vertices_m=tuple(vertices), triangles=tuple(triangles),
        ))
        if len(rocks) == config.count:
            return tuple(rocks)
    raise ValueError(f"could only place {len(rocks)}/{config.count} rocks safely; reduce --rock-count")


def generate_mountain_rocks(terrain, config):
    """Small route stones plus larger, varied boulders across the hillside.

    Background boulders avoid the walking route and the initial settling area.
    All objects use real convex collision meshes. Rock count is the combined
    total; the many smaller bedrock formations live in the heightfield itself.
    """
    if config.distribution != "mountain":
        raise ValueError("mountain-wide rocks require distribution='mountain'")
    rng = random.Random(config.seed + 17041)
    foreground_count = min(config.count, {"low": 140, "medium": 200, "high": 240}[config.roughness])
    layout_length = min(24.0, terrain.route_distance_m[-1])
    templates = generate_rocks(RockConfig(seed=config.seed + 843, rock_count=24))
    rocks = []
    # Closely spaced, partly buried blocks replace the old pale, flat tiles.
    # Mountain mode has no empty bypass strip; only the spawn patch is clear.
    for _ in range(max(1000, foreground_count * 500)):
        if len(rocks) == foreground_count:
            break
        radius = rng.uniform(0.12, 0.26)
        u = rng.uniform(-1.5 + radius, 1.5 - radius)
        s = rng.uniform(config.start_clearance_m + radius, layout_length - config.target_clearance_m - radius)
        if any(math.hypot(u-r.centre_local_m[0], s-r.centre_local_m[1])
               < radius + r.bounding_radius_m + 0.012 for r in rocks):
            continue
        template = rng.choice(templates)
        height = radius * rng.uniform(0.65, 1.05)
        vertices = tuple((x * radius / template.bounding_radius_m, y * radius / template.bounding_radius_m,
                          z * height / template.height_m) for x, y, z in template.vertices_m)
        shade = rng.uniform(0.27, 0.43)
        rocks.append(replace(template, centre_local_m=(u, s, 0), yaw_rad=rng.uniform(-math.pi, math.pi),
            bounding_radius_m=radius, height_m=height, vertices_m=vertices,
            friction=rng.uniform(*config.friction_range), color=(shade, shade * 0.99, shade * 0.96),
            coordinate_frame="route", embedded_fraction=rng.uniform(0.5, 0.65)))
    if len(rocks) != foreground_count:
        raise ValueError("could not pack mountain outcrops safely; reduce --rock-count")
    route_tree = cKDTree(terrain.route[:, :2])
    bins = {}
    for _ in range(max(1000, config.count * 150)):
        if len(rocks) == config.count:
            return tuple(rocks)
        radius = rng.uniform(0.65, 1.25) if rng.random() < 0.10 else rng.uniform(0.20, 0.65)
        u = rng.uniform(-config.width_m / 2 + radius + 1, config.width_m / 2 - radius - 1)
        v = rng.uniform(-config.length_m / 2 + radius + 1, config.length_m / 2 - radius - 1)
        if terrain.height_at(u, v) < 0.15 or route_tree.query((u, v))[0] < 0.35 + radius:
            continue
        if np.linalg.norm(np.array((u, v)) - terrain.route[0, :2]) < 1.0 + radius:
            continue
        cell = (math.floor(u / 3), math.floor(v / 3))
        neighbours = [item for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for item in bins.get((cell[0] + dx, cell[1] + dy), ())]
        if any(math.hypot(u - x, v - y) < radius + r + 0.04 for x, y, r in neighbours):
            continue
        template = rng.choice(templates)
        height = radius * rng.uniform(0.35, 0.9)
        vertices = tuple((x * radius / template.bounding_radius_m, y * radius / template.bounding_radius_m,
                          z * height / template.height_m) for x, y, z in template.vertices_m)
        shade = rng.uniform(0.23, 0.47)
        rocks.append(replace(template, centre_local_m=(u, v, 0), yaw_rad=rng.uniform(-math.pi, math.pi),
            bounding_radius_m=radius, height_m=height, vertices_m=vertices,
            friction=rng.uniform(*config.friction_range), color=(shade, shade * 0.96, shade * 0.90),
            coordinate_frame="world_xy", embedded_fraction=rng.uniform(0.25, 0.5)))
        bins.setdefault(cell, []).append((u, v, radius))
    raise ValueError(f"could only place {len(rocks)}/{config.count} mountain rocks; reduce --rock-count")


def add_static_rocks(builder, terrain, rocks):
    """Use the generated convex meshes for BOTH collision and rendering."""
    import newton
    import warp as wp

    rotation = wp.quat_from_matrix(wp.mat33(*(
        value for row in terrain.rotation_matrix for value in row
    )))
    shape_ids = []
    for i, rock in enumerate(rocks):
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = rock.friction
        cfg.has_shape_collision = True
        cfg.is_visible = True
        yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rock.yaw_rad)
        mesh = newton.Mesh(
            vertices=np.asarray(rock.vertices_m),
            indices=np.asarray(rock.triangles, dtype=np.int32).reshape(-1),
            compute_inertia=False,
        )
        shape_ids.append(builder.add_shape_mesh(
            body=-1, mesh=mesh, cfg=cfg,
            xform=wp.transform(wp.vec3(*terrain.local_to_world(rock.centre_local_m)), wp.mul(rotation, yaw)),
            color=rock.color, label=f"fixed_rock_{i:03d}",
        ))
    return shape_ids
