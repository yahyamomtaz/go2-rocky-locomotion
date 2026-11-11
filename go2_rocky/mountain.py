"""Seeded mountain heightfields and a surface-following uphill trail.

OpenSimplex supplies several scales of ridged noise. The native heightfield
is used for collision; there is no inclined support box in mountain mode.
"""

from dataclasses import dataclass
import math

import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
from go2_rocky.surface import RockSurface, SURFACE_AMPLITUDES


@dataclass(frozen=True)
class MountainConfig:
    seed: int = 1
    size_m: float = 80.0
    height_m: float = 26.0
    resolution_m: float = 0.2
    trail_grade_deg: float = 15.0
    trail_length_m: float = 40.0
    trail_half_width_m: float = 1.0
    surface_rockiness: float = 1.0
    roughness: str = "medium"

    def __post_init__(self):
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("size_m", "height_m", "resolution_m", "trail_length_m", "trail_half_width_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.size_m < 40:
            raise ValueError("mountain size must be at least 40 m")
        if not math.isfinite(self.trail_grade_deg) or not 0 < self.trail_grade_deg <= 25:
            raise ValueError("trail_grade_deg must be in (0, 25]")
        if self.resolution_m > 0.5 or math.ceil(self.size_m / self.resolution_m) > 1200:
            raise ValueError("terrain resolution must be <=0.5 m and use at most 1201 grid points per axis")
        if not math.isfinite(self.surface_rockiness) or not 0 <= self.surface_rockiness <= 3:
            raise ValueError("surface_rockiness must be in [0, 3]")
        if self.roughness not in SURFACE_AMPLITUDES:
            raise ValueError("roughness must be low, medium, or high")


class MountainTerrain:
    direction = "uphill"

    def __init__(self, config=MountainConfig()):
        self.config = config
        self.slope_deg = config.trail_grade_deg  # Design target, not a constant surface slope.
        self.width_m = self.length_m = config.size_m
        self.n = math.ceil(config.size_m / config.resolution_m) + 1
        self.axis = np.linspace(-config.size_m / 2, config.size_m / 2, self.n)
        self.spacing = config.size_m / (self.n - 1)
        x, y = np.meshgrid(self.axis, self.axis)
        # Broad asymmetric peaks and ridges, rather than a tilted plane.
        scale = config.size_m / 80
        main = np.exp(-((x / scale - 4) / 15)**2 - ((y / scale - 13) / 18)**2)
        west = np.exp(-((x / scale + 17) / 11)**2 - ((y / scale - 4) / 14)**2)
        east = np.exp(-((x / scale - 20) / 10)**2 - ((y / scale - 19) / 12)**2)
        envelope = np.maximum.reduce((main, west * 0.7, east * 0.6))
        noise = OpenSimplex(config.seed)
        ridges = np.zeros_like(x)
        for frequency, amplitude in ((0.08, 3.0), (0.17, 1.2), (0.36, 0.4)):
            layer = noise.noise2array(self.axis * frequency / scale, self.axis * frequency / scale)
            ridges += amplitude * (1 - np.abs(layer))
        edge = np.clip((config.size_m / 2 - np.abs(x)) / (5 * scale), 0, 1)
        edge *= np.clip((config.size_m / 2 - np.abs(y)) / (5 * scale), 0, 1)
        heights = config.height_m * (main + 0.48 * west + 0.38 * east)
        heights += envelope * ridges * config.height_m / 26
        self.heights = gaussian_filter(heights * edge, sigma=1.2).astype(np.float32)

        self._make_route()
        # A narrow walking trail is blended into the existing hillside. Its
        # elevations follow the mountain; we do not cut a linear ramp to a peak.
        grid_xy = np.column_stack((x.ravel(), y.ravel()))
        _, nearest = cKDTree(self.route[:, :2]).query(grid_xy)
        smooth_z = gaussian_filter1d(self.route[:, 2], sigma=3)
        distance_sq = np.full(len(grid_xy), np.inf)
        trail_z = np.zeros(len(grid_xy))
        # Project to adjacent segments so trail elevations interpolate smoothly
        # instead of making a staircase from nearest-point samples.
        for offset in (-1, 0):
            index = np.clip(nearest + offset, 0, len(self.route) - 2)
            delta = self.route[index + 1, :2] - self.route[index, :2]
            fraction = np.clip(np.sum((grid_xy - self.route[index, :2]) * delta, axis=1)
                               / np.sum(delta**2, axis=1), 0, 1)
            projected = self.route[index, :2] + fraction[:, None] * delta
            candidate = np.sum((grid_xy - projected)**2, axis=1)
            better = candidate < distance_sq
            distance_sq[better] = candidate[better]
            heights_on_segment = smooth_z[index] + fraction * (smooth_z[index + 1] - smooth_z[index])
            trail_z[better] = heights_on_segment[better]
        blend = np.exp(-(np.sqrt(distance_sq) / config.trail_half_width_m)**4)
        flattened = self.heights.ravel()
        flattened[:] = flattened * (1 - blend) + trail_z * blend
        # A small clear, level settling patch at the trailhead.
        start = self.route[0]
        radius = np.hypot(x - start[0], y - start[1])
        pad = np.clip((1.2 - radius) / 0.6, 0, 1)
        pad = pad * pad * (3 - 2 * pad)
        self.heights[:] = self.heights * (1 - pad) + start[2] * pad
        self.surface = RockSurface(config.seed, config.size_m, config.roughness)
        self.surface_relief_m = (
            self.surface.relief(x, y, np.sqrt(distance_sq).reshape(x.shape))
            * config.surface_rockiness * (1 - pad) * edge
        ).astype(np.float32)
        self.heights += self.surface_relief_m
        self.route[:, 2] = self.height_at(self.route[:, 0], self.route[:, 1])
        self._cache_route()

    def _query(self, x, y):
        """Height and derivatives on the same diagonal used by Newton's mesh."""
        col = np.clip((np.asarray(x) + self.width_m / 2) / self.spacing, 0, self.n - 1)
        row = np.clip((np.asarray(y) + self.length_m / 2) / self.spacing, 0, self.n - 1)
        i = np.minimum(np.floor(col).astype(int), self.n - 2)
        j = np.minimum(np.floor(row).astype(int), self.n - 2)
        fx, fy = col - i, row - j
        h00, h10 = self.heights[j, i], self.heights[j, i + 1]
        h01, h11 = self.heights[j + 1, i], self.heights[j + 1, i + 1]
        lower = fx >= fy
        dx = np.where(lower, h10 - h00, h11 - h01) / self.spacing
        dy = np.where(lower, h11 - h10, h01 - h00) / self.spacing
        height = np.where(lower, h00 + fx * (h10 - h00) + fy * (h11 - h10),
                          h00 + fx * (h11 - h01) + fy * (h01 - h00))
        return height, dx, dy

    def height_at(self, x, y):
        result = self._query(x, y)[0]
        return float(result) if np.ndim(result) == 0 else result

    def normal_at(self, x, y):
        _, dx, dy = self._query(x, y)
        normal = np.array((-float(dx), -float(dy), 1.0))
        return normal / np.linalg.norm(normal)

    def _make_route(self):
        scale = self.config.size_m / 80
        point = np.array((-5.0, -21.0)) * scale
        previous = np.array((1.0, 1.0)) / math.sqrt(2)
        points = []
        side = 1
        for _ in range(math.ceil(self.config.trail_length_m / 0.25) + 1):
            points.append((*point, self.height_at(*point)))
            _, gx, gy = self._query(*point)
            gradient = np.array((float(gx), float(gy)))
            norm = float(np.linalg.norm(gradient))
            up = gradient / max(norm, 1e-8)
            contour = np.array((up[1], -up[0]))
            if point[0] > 15 * scale:
                side = -1
            elif point[0] < -15 * scale:
                side = 1
            ratio = min(1.0, math.tan(math.radians(self.config.trail_grade_deg)) / max(norm, 1e-8))
            desired = ratio * up + side * math.sqrt(max(0, 1 - ratio**2)) * contour
            # Smooth turns at switchbacks; the resulting actual grade varies.
            turn = math.atan2(previous[0] * desired[1] - previous[1] * desired[0], np.dot(previous, desired))
            angle = math.atan2(previous[1], previous[0]) + float(np.clip(turn, -0.08, 0.08))
            previous = np.array((math.cos(angle), math.sin(angle)))
            point = point + previous * 0.25
            if np.max(np.abs(point)) > self.config.size_m / 2 - 3:
                break
        self.route = np.asarray(points, dtype=float)
        if len(points) < 45:
            raise ValueError("generated trail is too short; choose a larger mountain")

    def _cache_route(self):
        self._segments = np.diff(self.route, axis=0)
        self._xy_length_sq = np.sum(self._segments[:, :2]**2, axis=1)
        self.route_distance_m = np.r_[0, np.cumsum(np.linalg.norm(self._segments, axis=1))]

    def _nearest(self, position):
        difference = np.asarray(position[:2]) - self.route[:-1, :2]
        fraction = np.clip(np.sum(difference * self._segments[:, :2], axis=1) / self._xy_length_sq, 0, 1)
        projected = self.route[:-1, :2] + fraction[:, None] * self._segments[:, :2]
        index = int(np.argmin(np.sum((projected - position[:2])**2, axis=1)))
        return index, float(fraction[index])

    def frame_at(self, position):
        index, _ = self._nearest(position)
        normal = self.normal_at(*position[:2])
        tangent = self._segments[index].copy()
        tangent -= normal * np.dot(tangent, normal)
        tangent /= np.linalg.norm(tangent)
        across = np.cross(tangent, normal)
        return np.column_stack((across, tangent, normal))

    def path_state(self, position):
        index, fraction = self._nearest(position)
        closest = self.route[index] + fraction * self._segments[index]
        frame = self.frame_at(position)
        progress = self.route_distance_m[index] + fraction * np.linalg.norm(self._segments[index])
        cross_track = float(np.dot(np.asarray(position) - closest, frame[:, 0]))
        clearance = (position[2] - self.height_at(*position[:2])) * frame[2, 2]
        return (cross_track, float(progress), float(clearance))

    def progress(self, start, end):
        return self.path_state(end)[1] - self.path_state(start)[1]

    def route_point(self, distance):
        s = float(np.clip(distance, 0, self.route_distance_m[-1]))
        return np.array([np.interp(s, self.route_distance_m, self.route[:, i]) for i in range(3)])

    def add_collision(self, builder, *, textured=False):
        import newton
        field = newton.Heightfield(data=self.heights, nrow=self.n, ncol=self.n,
                                  hx=self.width_m / 2, hy=self.length_m / 2)
        cfg = builder.default_shape_cfg.copy()
        cfg.is_visible = not textured
        return builder.add_shape_heightfield(heightfield=field, cfg=cfg,
                                            color=(0.42, 0.38, 0.31), label="mountain_surface")

    def visual_arrays(self):
        """Flat-shaded triangles with exactly the collision grid's vertices."""
        x, y = np.meshgrid(self.axis, self.axis)
        vertices = np.column_stack((x.ravel(), y.ravel(), self.heights.ravel())).astype(np.float32)
        ids = np.arange(self.n * self.n).reshape(self.n, self.n)
        a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
        faces = np.concatenate((np.column_stack((a, b, d)), np.column_stack((a, d, c))))
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        points = triangles.reshape(-1, 3)
        uvs = (points[:, :2] + self.width_m / 2) / self.width_m
        return points, np.arange(len(points), dtype=np.int32), np.repeat(normals, 3, axis=0), uvs

    def log_visual(self, viewer, device):
        import warp as wp
        points, indices, normals, uvs = self.visual_arrays()
        texture = self.surface.material() if self.config.surface_rockiness else None
        viewer.log_mesh("mountain_surface_prototype",
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(indices, dtype=wp.int32, device=device),
            normals=wp.array(normals, dtype=wp.vec3, device=device),
            uvs=wp.array(uvs, dtype=wp.vec2, device=device), texture=texture, hidden=True,
            color=(1.0, 1.0, 1.0) if texture is not None else (0.42, 0.38, 0.31),
            roughness=1.0, metallic=0.0)
        # The installed viewer's direct log_mesh path leaves its texture-enable
        # material channel at zero. Use the public instancing API to set it.
        viewer.log_instances("mountain_material_surface", mesh="mountain_surface_prototype",
            xforms=wp.array([wp.transform(wp.vec3(0, 0, 0), wp.quat_identity())], dtype=wp.transform, device=device),
            scales=wp.array([(1, 1, 1)], dtype=wp.vec3, device=device),
            colors=wp.array([(1, 1, 1) if texture is not None else (0.42, 0.38, 0.31)], dtype=wp.vec3, device=device),
            materials=wp.array([(1, 0, 0, int(texture is not None))], dtype=wp.vec4, device=device))


def add_mountain_rocks(builder, mountain, rocks, layout_length):
    """Place fixed convex stones in local frames along the curved trail."""
    import newton
    import warp as wp

    shape_ids, placements = [], []
    for i, rock in enumerate(rocks):
        u, v, _ = rock.centre_local_m
        if rock.coordinate_frame == "world_xy":
            origin = np.array((u, v, 0.0))
        else:
            origin = mountain.route_point(v if rock.coordinate_frame == "route" else v + layout_length / 2)
            frame = mountain.frame_at(origin)
            origin += frame[:, 0] * u
        origin[2] = mountain.height_at(*origin[:2])
        frame = mountain.frame_at(origin)
        c, s = math.cos(rock.yaw_rad), math.sin(rock.yaw_rad)
        rotation = frame @ np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        vertices = np.asarray(rock.vertices_m)
        world = vertices @ rotation.T + origin
        # Support the flat bottom on the curved surface. Include edge samples,
        # not only the corners, when measuring how far it must be lifted.
        bottom = world[:6]
        samples = np.concatenate([world, (bottom + np.roll(bottom, -1, axis=0)) / 2, origin[None, :]])
        if rock.embedded_fraction:
            # Fixed geological outcrops are fused with the heightfield. Their
            # buried bases must not be confused with dynamic-body penetration.
            bottom_residual = mountain.height_at(bottom[:, 0], bottom[:, 1]) - bottom[:, 2]
            shift = float(np.median(bottom_residual) - rock.embedded_fraction * rock.height_m * frame[2, 2])
            exposed = world[:, 2] + shift - mountain.height_at(world[:, 0], world[:, 1])
            shift += max(0.0, 0.025 - float(exposed.max()))
            origin[2] += shift
        else:
            lift = float(max(0, np.max(mountain.height_at(samples[:, 0], samples[:, 1]) - samples[:, 2])))
            origin[2] += lift
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = rock.friction
        mesh = newton.Mesh(vertices=vertices, indices=np.asarray(rock.triangles).reshape(-1), compute_inertia=False)
        quaternion = wp.quat_from_matrix(wp.mat33(*rotation.ravel()))
        shape_ids.append(builder.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg,
            xform=wp.transform(wp.vec3(*origin), quaternion), color=rock.color, label=f"mountain_rock_{i:03d}"))
        placements.append((*map(float, origin), *map(float, quaternion)))
    return shape_ids, np.asarray(placements).reshape(-1, 7)
