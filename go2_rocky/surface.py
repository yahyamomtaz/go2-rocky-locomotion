"""Deterministic fractured bedrock relief and matching procedural materials.

The rock formations are baked into the collision heightfield. The material
adds mineral/color variation; it does not stand in for physical obstacles.
"""

import numpy as np
from scipy.spatial import cKDTree


SURFACE_AMPLITUDES = {
    # Small-block and larger-outcrop relief, away from and on the trail.
    "low": (0.16, 0.35, 0.014, 0.012),
    "medium": (0.30, 0.65, 0.026, 0.022),
    "high": (0.48, 1.00, 0.050, 0.045),
}


class RockSurface:
    def __init__(self, seed, size_m, roughness):
        self.size_m = size_m
        self.roughness = roughness
        rng = np.random.default_rng((seed + 934817) % (2**64))
        self.layers = []
        for spacing in (0.75, 2.8):
            axis = np.arange(-size_m / 2 - 2 * spacing, size_m / 2 + 2 * spacing, spacing)
            x, y = np.meshgrid(axis, axis)
            centres = np.column_stack((x.ravel(), y.ravel()))
            centres += rng.uniform(-0.32, 0.32, centres.shape) * spacing
            amplitudes = rng.uniform(0.55, 1.25, len(centres))
            colors = rng.uniform(0, 1, len(centres))
            self.layers.append((cKDTree(centres), amplitudes, colors))
        centres = self.layers[0][0].data
        self.cell_count = int(np.count_nonzero(np.max(np.abs(centres), axis=1) < size_m / 2))

    def sample(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        xy = np.column_stack((x.ravel(), y.ravel()))
        values = []
        for tree, amplitudes, colors in self.layers:
            distances, ids = tree.query(xy, k=2)
            fracture = np.clip((distances[:, 1] - distances[:, 0]) / (distances[:, 1] + 1e-8), 0, 1)
            # Plateaus surrounded by angular faces and narrow joints.
            block = np.minimum(1, fracture * 2.2)**0.7
            values.append((block * amplitudes[ids[:, 0]], fracture, colors[ids[:, 0]]))
        return tuple(tuple(a.reshape(x.shape) for a in layer) for layer in values)

    def relief(self, x, y, trail_distance):
        small, large = self.sample(x, y)
        a, b, trail_a, trail_b = SURFACE_AMPLITUDES[self.roughness]
        # The trail is still rocky, with smaller formations under the feet.
        trail = np.exp(-(np.asarray(trail_distance) / 1.1)**4)
        return small[0] * (a * (1 - trail) + trail_a * trail) + large[0] * (b * (1 - trail) + trail_b * trail)

    def material(self, resolution=1024):
        axis = np.linspace(-self.size_m / 2, self.size_m / 2, resolution)
        x, y = np.meshgrid(axis, axis)
        small, large = self.sample(x, y)
        # Slate/granite, warmer weathered rock, and occasional lighter minerals.
        mineral = small[2]
        color = np.empty((*x.shape, 3), dtype=float)
        color[..., 0] = 0.26 + 0.16 * mineral + 0.025 * large[2]
        color[..., 1] = 0.26 + 0.14 * mineral
        color[..., 2] = 0.25 + 0.12 * mineral
        joints = 0.58 + 0.42 * np.clip(small[1] / 0.12, 0, 1)
        grain_rng = np.random.default_rng(1183)
        grain = grain_rng.normal(0, 0.016, x.shape)
        color *= joints[..., None]
        color += grain[..., None]
        # Newton uploads conventional top-left-origin images with a vertical
        # flip. Our sampling grid starts at world -Y, so return image +Y first.
        return np.ascontiguousarray(np.flipud(np.uint8(np.clip(color, 0, 1) * 255)))
