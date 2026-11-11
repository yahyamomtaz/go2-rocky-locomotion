"""Slope geometry, independent of the physics engine.

World +Z points up and the robot travels along +Y. The rotation maps terrain
coordinates (u across, v along, w normal) into world coordinates. Gravity is
never part of this transform: it stays vertical in the simulation.
"""

from dataclasses import dataclass
import math
from typing import Literal


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Direction = Literal["uphill", "downhill", "cross"]


@dataclass(frozen=True)
class SlopeTerrain:
    """An ideal slope surface passing through world point (0, 0, z0).

    This describes geometry only; it does not yet create a collision shape.
    Use a nonnegative slope angle and choose its sign through ``direction``.
    """

    slope_deg: float = 0.0
    direction: Direction = "uphill"
    z0: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.slope_deg) or not 0 <= self.slope_deg < 90:
            raise ValueError("slope_deg must be finite and in [0, 90)")
        if self.direction not in ("uphill", "downhill", "cross"):
            raise ValueError("direction must be uphill, downhill, or cross")
        if not math.isfinite(self.z0):
            raise ValueError("z0 must be finite")

    @property
    def rotation_matrix(self) -> Matrix3:
        """Return the terrain-to-world rotation, stored as rows."""
        theta = math.radians(self.slope_deg)
        c, s = math.cos(theta), math.sin(theta)
        if self.direction == "uphill":
            # Rotate about +X: local +Y gains a positive world-Z component.
            return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
        if self.direction == "downhill":
            # Rotate about -X: local +Y loses height.
            return ((1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c))
        # Rotate about -Y: local +X gains height; local +Y stays horizontal.
        return ((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c))

    def local_to_world(self, point: Vector3) -> Vector3:
        """Rotate a terrain point and translate the surface to height z0."""
        r = self.rotation_matrix
        x, y, z = (sum(row[i] * point[i] for i in range(3)) for row in r)
        return (x, y, z + self.z0)

    def world_to_local(self, point: Vector3) -> Vector3:
        """Undo translation, then use R transpose as the inverse rotation."""
        relative = (point[0], point[1], point[2] - self.z0)
        r = self.rotation_matrix
        u, v, w = (
            sum(r[j][i] * relative[j] for j in range(3)) for i in range(3)
        )
        return (u, v, w)

    @property
    def normal(self) -> Vector3:
        """Upward unit surface normal: R times local (0, 0, 1)."""
        r = self.rotation_matrix
        return (r[0][2], r[1][2], r[2][2])

    @property
    def path_tangent(self) -> Vector3:
        """Unit direction of travel: R times local (0, 1, 0)."""
        r = self.rotation_matrix
        return (r[0][1], r[1][1], r[2][1])

    def height_at(self, x: float, y: float) -> float:
        """World-Z height of the ideal surface at world position (x, y).

        Later, collision shapes will use the same transform. This analytic
        surface extends infinitely; finite bed boundaries are a separate step.
        """
        gradient = math.tan(math.radians(self.slope_deg))
        if self.direction == "uphill":
            return self.z0 + y * gradient
        if self.direction == "downhill":
            return self.z0 - y * gradient
        return self.z0 + x * gradient

    def normal_at(self, x, y):
        return self.normal

    def frame_at(self, position):
        return self.rotation_matrix

    def path_state(self, position):
        return self.world_to_local(position)

    def progress(self, start, end):
        return sum((end[i] - start[i]) * self.path_tangent[i] for i in range(3))
