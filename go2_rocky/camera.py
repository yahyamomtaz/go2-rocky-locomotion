"""Robot-centred chase view with a stable heading and terrain clearance."""

import math

import numpy as np
from scipy.spatial.transform import Rotation


def look_angles(position, target):
    delta = np.asarray(target)-position
    return (math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))),
            math.degrees(math.atan2(delta[1], delta[0])))


class FollowCamera:
    DEFAULT_DISTANCE = 3.0
    DEFAULT_HEIGHT = 1.8

    def __init__(self, terrain, bounds=None, distance=DEFAULT_DISTANCE, height=DEFAULT_HEIGHT):
        if any(not math.isfinite(v) or v <= 0 for v in (distance, height)):
            raise ValueError("camera distance and height must be finite and positive")
        self.terrain = terrain
        self.bounds = np.empty((0, 2, 3)) if bounds is None else np.asarray(bounds)
        self.distance, self.height = distance, height
        self.position = self.target = self.time = None
        self.angle = 15.0

    @classmethod
    def from_simulation(cls, sim, *, distance=DEFAULT_DISTANCE, height=DEFAULT_HEIGHT):
        transforms = sim.model.shape_transform.numpy()[sim.rock_shape_ids]
        if len(transforms):
            vertices = np.asarray([rock.vertices_m for rock in sim.rocks])
            rotations = Rotation.from_quat(transforms[:, 3:]).as_matrix()
            world = np.einsum("nij,nvj->nvi", rotations, vertices)+transforms[:, None, :3]
            bounds = np.stack((world.min(axis=1), world.max(axis=1)), axis=1)
        else:
            bounds = None
        return cls(sim.terrain, bounds, distance=distance, height=height)

    def _ground_height(self, xy):
        try:
            return self.terrain.height_at(xy[:, 0], xy[:, 1])
        except (TypeError, ValueError):
            return np.array([self.terrain.height_at(x, y) for x, y in xy])

    def _clear_height(self, position, target, nearby):
        # Line-of-sight samples include the camera endpoint, avoiding a camera
        # inside the hillside or a boulder. Skip the robot's immediate vicinity.
        fraction = np.linspace(0.12, 1, 40)
        xy = target[None, :2]+fraction[:, None]*(position[:2]-target[:2])
        top = self._ground_height(xy)+0.12
        height = max(position[2], float(np.max(target[2]+(top-target[2])/fraction)))
        if len(nearby):
            # Exact XY slab intersections catch even narrow rocks between the
            # heightfield samples. Bounds are conservative around convex rocks.
            delta = position[:2]-target[:2]
            parallel = np.abs(delta) < 1e-10
            safe_delta = np.where(parallel, 1, delta)
            a = (nearby[:, 0, :2]-target[:2])/safe_delta
            b = (nearby[:, 1, :2]-target[:2])/safe_delta
            low, high = np.minimum(a, b), np.maximum(a, b)
            low[:, parallel], high[:, parallel] = -np.inf, np.inf
            outside = (((target[:2] < nearby[:, 0, :2]) | (target[:2] > nearby[:, 1, :2])) & parallel).any(-1)
            entry = np.maximum(low.max(-1), 0.12)
            exit = np.minimum(high.min(-1), 1)
            hit = (entry <= exit) & ~outside
            if hit.any():
                height = max(height, float(np.max(target[2]+(nearby[hit, 1, 2]+0.08-target[2])/entry[hit])))
        return height

    def update(self, base, time):
        base = np.asarray(base)
        if hasattr(self.terrain, "route_point"):
            progress = self.terrain.path_state(base)[1]
            heading = self.terrain.route_point(progress+0.8)-self.terrain.route_point(progress-0.8)
        else:
            heading = np.asarray(self.terrain.path_tangent).copy()
        heading[2] = 0
        heading /= max(np.linalg.norm(heading), 1e-9)
        across = np.array((heading[1], -heading[0], 0))
        target = base+np.array((0, 0, 0.04))
        if len(self.bounds):
            centre = self.bounds.mean(axis=1)
            nearby = self.bounds[np.linalg.norm(centre[:, :2]-base[:2], axis=1) < self.distance+3]
        else:
            nearby = self.bounds
        choices = []
        for degrees in (self.angle, 15., -15., 0., 40., -40., 70., -70.):
            angle = math.radians(degrees)
            offset = self.distance*(-math.cos(angle)*heading+math.sin(angle)*across)
            position = target+offset+np.array((0, 0, self.height))
            position[2] = self._clear_height(position, target, nearby)
            choices.append((position[2]+0.002*abs(degrees-15), degrees, position))
        best = min(choices, key=lambda item: item[0])
        if choices[0][0] <= best[0]+0.12:
            best = choices[0]  # avoid switching sides for tiny height differences
        _, self.angle, desired = best
        if self.time is None:
            self.position, self.target = desired, target
        else:
            alpha = 1-math.exp(-max(0, time-self.time)/0.16)
            self.position += alpha*(desired-self.position)
            self.target += alpha*(target-self.target)
        self.position[2] = self._clear_height(self.position, self.target, nearby)
        self.time = time
        return self.position.copy(), *look_angles(self.position, self.target)


def set_view(viewer, sim, mode, follow, time, *, initial=False):
    """Free mode sets a useful initial view, then preserves mouse adjustments."""
    import warp as wp
    if mode == "wide":
        if initial:
            if sim.is_mountain:
                peak = float(sim.terrain.heights.max())
                target = np.array((0, 0, peak*0.3))
                position = np.array((sim.width_m*0.7, -sim.length_m*0.85, peak*1.7))
            else:
                target = np.asarray(sim.terrain.local_to_world((0, 0, 0.3)))
                position = target+np.array((8, -10, 7))
            pitch, yaw = look_angles(position, target)
            viewer.set_camera(pos=wp.vec3(*position), pitch=pitch, yaw=yaw)
    elif mode == "follow" or initial:
        position, pitch, yaw = follow.update(sim.state.joint_q.numpy()[:3], time)
        viewer.set_camera(pos=wp.vec3(*position), pitch=pitch, yaw=yaw)
