"""Geometric camera clearance and simulation-time video sampling."""

from types import SimpleNamespace

import numpy as np
import pytest

from go2_rocky.camera import FollowCamera, set_view
from go2_rocky.terrain import SlopeTerrain
from go2_rocky.video import VideoRecorder


def test_follow_camera_aims_at_the_robot_and_clears_ground():
    terrain = SlopeTerrain(25)
    base = np.asarray(terrain.local_to_world((0, 0, 0.34)))
    follow = FollowCamera(terrain)
    position, pitch, yaw = follow.update(base, 0)
    np.testing.assert_allclose(follow.target, base+(0, 0, 0.04))
    assert np.linalg.norm(position[:2]-base[:2]) == pytest.approx(3.0)
    assert pitch < 0
    radians = np.radians((pitch, yaw))
    view = np.array((np.cos(radians[0])*np.cos(radians[1]),
                     np.cos(radians[0])*np.sin(radians[1]), np.sin(radians[0])))
    np.testing.assert_allclose(view, (follow.target-position)/np.linalg.norm(follow.target-position), atol=1e-7)
    points = np.linspace(follow.target, position, 100)
    assert np.min(points[:, 2]-terrain.height_at(points[:, 0], points[:, 1])) > 0.1


def test_rock_occlusion_is_cleared_even_for_a_thin_rock():
    follow = FollowCamera(SlopeTerrain())
    target, camera = np.array((0., 0., 0.4)), np.array((0., -2., 1.8))
    rock = np.array([[[-0.3, -0.52, 0.], [0.3, -0.50, 1.4]]])
    assert follow._clear_height(camera, target, rock) == pytest.approx(4.72)


def test_free_camera_is_not_overwritten_after_initialization():
    calls = []
    viewer = SimpleNamespace(set_camera=lambda **kw: calls.append(kw))
    sim = SimpleNamespace(state=SimpleNamespace(joint_q=SimpleNamespace(
        numpy=lambda: np.array([0., 0., 0.34]))))
    follow = FollowCamera(SlopeTerrain())
    set_view(viewer, sim, "free", follow, 0, initial=True)
    set_view(viewer, sim, "free", follow, 1)
    assert len(calls) == 1


def test_video_clock_preserves_duration_and_does_not_record_pauses():
    # Exercise the clock with a fake sink; no encoder/GPU dependency needed.
    recorder = VideoRecorder.__new__(VideoRecorder)
    samples = []
    recorder.writer = SimpleNamespace(append_data=lambda frame: samples.append(frame))
    recorder.fps, recorder.frames = 30, 0
    for time in np.arange(0, 1.001, 0.02):
        recorder.capture(float(time), lambda t=time: float(t))
        recorder.capture(float(time), lambda: pytest.fail("paused frames must not be read twice"))
    assert recorder.frames == 31
    np.testing.assert_allclose(samples[0], 0)
    assert samples[-1] == pytest.approx(1)
    assert max(abs(np.asarray(samples)-np.arange(31)/30)) <= 0.02
