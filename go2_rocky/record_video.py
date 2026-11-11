"""Render saved trajectories to MP4 without rerunning physics.

Use --log for one clip, or --before/--after for a labelled comparison. The
comparison retains separate clips and holds each final frame until both end.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import newton
import newton.viewer
import numpy as np
import warp as wp

from go2_rocky.analyze_trajectory import recorded_surface
from go2_rocky.camera import FollowCamera, set_view
from go2_rocky.mountain import MountainConfig
from go2_rocky.rocks import Rock, RockConfig
from go2_rocky.run_go2 import Go2Run
from go2_rocky.surface import RockSurface
from go2_rocky.terrain import SlopeTerrain
from go2_rocky.video import VideoRecorder, caption


def load_log(path):
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    return data, json.loads(data["metadata_json"].item())


def validate_pair(before, after):
    a, ma = load_log(before)
    b, mb = load_log(after)
    for key in ("speed_m_s", "settle_s", "walking_seconds_requested", "goal_distance_m",
                "slope_deg", "direction", "controller", "controller_parameters", "friction",
                "mountain_config", "rock_config", "rocks"):
        if ma.get(key) != mb.get(key):
            raise ValueError(f"before/after settings differ: {key}")
    np.testing.assert_array_equal(a["joint_q"][0], b["joint_q"][0])
    for key in ("mountain_elevation_m", "rock_transform_world"):
        if key in a or key in b:
            np.testing.assert_array_equal(a[key], b[key])


def render_log(args):
    data, meta = load_log(args.log)
    if "mountain_elevation_m" in data:
        terrain = recorded_surface(data)
        terrain.config = MountainConfig(**meta["mountain_config"])
        terrain.slope_deg = terrain.config.trail_grade_deg
        terrain.surface = RockSurface(terrain.config.seed, terrain.config.size_m, terrain.config.roughness)
        terrain.surface_relief_m = data.get("surface_relief_m", np.zeros_like(terrain.heights))
    else:
        terrain = SlopeTerrain(meta["slope_deg"], meta["direction"])
    rock_config = RockConfig(**meta["rock_config"]) if meta.get("rock_config") else None
    saved_time = data["time_s"]
    if len(saved_time) < 1 or not np.isfinite(saved_time).all() or (np.diff(saved_time) <= 0).any():
        raise ValueError("trajectory must have finite, increasing timestamps")
    end = float(saved_time[-1])
    count = int(math.floor((end+args.end_hold)*args.fps+1e-8))+1
    with wp.ScopedDevice(args.device):
        # Build the same model/scene; sim.step() is never called during replay.
        sim = Go2Run(terrain, speed=meta["speed_m_s"], settle=meta["settle_s"],
                     friction=meta["friction"], rock_config=rock_config, device=args.device,
                     rock_overrides=[Rock(**rock) for rock in meta.get("rocks", [])])
        if "rock_transform_world" in data:
            transforms = sim.model.shape_transform.numpy()
            transforms[sim.rock_shape_ids] = data["rock_transform_world"]
            sim.model.shape_transform.assign(transforms)
        viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True, num_frames=count)
        recorder = None
        try:
            recorder = VideoRecorder(args.output, args.fps)
            viewer.set_model(sim.model)
            if sim.is_mountain:
                terrain.log_visual(viewer, sim.device)
            follow = FollowCamera.from_simulation(sim, distance=args.camera_distance, height=args.camera_height)
            for frame in range(count):
                playback_time = frame/args.fps
                physical_time = min(playback_time, end)
                index = min(int(np.searchsorted(saved_time, physical_time)), len(saved_time)-1)
                sim.state.joint_q.assign(data["joint_q"][index])
                sim.state.joint_qd.assign(data["joint_qd"][index])
                newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
                set_view(viewer, sim, args.camera, follow, physical_time, initial=frame == 0)
                viewer.begin_frame(playback_time)
                viewer.log_state(sim.state)
                viewer.end_frame()
                status = (meta["status"] if physical_time >= end else
                          "settling" if physical_time < meta["settle_s"] else "walking")
                def read_frame():
                    return caption(viewer.get_frame().numpy(), args.title or args.log.stem,
                                   float(saved_time[index]), status, text_scale=args.text_scale)
                recorder.capture(playback_time, read_frame)
            info = {"source_log": str(args.log), "source_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
                    "policy_sha256": meta.get("policy_sha256"), "status": meta["status"],
                    "path_progress_m": meta["path_progress_m"], "frame_count": recorder.frames,
                    "fps": args.fps, "camera": args.camera, "source_duration_s": end,
                    "camera_distance_m": args.camera_distance, "camera_height_m": args.camera_height,
                    "text_scale": args.text_scale,
                    "end_hold_s": args.end_hold, "physics_rerun": False,
                    "note": "Saved poses sampled at video timestamps; final state held at the end."}
            args.output.with_suffix(".json").write_text(json.dumps(info, indent=2)+"\n")
        finally:
            try:
                if recorder:
                    recorder.close()
            finally:
                viewer.close()
    print(f"Saved {args.output}", flush=True)


def comparison(args):
    import imageio_ffmpeg
    validate_pair(args.before, args.after)
    clips = [args.output.with_name(args.output.stem+suffix+".mp4") for suffix in ("_before", "_after")]
    for path in [args.output, *clips]:
        if path.exists():
            raise FileExistsError(f"video already exists: {path}")
    for source, clip, title in zip((args.before, args.after), clips,
                                   ("Before - original policy", "After - trained policy")):
        subprocess.run([sys.executable, "-m", "go2_rocky.record_video", "--log", str(source),
                        "--output", str(clip), "--title", title, "--fps", str(args.fps),
                        "--width", str(args.width), "--height", str(args.height), "--camera", args.camera,
                        "--camera-distance", str(args.camera_distance), "--camera-height", str(args.camera_height),
                        "--text-scale", str(args.text_scale),
                        "--end-hold", str(args.end_hold), "--device", args.device], check=True)
    durations = []
    for clip in clips:
        info = json.loads(clip.with_suffix(".json").read_text())
        durations.append(info["frame_count"]/info["fps"])
    duration = max(durations)
    filters = (f"[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={duration}[a];"
               f"[1:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={duration}[b];"
               "[a][b]hstack=inputs=2:shortest=1[out]")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-loglevel", "error",
                    "-i", str(clips[0]), "-i", str(clips[1]), "-filter_complex", filters,
                    "-map", "[out]", "-t", str(duration), "-an", "-c:v", "libx264",
                    "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.output)], check=True)
    args.output.with_suffix(".json").write_text(json.dumps({
        "before": str(args.before), "after": str(args.after), "clips": [str(p) for p in clips],
        "fps": args.fps, "duration_s": duration, "matched_geometry_and_initial_pose": True,
        "camera_distance_m": args.camera_distance, "camera_height_m": args.camera_height,
        "text_scale": args.text_scale,
        "note": "Shorter clip holds its labelled final frame; no additional physics is simulated."
    }, indent=2)+"\n")
    print(f"Saved comparison {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path)
    source.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--text-scale", type=float, default=1.0,
                        help="Label size multiplier; 2 doubles title/status text (default 1)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera", choices=("follow", "wide"), default="follow")
    parser.add_argument("--camera-distance", type=float, default=FollowCamera.DEFAULT_DISTANCE)
    parser.add_argument("--camera-height", type=float, default=FollowCamera.DEFAULT_HEIGHT)
    parser.add_argument("--end-hold", type=float, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if bool(args.before) != bool(args.after):
        parser.error("use --before and --after together")
    if not math.isfinite(args.text_scale) or not 0.5 <= args.text_scale <= 4:
        parser.error("text-scale must be finite and in [0.5, 4]")
    if args.output.suffix.lower() != ".mp4" or not 1 <= args.fps <= 50:
        parser.error("use an .mp4 output and fps in [1, 50]")
    if any(v < 64 or v % 2 for v in (args.width, args.height)):
        parser.error("video dimensions must be even and at least 64")
    if not math.isfinite(args.end_hold) or args.end_hold < 0:
        parser.error("end-hold must be finite and nonnegative")
    if any(not math.isfinite(v) or v <= 0 for v in (args.camera_distance, args.camera_height)):
        parser.error("camera distance and height must be finite and positive")
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    if args.before:
        comparison(args)
    else:
        render_log(args)


if __name__ == "__main__":
    main()
