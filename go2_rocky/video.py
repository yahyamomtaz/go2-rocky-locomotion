"""Stream MP4 frames using an optional bundled FFmpeg encoder."""

from pathlib import Path


class VideoRecorder:
    def __init__(self, path, fps=30):
        if not isinstance(fps, int) or not 1 <= fps <= 50:
            raise ValueError("video fps must be an integer in [1, 50]")
        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("video output must end in .mp4")
        if self.path.exists():
            raise FileExistsError(f"video already exists: {self.path}")
        try:
            import imageio.v2 as imageio
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("Install video recording with: uv sync --extra video") from error
        imageio_ffmpeg.get_ffmpeg_exe()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=8,
                                        macro_block_size=1, pixelformat="yuv420p",
                                        ffmpeg_log_level="error", output_params=["-movflags", "+faststart"])
        self.fps, self.frames = fps, 0

    def capture(self, time, read_frame):
        """Sample simulation time so slow rendering never changes playback speed."""
        if self.frames/self.fps > time+1e-9:
            return
        pixels = read_frame()
        while self.frames/self.fps <= time+1e-9:
            self.writer.append_data(pixels)
            self.frames += 1

    def close(self):
        self.writer.close()


def caption(pixels, title, time, status, text_scale=1.0):
    from PIL import Image, ImageDraw
    frame = Image.fromarray(pixels)
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((12, 12, min(frame.width-12, 12+608*text_scale), 12+66*text_scale),
                   fill=(0, 0, 0, 180))
    draw.text((12+12*text_scale, 12+8*text_scale), title, fill="white",
              font_size=max(1, round(22*text_scale)))
    draw.text((12+12*text_scale, 12+37*text_scale),
              f"Simulation time: {time:.2f} s   |   {status.replace('_', ' ')}",
              fill="white", font_size=max(1, round(17*text_scale)))
    import numpy as np
    return np.asarray(frame)
