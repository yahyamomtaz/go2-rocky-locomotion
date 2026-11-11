"""Serializable settings for rigid-slope PPO; no optional ML imports here."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TrainingConfig:
    num_envs: int = 256
    seed: int = 11
    stage: str = "slope"
    max_slope_deg: float = 5.0
    roughness: str = "low"
    rock_count: int = 32
    episode_s: float = 20.0
    speed_min: float = 0.3
    speed_max: float = 0.8
    friction: float = 0.75
    observation_noise: bool = True
    capture: bool = True
    surface_rockiness: float = 0.0

    def __post_init__(self):
        if self.num_envs < 1 or self.rock_count < 1:
            raise ValueError("num_envs and rock_count must be positive")
        if self.stage not in ("flat", "slope", "rocky"):
            raise ValueError("stage must be flat, slope, or rocky")
        if self.roughness not in ("low", "medium", "high"):
            raise ValueError("unknown roughness")
        if not math.isfinite(self.max_slope_deg) or not 0 <= self.max_slope_deg <= 35:
            raise ValueError("max_slope_deg must be in [0, 35]")
        if not all(math.isfinite(v) for v in (self.speed_min, self.speed_max, self.episode_s, self.friction)):
            raise ValueError("training settings must be finite")
        if not 0 < self.speed_min <= self.speed_max <= 2 or self.episode_s < 1 or self.friction <= 0:
            raise ValueError("invalid speeds, episode length, or friction")
        if not math.isfinite(self.surface_rockiness) or not 0 <= self.surface_rockiness <= 3:
            raise ValueError("surface_rockiness must be in [0, 3]")
        if self.surface_rockiness and self.stage != "rocky":
            raise ValueError("physical rock relief requires --stage rocky")
