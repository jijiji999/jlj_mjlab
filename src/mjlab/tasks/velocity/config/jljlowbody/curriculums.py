"""JLJLowBody-private curriculum terms and schedules."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class StandingVelocityStage(TypedDict):
  step: int
  rel_standing_envs: float
  rel_low_speed_envs: float
  low_speed_lin_vel_x: tuple[float, float] | None
  low_speed_lin_vel_y: tuple[float, float] | None
  low_speed_ang_vel_z: tuple[float, float] | None


class TerrainLevelStage(TypedDict):
  step: int
  min_level: int
  max_level: int


JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE = (6.0, 6.0)
JLJLOWBODY_BLIND_ROUGH_NUM_ROWS = 6
JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES = (
  "flat",
  "random_rough_low",
)
JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES: list[TerrainLevelStage] = [
  {"step": 0, "min_level": 0, "max_level": 0},
  {"step": 10000 * 24, "min_level": 0, "max_level": 1},
  {"step": 20000 * 24, "min_level": 1, "max_level": 3},
  {"step": 50000 * 24, "min_level": 2, "max_level": 5},
]
JLJLOWBODY_STANDING_COMMAND_STAGES: list[StandingVelocityStage] = [
  {
    "step": 0,
    "rel_standing_envs": 0.35,
    "rel_low_speed_envs": 0.2,
    "low_speed_lin_vel_x": (-0.04, 0.04),
    "low_speed_lin_vel_y": (-0.04, 0.04),
    "low_speed_ang_vel_z": (-0.04, 0.04),
  },
  {
    "step": 20000 * 24,
    "rel_standing_envs": 0.25,
    "rel_low_speed_envs": 0.15,
    "low_speed_lin_vel_x": (-0.05, 0.05),
    "low_speed_lin_vel_y": (-0.05, 0.05),
    "low_speed_ang_vel_z": (-0.05, 0.05),
  },
  {
    "step": 40000 * 24,
    "rel_standing_envs": 0.15,
    "rel_low_speed_envs": 0.1,
    "low_speed_lin_vel_x": (-0.06, 0.06),
    "low_speed_lin_vel_y": (-0.06, 0.06),
    "low_speed_ang_vel_z": (-0.06, 0.06),
  },
]


def _terrain_level_stats(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def _validate_terrain_level_stages(stages: list[TerrainLevelStage]) -> None:
  if not stages:
    raise ValueError("Terrain curriculum must define at least one stage.")
  for i in range(1, len(stages)):
    if stages[i]["step"] < stages[i - 1]["step"]:
      raise ValueError("Terrain curriculum stages must be in nondecreasing step order.")
  for stage in stages:
    if stage["min_level"] < 0:
      raise ValueError(
        f"Terrain curriculum min_level must be nonnegative, got {stage['min_level']}."
      )
    if stage["max_level"] < stage["min_level"]:
      raise ValueError(
        "Terrain curriculum max_level must be >= min_level, "
        f"got {stage['max_level']} < {stage['min_level']}."
      )


class terrain_levels_by_step:
  """Apply JLJLowBody step-based terrain level scheduling."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    stages: list[TerrainLevelStage] = cfg.params["stages"]
    _validate_terrain_level_stages(stages)
    terrain = env.scene.terrain
    assert terrain is not None
    assert terrain.terrain_origins is not None
    self._terrain = terrain
    self._stages = stages
    self._device = terrain.terrain_levels.device
    self._max_level = terrain.terrain_origins.shape[0] - 1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    stages: list[TerrainLevelStage],
  ) -> dict[str, torch.Tensor]:
    del stages
    stage = self._stages[0]
    for candidate in self._stages:
      if env.common_step_counter >= candidate["step"]:
        stage = candidate

    min_level = min(stage["min_level"], self._max_level)
    max_level = min(stage["max_level"], self._max_level)
    max_level = max(max_level, min_level)

    if isinstance(env_ids, slice):
      env_ids = torch.arange(
        self._terrain.terrain_levels.shape[0], device=self._device
      )[env_ids]

    num_envs = len(env_ids)
    if num_envs > 0:
      if min_level == max_level:
        levels = torch.full(
          (num_envs,), min_level, device=self._device, dtype=torch.long
        )
      else:
        levels = torch.randint(
          min_level,
          max_level + 1,
          (num_envs,),
          device=self._device,
        )
      self._terrain.terrain_levels[env_ids] = levels
      assert self._terrain.env_origins is not None
      assert self._terrain.terrain_origins is not None
      self._terrain.env_origins[env_ids] = self._terrain.terrain_origins[
        self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids]
      ]

    result = _terrain_level_stats(env)
    result["stage_step"] = torch.tensor(stage["step"])
    result["stage_min_level"] = torch.tensor(min_level)
    result["stage_max_level"] = torch.tensor(max_level)
    return result


def standing_commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[StandingVelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  stage = stages[0]
  for candidate in stages:
    if env.common_step_counter >= candidate["step"]:
      stage = candidate

  cfg.rel_standing_envs = stage["rel_standing_envs"]
  cfg.rel_low_speed_envs = stage["rel_low_speed_envs"]
  if stage["low_speed_lin_vel_x"] is not None:
    cfg.low_speed_ranges.lin_vel_x = stage["low_speed_lin_vel_x"]
  if stage["low_speed_lin_vel_y"] is not None:
    cfg.low_speed_ranges.lin_vel_y = stage["low_speed_lin_vel_y"]
  if stage["low_speed_ang_vel_z"] is not None:
    cfg.low_speed_ranges.ang_vel_z = stage["low_speed_ang_vel_z"]

  return {
    "stage_step": torch.tensor(stage["step"]),
    "rel_standing_envs": torch.tensor(cfg.rel_standing_envs),
    "rel_low_speed_envs": torch.tensor(cfg.rel_low_speed_envs),
    "low_speed_lin_vel_x_min": torch.tensor(cfg.low_speed_ranges.lin_vel_x[0]),
    "low_speed_lin_vel_x_max": torch.tensor(cfg.low_speed_ranges.lin_vel_x[1]),
    "low_speed_lin_vel_y_min": torch.tensor(cfg.low_speed_ranges.lin_vel_y[0]),
    "low_speed_lin_vel_y_max": torch.tensor(cfg.low_speed_ranges.lin_vel_y[1]),
    "low_speed_ang_vel_z_min": torch.tensor(cfg.low_speed_ranges.ang_vel_z[0]),
    "low_speed_ang_vel_z_max": torch.tensor(cfg.low_speed_ranges.ang_vel_z[1]),
  }
