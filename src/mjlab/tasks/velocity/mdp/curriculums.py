from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


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


def _terrain_level_stats(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  """Summarize the current terrain levels for logging."""
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
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


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)
  return _terrain_level_stats(env)


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
  """Apply step-based terrain levels, similar to the velocity command curriculum."""

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


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def standing_commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: list[StandingVelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  stage = stages[0]
  for candidate in stages:
    if env.common_step_counter >= candidate["step"]:
      stage = candidate

  cfg.rel_standing_envs = stage["rel_standing_envs"]
  cfg.rel_low_speed_envs = stage["rel_low_speed_envs"]
  if "low_speed_lin_vel_x" in stage and stage["low_speed_lin_vel_x"] is not None:
    cfg.low_speed_ranges.lin_vel_x = stage["low_speed_lin_vel_x"]
  if "low_speed_lin_vel_y" in stage and stage["low_speed_lin_vel_y"] is not None:
    cfg.low_speed_ranges.lin_vel_y = stage["low_speed_lin_vel_y"]
  if "low_speed_ang_vel_z" in stage and stage["low_speed_ang_vel_z"] is not None:
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
