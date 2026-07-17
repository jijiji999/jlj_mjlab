"""Tests for play-time terrain overrides."""

import pytest

from mjlab.scripts.play import (
  PLAY_RANDOM_ROUGH_BASE_THICKNESS_RATIO,
  PlayConfig,
  _apply_play_terrain_override,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.terrains import HfRandomUniformTerrainCfg


def test_play_terrain_override_keeps_task_default() -> None:
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat", play=True)

  _apply_play_terrain_override(cfg, PlayConfig())

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None


def test_play_terrain_override_can_force_plane() -> None:
  cfg = load_env_cfg("JLJLowBody-Velocity-Blind-Rough", play=True)

  _apply_play_terrain_override(cfg, PlayConfig(play_terrain="plane"))

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None
  assert cfg.scene.terrain.max_init_terrain_level is None


def test_play_terrain_override_can_force_random_rough() -> None:
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat", play=True)

  _apply_play_terrain_override(
    cfg,
    PlayConfig(
      play_terrain="random_rough",
      terrain_difficulty=0.4,
      terrain_rows=2,
      terrain_cols=3,
      terrain_seed=123,
    ),
  )

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "generator"
  assert cfg.scene.terrain.max_init_terrain_level == 1

  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert generator.curriculum is False
  assert generator.difficulty_range == (0.4, 0.4)
  assert generator.num_rows == 2
  assert generator.num_cols == 3
  assert generator.seed == 123
  assert tuple(generator.sub_terrains) == ("random_rough",)

  random_rough_cfg = generator.sub_terrains["random_rough"]
  assert isinstance(random_rough_cfg, HfRandomUniformTerrainCfg)
  assert random_rough_cfg.base_thickness_ratio == PLAY_RANDOM_ROUGH_BASE_THICKNESS_RATIO
  assert random_rough_cfg.scale_with_difficulty is True


def test_play_terrain_override_can_force_standard_rough() -> None:
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat", play=True)

  _apply_play_terrain_override(
    cfg,
    PlayConfig(
      play_terrain="rough",
      terrain_difficulty=0.8,
      terrain_rows=1,
      terrain_cols=4,
    ),
  )

  assert cfg.scene.terrain is not None
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert cfg.scene.terrain.terrain_type == "generator"
  assert generator.curriculum is False
  assert generator.difficulty_range == (0.8, 0.8)
  assert generator.num_rows == 1
  assert generator.num_cols == 4
  assert "pyramid_stairs" in generator.sub_terrains
  assert "random_rough" in generator.sub_terrains


def test_play_terrain_override_validates_difficulty() -> None:
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat", play=True)

  with pytest.raises(ValueError, match="terrain_difficulty"):
    _apply_play_terrain_override(
      cfg,
      PlayConfig(play_terrain="random_rough", terrain_difficulty=1.1),
    )
