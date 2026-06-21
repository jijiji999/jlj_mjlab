"""JLJBot-specific reward terms.

Keep robot-private reward functions here instead of adding them to the shared
velocity MDP module. Enable them from ``env_cfgs.py`` with ``RewardTermCfg`` when
you want them to participate in training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _joint_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  joint_names: tuple[str, ...],
  asset_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
  if std <= 0.0:
    raise ValueError(f"std must be positive, got {std}.")

  robot: Entity = env.scene["robot" if asset_cfg is None else asset_cfg.name]
  if asset_cfg is None:
    joint_ids, _ = robot.find_joints(joint_names)
  else:
    joint_ids = asset_cfg.joint_ids

  default_joint_pos = robot.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos = robot.data.joint_pos[:, joint_ids]
  init_joint_pos = default_joint_pos[:, joint_ids]
  normalized_error = (joint_pos - init_joint_pos) / std
  return torch.sum(torch.square(normalized_error), dim=1)


def zero(env: ManagerBasedRlEnv) -> torch.Tensor:
  """No-op reward term useful for checking wiring and logging."""
  return torch.zeros(env.num_envs, device=env.device)


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic base-height error for JLJBot-specific height shaping.

  Use a negative reward weight if you want this to act as a penalty.
  """
  robot: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    height = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
  else:
    height = robot.data.root_link_pos_w[:, 2]
  return torch.square(height - target_height)


def hip_roll_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Quadratic penalty on hip-roll deviation from the initial joint angles.

  The return value is a positive cost:

  ``sum(((q_hip_roll - q_init_hip_roll) / std) ** 2)``.

  This is gentle for small deviations and grows faster as the hip-roll joints move
  farther from the default pose. Use a negative reward weight in ``RewardTermCfg``.
  """
  return _joint_initial_deviation_l2(
    env,
    std=std,
    joint_names=(".*_hip_roll_joint",),
    asset_cfg=asset_cfg,
  )


def hip_pitch_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Quadratic penalty on hip-pitch deviation from the initial joint angles.

  The return value is a positive cost:

  ``sum(((q_hip_pitch - q_init_hip_pitch) / std) ** 2)``.

  This is gentle for small deviations and grows faster as the hip-pitch joints move
  farther from the default pose. Use a negative reward weight in ``RewardTermCfg``.
  """
  return _joint_initial_deviation_l2(
    env,
    std=std,
    joint_names=(".*_hip_pitch_joint",),
    asset_cfg=asset_cfg,
  )


def hip_yaw_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Quadratic penalty on hip-yaw deviation from the initial joint angles.

  The return value is a positive cost:

  ``sum(((q_hip_yaw - q_init_hip_yaw) / std) ** 2)``.

  This is gentle for small deviations and grows faster as the hip-yaw joints move
  farther from the default pose. Use a negative reward weight in ``RewardTermCfg``.
  """
  return _joint_initial_deviation_l2(
    env,
    std=std,
    joint_names=(".*_hip_yaw_joint",),
    asset_cfg=asset_cfg,
  )


def waist_roll_pitch_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Quadratic penalty on waist roll/pitch deviation from initial angles.

  The return value is a positive cost:

  ``sum(((q_waist - q_init_waist) / std) ** 2)`` for waist roll and pitch.

  Use a negative reward weight in ``RewardTermCfg`` to discourage large torso
  side-bending and forward/backward bending while still allowing small posture
  corrections.
  """
  return _joint_initial_deviation_l2(
    env,
    std=std,
    joint_names=("waist_roll_joint", "waist_pitch_joint"),
    asset_cfg=asset_cfg,
  )
