"""JLJLowBody-private reward helpers and reward tuning constants."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


JLJLOWBODY_ACTION_ACC_WEIGHT = -0.02
JLJLOWBODY_AIR_TIME_COMMAND_THRESHOLD = 0.2
JLJLOWBODY_GAIT_PHASE_PERIOD = 0.8
JLJLOWBODY_FOOT_GAIT_WEIGHT = 0.5
JLJLOWBODY_FOOT_GAIT_PARAMS: dict[str, str | float | list[float]] = {
  "period": JLJLOWBODY_GAIT_PHASE_PERIOD,
  "offset": [0.0, 0.5],
  "threshold": 0.56,
  "command_threshold": 0.1,
  "command_name": "twist",
  "sensor_name": "feet_ground_contact",
}
JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS: dict[str, str | float] = {
  "sensor_name": "feet_ground_contact",
  "height_sensor_name": "foot_height_scan",
  "target_height": 0.2,
  "command_name": "twist",
  "command_threshold": 0.05,
}


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


def hip_roll_initial_deviation_l2(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Quadratic penalty on lowbody hip-roll deviation from initial angles."""
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
  """Quadratic penalty on lowbody hip-pitch deviation from initial angles."""
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
  """Quadratic penalty on lowbody hip-yaw deviation from initial angles."""
  return _joint_initial_deviation_l2(
    env,
    std=std,
    joint_names=(".*_hip_yaw_joint",),
    asset_cfg=asset_cfg,
  )
