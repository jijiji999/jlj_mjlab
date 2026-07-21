"""LowBodyNormal-private MDP helpers matching the reference G1 flat task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return foot site world heights, as in the reference Unitree-G1-Flat task."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded base linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return torch.exp(-(xy_error + (2.0 * z_error)) / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward tracking commanded yaw angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + (0.05 * xy_error)) / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize tilt of the selected body, or the root body if none is selected."""
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target foot height weighted by horizontal speed."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  cost = torch.sum(
    torch.abs(foot_z - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1
  )
  if command_name is None:
    return cost

  command = env.command_manager.get_command(command_name)
  if command is None:
    return cost
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return cost * (total_command > command_threshold).float()


def stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviation from default pose when the command is near zero."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (
    torch.sum(torch.square(joint_error), dim=1)
    * (total_command < command_threshold).float()
  )
