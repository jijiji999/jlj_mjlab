"""JLJLowBody-private randomization and observation corruption wiring."""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

_LINK_MASS_SCALE_RANGE = (0.9, 1.3)
_LINK_MASS_ALPHA_RANGE = (
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[0]),
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[1]),
)

JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE = (0.8, 1.2)
JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE = (0.9, 1.1)
JLJLOWBODY_ANKLE_ENCODER_BIAS_RANGE = (-0.06, 0.06)
JLJLOWBODY_ANKLE_JOINT_NAMES = (".*_ankle_.*_joint",)
JLJLOWBODY_FOOT_SOLREF_TIMECONST_RANGE = (0.004, 0.015)
JLJLOWBODY_ACTOR_NOISE_RANGES: dict[str, tuple[float, float] | None] = {
  "base_lin_vel": (-0.5, 0.5),
  "base_ang_vel": (-0.2, 0.2),
  "projected_gravity": (-0.05, 0.05),
  "joint_pos": (-0.04, 0.04),
  "joint_vel": (-1.5, 1.5),
  "height_scan": (-0.1, 0.1),
}


def apply_actor_noise_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Detach actor observation terms and apply JLJLowBody-local noise settings."""
  actor_group = cfg.observations["actor"]
  actor_terms = dict(actor_group.terms)

  for term_name, noise_range in JLJLOWBODY_ACTOR_NOISE_RANGES.items():
    if term_name not in actor_terms:
      continue

    term_cfg = deepcopy(actor_terms[term_name])
    term_cfg.noise = (
      None
      if noise_range is None
      else Unoise(n_min=noise_range[0], n_max=noise_range[1])
    )
    actor_terms[term_name] = term_cfg

  actor_group.terms = actor_terms


def apply_domain_randomization(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_collision_names: tuple[str, ...],
  randomize_pd_gains: bool,
  play: bool,
) -> None:
  """Install JLJLowBody-only domain randomization events into a velocity config."""
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_collision_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  base_com_event = cfg.events.pop("base_com")
  cfg.events["link_pseudo_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": _LINK_MASS_ALPHA_RANGE,
    },
  )
  if randomize_pd_gains and not play:
    cfg.events["pd_gains"] = EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=".*"),
        "kp_range": JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE,
        "kd_range": JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE,
        "operation": "scale",
      },
    )
  cfg.events["base_com"] = base_com_event


def apply_ankle_encoder_bias_randomization(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> None:
  """Add a JLJLowBody-only ankle encoder bias randomization event."""
  if play:
    return

  cfg.events["ankle_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=JLJLOWBODY_ANKLE_JOINT_NAMES),
      "bias_range": JLJLOWBODY_ANKLE_ENCODER_BIAS_RANGE,
    },
  )


def apply_foot_contact_softness_randomization(
  cfg: ManagerBasedRlEnvCfg,
  *,
  foot_collision_names: tuple[str, ...],
  play: bool,
) -> None:
  """Randomize JLJLowBody foot contact softness through ``solref[0]``."""
  if play:
    return

  cfg.events["foot_contact_softness"] = EventTermCfg(
    mode="startup",
    func=dr.geom_solref,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_collision_names),
      "ranges": JLJLOWBODY_FOOT_SOLREF_TIMECONST_RANGE,
      "operation": "abs",
      "axes": [0],
      "shared_random": True,
    },
  )
