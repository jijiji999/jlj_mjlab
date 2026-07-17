"""Standalone JLJLowBody velocity environment configurations."""

import math
from copy import deepcopy

from mjlab.asset_zoo.robots import (
  JLJLOWBODY_ACTION_SCALE,
  JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES,
  JLJLOWBODY_FOOT_COLLISION_NAMES,
  get_jljlowbody_capsule_robot_cfg,
  get_jljlowbody_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.jljbot import rewards as jljbot_rewards
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import flat, random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

_LINK_MASS_SCALE_RANGE = (0.9, 1.2)
_LINK_MASS_ALPHA_RANGE = (
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[0]),
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[1]),
)
_LOWBODY_FOOT_SITE_NAMES = ("left_foot", "right_foot")

JLJLOWBODY_FIXED_ACTION_SCALE = 0.5
JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE = (0.8, 1.2)
JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE = (0.8, 1.2)
JLJLOWBODY_ACTION_ACC_WEIGHT = -0.02
JLJLOWBODY_AIR_TIME_COMMAND_THRESHOLD = 0.2
JLJLOWBODY_ACTOR_NOISE_RANGES: dict[str, tuple[float, float] | None] = {
  # Tune these per-term ranges here without affecting other velocity tasks.
  "base_lin_vel": (-0.5, 0.5),
  "base_ang_vel": (-0.2, 0.2),
  "projected_gravity": (-0.08, 0.08),
  "joint_pos": (-0.04, 0.04),
  "joint_vel": (-1.5, 1.5),
  "height_scan": (-0.1, 0.1),
}
JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS: dict[str, str | float] = {
  # Tune this reward locally for JLJLowBody without touching other tasks.
  "sensor_name": "feet_ground_contact",
  "height_sensor_name": "foot_height_scan",
  "target_height": 0.15,
  "command_name": "twist",
  "command_threshold": 0.05,
}
JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE = (6.0, 6.0)
JLJLOWBODY_BLIND_ROUGH_NUM_ROWS = 6
JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES = (
  "flat",
  "random_rough_low",
)
JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES: list[mdp.TerrainLevelStage] = [
  {"step": 0, "min_level": 0, "max_level": 0},
  {"step": 10000 * 24, "min_level": 0, "max_level": 1},
  {"step": 20000 * 24, "min_level": 1, "max_level": 3},
  {"step": 50000 * 24, "min_level": 2, "max_level": 5},
]
JLJLOWBODY_STANDING_COMMAND_STAGES: list[mdp.StandingVelocityStage] = [
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


def _get_jljlowbody_action_scale(
  use_fixed_action_scale: bool,
) -> float | dict[str, float]:
  """Return the JLJLowBody action scale configuration."""
  if use_fixed_action_scale:
    return JLJLOWBODY_FIXED_ACTION_SCALE
  return JLJLOWBODY_ACTION_SCALE


def _apply_jljlowbody_actor_noise_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
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


def _remove_terrain_scan_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Make the task blind to terrain height by dropping terrain scan observations."""
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def _make_jljlowbody_blind_rough_terrain_cfg(play: bool) -> TerrainGeneratorCfg:
  """Build a light undulating-terrain curriculum for blind rough walking."""
  terrain_cfg = TerrainGeneratorCfg(
    size=JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE,
    border_width=20.0,
    num_rows=JLJLOWBODY_BLIND_ROUGH_NUM_ROWS,
    num_cols=len(JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES),
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.3),
      "random_rough_low": random_rough(
        proportion=0.7,
        noise_range=(0.01, 0.08),
        noise_step=0.01,
        horizontal_scale=0.2,
        downsampled_scale=0.2,
        border_width=0.5,
        base_thickness_ratio=10.0,
        scale_with_difficulty=True,
      ),
    },
    add_lights=True,
  )

  if play:
    terrain_cfg.curriculum = False
    terrain_cfg.num_cols = 5
    terrain_cfg.num_rows = 5
    terrain_cfg.border_width = 10.0

  return terrain_cfg


def _customize_jljlowbody_cfg(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
  use_fixed_action_scale: bool,
  use_capsule_feet: bool = False,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Apply JLJLowBody-specific velocity task customizations."""
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 1024
  cfg.sim.nconmax = 256

  robot_cfg_fn = (
    get_jljlowbody_capsule_robot_cfg if use_capsule_feet else get_jljlowbody_robot_cfg
  )
  foot_collision_names = (
    JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES
    if use_capsule_feet
    else JLJLOWBODY_FOOT_COLLISION_NAMES
  )

  cfg.scene.entities = {"robot": robot_cfg_fn(enable_actuator_delay=use_actuator_delay)}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site_name, entity="robot")
        for site_name in _LOWBODY_FOOT_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = _get_jljlowbody_action_scale(use_fixed_action_scale)
  _apply_jljlowbody_actor_noise_overrides(cfg)

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  standing_stage = JLJLOWBODY_STANDING_COMMAND_STAGES[0]
  low_speed_lin_vel_x = standing_stage["low_speed_lin_vel_x"]
  low_speed_lin_vel_y = standing_stage["low_speed_lin_vel_y"]
  low_speed_ang_vel_z = standing_stage["low_speed_ang_vel_z"]
  assert low_speed_lin_vel_x is not None
  assert low_speed_lin_vel_y is not None
  assert low_speed_ang_vel_z is not None
  twist_cmd.rel_standing_envs = standing_stage["rel_standing_envs"]
  twist_cmd.rel_low_speed_envs = standing_stage["rel_low_speed_envs"]
  twist_cmd.low_speed_ranges.lin_vel_x = low_speed_lin_vel_x
  twist_cmd.low_speed_ranges.lin_vel_y = low_speed_lin_vel_y
  twist_cmd.low_speed_ranges.ang_vel_z = low_speed_ang_vel_z
  cfg.curriculum["standing_commands"] = CurriculumTermCfg(
    func=mdp.standing_commands_vel,
    params={
      "command_name": "twist",
      "stages": JLJLOWBODY_STANDING_COMMAND_STAGES,
    },
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_collision_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  base_com_event = cfg.events.pop("base_com")
  cfg.events["link_pseudo_inertia"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      # dr.pseudo_inertia scales mass and inertia by exp(2 * alpha).
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

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.03}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = _LOWBODY_FOOT_SITE_NAMES

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=mdp.action_acc_l2,
    weight=JLJLOWBODY_ACTION_ACC_WEIGHT,
  )
  cfg.rewards["air_time"].weight = 0.05
  cfg.rewards["air_time"].params["command_threshold"] = (
    JLJLOWBODY_AIR_TIME_COMMAND_THRESHOLD
  )
  cfg.rewards["foot_swing_height"].params.update(JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["hip_roll_deviation"] = RewardTermCfg(
    func=jljbot_rewards.hip_roll_initial_deviation_l2,
    weight=-0.4,
    params={"std": 0.3},
  )
  cfg.rewards["hip_pitch_deviation"] = RewardTermCfg(
    func=jljbot_rewards.hip_pitch_initial_deviation_l2,
    weight=-0.1,
    params={"std": 0.3},
  )
  cfg.rewards["hip_yaw_deviation"] = RewardTermCfg(
    func=jljbot_rewards.hip_yaw_initial_deviation_l2,
    weight=-0.15,
    params={"std": 0.3},
  )

  cfg.rewards.pop("waist_roll_pitch_deviation", None)
  cfg.rewards.pop("arm_deviation", None)

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", True)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if (
      cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
    ):
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def jljlowbody_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  use_capsule_feet: bool = False,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg(include_actor_base_lin_vel=include_actor_base_lin_vel)
  return _customize_jljlowbody_cfg(
    cfg,
    play=play,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=use_capsule_feet,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )


def jljlowbody_flat_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  use_capsule_feet: bool = False,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody flat terrain velocity configuration."""
  cfg = jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=use_capsule_feet,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  _remove_terrain_scan_observations(cfg)

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def jljlowbody_blind_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  use_capsule_feet: bool = False,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create blind JLJLowBody rough terrain velocity configuration."""
  cfg = jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=use_capsule_feet,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _make_jljlowbody_blind_rough_terrain_cfg(play)
  cfg.scene.terrain.max_init_terrain_level = 0 if not play else 5
  _remove_terrain_scan_observations(cfg)
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_by_step,
      params={"stages": JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES},
    )

  return cfg


def jljlowbody_capsule_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody rough velocity config with capsule foot collisions."""
  return jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )


def jljlowbody_capsule_flat_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody flat velocity config with capsule foot collisions."""
  return jljlowbody_flat_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )


def jljlowbody_capsule_blind_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create blind rough JLJLowBody config with capsule foot collisions."""
  return jljlowbody_blind_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
  )
