"""Standalone JLJLowBody velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  JLJLOWBODY_ACTION_SCALE,
  JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES,
  JLJLOWBODY_FOOT_COLLISION_NAMES,
  get_jljlowbody_capsule_robot_cfg,
  get_jljlowbody_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.jljlowbody import curriculums as lowbody_curriculums
from mjlab.tasks.velocity.config.jljlowbody import (
  randomization as lowbody_randomization,
)
from mjlab.tasks.velocity.config.jljlowbody import rewards as lowbody_rewards
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import flat, random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

_LOWBODY_FOOT_SITE_NAMES = ("left_foot", "right_foot")

JLJLOWBODY_FIXED_ACTION_SCALE = 0.5


def _get_jljlowbody_action_scale(
  use_fixed_action_scale: bool,
) -> float | dict[str, float]:
  """Return the JLJLowBody action scale configuration."""
  if use_fixed_action_scale:
    return JLJLOWBODY_FIXED_ACTION_SCALE
  return JLJLOWBODY_ACTION_SCALE


def _remove_terrain_scan_observations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Make the task blind to terrain height by dropping terrain scan observations."""
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)


def _insert_observation_after(
  terms: dict[str, ObservationTermCfg],
  anchor: str,
  name: str,
  term: ObservationTermCfg,
) -> dict[str, ObservationTermCfg]:
  """Return observation terms with ``term`` inserted after ``anchor``."""
  if name in terms:
    return terms

  updated: dict[str, ObservationTermCfg] = {}
  inserted = False
  for term_name, term_cfg in terms.items():
    updated[term_name] = term_cfg
    if term_name == anchor:
      updated[name] = term
      inserted = True

  if not inserted:
    updated[name] = term
  return updated


def _add_capsule_flat_phase_gait_terms(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add gait phase observation and matching contact reward for capsule-flat."""
  phase_term = ObservationTermCfg(
    func=mdp.phase,
    params={
      "period": lowbody_rewards.JLJLOWBODY_GAIT_PHASE_PERIOD,
      "command_name": "twist",
    },
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms = _insert_observation_after(
      cfg.observations[group_name].terms,
      anchor="command",
      name="phase",
      term=phase_term,
    )

  cfg.rewards["foot_gait"] = RewardTermCfg(
    func=mdp.feet_gait,
    weight=lowbody_rewards.JLJLOWBODY_FOOT_GAIT_WEIGHT,
    params=lowbody_rewards.JLJLOWBODY_FOOT_GAIT_PARAMS,
  )


def _make_jljlowbody_blind_rough_terrain_cfg(play: bool) -> TerrainGeneratorCfg:
  """Build a light undulating-terrain curriculum for blind rough walking."""
  terrain_cfg = TerrainGeneratorCfg(
    size=lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE,
    border_width=20.0,
    num_rows=lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_NUM_ROWS,
    num_cols=len(lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES),
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
  use_standing_command_curriculum: bool = False,
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
  lowbody_randomization.apply_actor_noise_overrides(cfg)

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  if use_standing_command_curriculum:
    standing_stage = lowbody_curriculums.JLJLOWBODY_STANDING_COMMAND_STAGES[0]
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
      func=lowbody_curriculums.standing_commands_vel,
      params={
        "command_name": "twist",
        "stages": lowbody_curriculums.JLJLOWBODY_STANDING_COMMAND_STAGES,
      },
    )

  lowbody_randomization.apply_domain_randomization(
    cfg,
    foot_collision_names=foot_collision_names,
    randomize_pd_gains=randomize_pd_gains,
    play=play,
  )

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
    weight=lowbody_rewards.JLJLOWBODY_ACTION_ACC_WEIGHT,
  )
  cfg.rewards["air_time"].weight = 0.1
  cfg.rewards["air_time"].params["command_threshold"] = (
    lowbody_rewards.JLJLOWBODY_AIR_TIME_COMMAND_THRESHOLD
  )
  cfg.rewards["foot_swing_height"].params.update(
    lowbody_rewards.JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS
  )

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["hip_roll_deviation"] = RewardTermCfg(
    func=lowbody_rewards.hip_roll_initial_deviation_l2,
    weight=-0.2,
    params={"std": 0.3},
  )
  cfg.rewards["hip_pitch_deviation"] = RewardTermCfg(
    func=lowbody_rewards.hip_pitch_initial_deviation_l2,
    weight=-0.05,
    params={"std": 0.3},
  )
  cfg.rewards["hip_yaw_deviation"] = RewardTermCfg(
    func=lowbody_rewards.hip_yaw_initial_deviation_l2,
    weight=-0.10,
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
  use_standing_command_curriculum: bool = False,
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
    use_standing_command_curriculum=use_standing_command_curriculum,
  )


def jljlowbody_flat_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  use_capsule_feet: bool = False,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
  use_standing_command_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody flat terrain velocity configuration."""
  cfg = jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=use_capsule_feet,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
    use_standing_command_curriculum=use_standing_command_curriculum,
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
  use_standing_command_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create blind JLJLowBody rough terrain velocity configuration."""
  cfg = jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=use_capsule_feet,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
    use_standing_command_curriculum=use_standing_command_curriculum,
  )

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = _make_jljlowbody_blind_rough_terrain_cfg(play)
  cfg.scene.terrain.max_init_terrain_level = 0 if not play else 5
  _remove_terrain_scan_observations(cfg)
  if not play:
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=lowbody_curriculums.terrain_levels_by_step,
      params={"stages": lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES},
    )

  return cfg


def jljlowbody_capsule_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
  use_standing_command_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody rough velocity config with capsule foot collisions."""
  return jljlowbody_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
    use_standing_command_curriculum=use_standing_command_curriculum,
  )


def jljlowbody_capsule_flat_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = False,
  randomize_pd_gains: bool = True,
  randomize_ankle_encoder_bias: bool = True,
  randomize_foot_contact_softness: bool = True,
  use_actuator_delay: bool = True,
  use_standing_command_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create JLJLowBody flat velocity config with capsule foot collisions."""
  cfg = jljlowbody_flat_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
    use_standing_command_curriculum=use_standing_command_curriculum,
  )
  _add_capsule_flat_phase_gait_terms(cfg)
  if randomize_foot_contact_softness:
    lowbody_randomization.apply_foot_contact_softness_randomization(
      cfg,
      foot_collision_names=JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES,
      play=play,
    )
  if randomize_ankle_encoder_bias:
    lowbody_randomization.apply_ankle_encoder_bias_randomization(cfg, play=play)
  return cfg


def jljlowbody_capsule_blind_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
  randomize_pd_gains: bool = True,
  use_actuator_delay: bool = True,
  use_standing_command_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create blind rough JLJLowBody config with capsule foot collisions."""
  return jljlowbody_blind_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
    use_capsule_feet=True,
    randomize_pd_gains=randomize_pd_gains,
    use_actuator_delay=use_actuator_delay,
    use_standing_command_curriculum=use_standing_command_curriculum,
  )
