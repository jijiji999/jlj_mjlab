"""JLJBot velocity environment configurations."""

import math

from mjlab.asset_zoo.robots import (
  JLJBOT_ACTION_SCALE,
  JLJBOT_FOOT_COLLISION_NAMES,
  get_jljbot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from . import rewards as jljbot_rewards

_LINK_MASS_SCALE_RANGE = (0.8, 1.2)
_LINK_MASS_ALPHA_RANGE = (
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[0]),
  0.5 * math.log(_LINK_MASS_SCALE_RANGE[1]),
)
JLJBOT_FIXED_ACTION_SCALE = 0.5
JLJBOT_AIR_TIME_COMMAND_THRESHOLD = 0.2


def _get_jljbot_action_scale(
  use_fixed_action_scale: bool,
) -> float | dict[str, float]:
  """Return the JLJBot action scale configuration."""
  if use_fixed_action_scale:
    return JLJBOT_FIXED_ACTION_SCALE
  return JLJBOT_ACTION_SCALE


def jljbot_rough_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create jljbot rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg(include_actor_base_lin_vel=include_actor_base_lin_vel)

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 1024
  cfg.sim.nconmax = 256

  cfg.scene.entities = {"robot": get_jljbot_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
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
  joint_pos_action.scale = _get_jljbot_action_scale(use_fixed_action_scale)

  cfg.viewer.body_name = "base_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = JLJBOT_FOOT_COLLISION_NAMES
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
  cfg.events["base_com"] = base_com_event

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
  # - Waist roll/pitch stay tight to keep the torso upright and stable.
  # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # Running values are ~1.5-2x walking values to accommodate larger motion range.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    r".*ankle_roll.*": 0.15,
    # Waist.
    r".*waist_yaw.*": 0.3,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.2,
    # Arms.
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.05
  cfg.rewards["air_time"].params["command_threshold"] = (
    JLJBOT_AIR_TIME_COMMAND_THRESHOLD
  )

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

  cfg.rewards["waist_roll_pitch_deviation"] = RewardTermCfg(
    func=jljbot_rewards.waist_roll_pitch_initial_deviation_l2,
    weight=-0.2,
    params={"std": 0.2},
  )

  cfg.rewards["arm_deviation"] = RewardTermCfg(
    func=jljbot_rewards.arm_initial_deviation_l2,
    weight=-0.05,
    params={"std": 0.35},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def jljbot_flat_env_cfg(
  play: bool = False,
  include_actor_base_lin_vel: bool = False,
  use_fixed_action_scale: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create JLJBot flat terrain velocity configuration."""
  cfg = jljbot_rough_env_cfg(
    play=play,
    include_actor_base_lin_vel=include_actor_base_lin_vel,
    use_fixed_action_scale=use_fixed_action_scale,
  )

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
