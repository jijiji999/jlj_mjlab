"""Tests specific to velocity tasks."""

import inspect
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  GO1_ACTION_SCALE,
  JLJBOT_ACTION_SCALE,
  JLJLOWBODY_ACTION_SCALE,
  JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES,
  JLJLOWBODY_FOOT_COLLISION_NAMES,
)
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.jljbot.env_cfgs import (
  JLJBOT_FIXED_ACTION_SCALE,
  jljbot_flat_env_cfg,
)
from mjlab.tasks.velocity.config.jljlowbody import curriculums as lowbody_curriculums
from mjlab.tasks.velocity.config.jljlowbody import env_cfgs as jljlowbody_env_cfgs
from mjlab.tasks.velocity.config.jljlowbody import (
  randomization as lowbody_randomization,
)
from mjlab.tasks.velocity.config.jljlowbody import rewards as lowbody_rewards
from mjlab.tasks.velocity.config.jljlowbody.env_cfgs import (
  JLJLOWBODY_FIXED_ACTION_SCALE,
  jljlowbody_flat_env_cfg,
)
from mjlab.tasks.velocity.config.lowbodynormal.env_cfgs import (
  LOWBODYNORMAL_BODY_MASS_SCALE_RANGE,
  lowbodynormal_flat_env_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg


@pytest.fixture(scope="module")
def velocity_task_ids() -> list[str]:
  """Get all velocity task IDs."""
  return [t for t in list_tasks() if "Velocity" in t]


@pytest.fixture(scope="module")
def g1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all G1 velocity task IDs."""
  return [t for t in velocity_task_ids if "G1" in t]


@pytest.fixture(scope="module")
def go1_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all Go1 velocity task IDs."""
  return [t for t in velocity_task_ids if "Go1" in t]


@pytest.fixture(scope="module")
def rough_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all rough terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Rough" in t]


@pytest.fixture(scope="module")
def flat_velocity_task_ids(velocity_task_ids: list[str]) -> list[str]:
  """Get all flat terrain velocity task IDs."""
  return [t for t in velocity_task_ids if "Flat" in t]


def test_velocity_tasks_have_twist_command(velocity_task_ids: list[str]) -> None:
  """All velocity tasks should have a velocity command."""
  for task_id in velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "twist" in cfg.commands, f"Task {task_id} missing 'twist' command"

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
      f"Task {task_id} twist command is not UniformVelocityCommandCfg"
    )


def test_g1_velocity_has_required_sensors(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should have feet/ground and self collision sensors."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_go1_velocity_has_required_sensors(go1_velocity_task_ids: list[str]) -> None:
  """Go1 velocity tasks should have feet/ground and collision sensors."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensor_names, (
      f"Task {task_id} missing feet_ground_contact sensor"
    )
    if "Rough" in task_id:
      for name in (
        "self_collision",
        "thigh_ground_touch",
        "shank_ground_touch",
        "trunk_ground_touch",
      ):
        assert name in sensor_names, f"Task {task_id} missing {name} sensor"


def test_flat_velocity_tasks_have_plane_terrain(
  flat_velocity_task_ids: list[str],
) -> None:
  """Flat velocity tasks should have terrain_type='plane' and no terrain_generator."""
  for task_id in flat_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "plane", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, expected 'plane'"
    )
    assert cfg.scene.terrain.terrain_generator is None, (
      f"Task {task_id} has terrain_generator, expected None for flat terrain"
    )


def test_rough_velocity_tasks_have_generator_terrain(
  rough_velocity_task_ids: list[str],
) -> None:
  """Rough velocity tasks should have generator terrain."""
  for task_id in rough_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_type == "generator", (
      f"Task {task_id} terrain_type={cfg.scene.terrain.terrain_type}, "
      "expected 'generator'"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator, expected one for rough terrain"
    )


def test_rough_velocity_training_has_curriculum_enabled() -> None:
  """Rough velocity training tasks should have terrain curriculum enabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.terrain is not None, f"Task {task_id} has no terrain config"
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is True, (
      f"Task {task_id} curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected True"
    )


def test_rough_velocity_play_has_curriculum_disabled() -> None:
  """Rough velocity play tasks should have terrain curriculum disabled."""
  rough_training_tasks = [
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
  ]

  for task_id in rough_training_tasks:
    cfg = load_env_cfg(task_id, play=True)

    assert cfg.scene.terrain is not None, (
      f"Task {task_id} (play mode) has no terrain config"
    )
    assert cfg.scene.terrain.terrain_generator is not None, (
      f"Task {task_id} (play mode) has no terrain_generator"
    )
    assert cfg.scene.terrain.terrain_generator.curriculum is False, (
      f"Task {task_id} (play mode) curriculum={cfg.scene.terrain.terrain_generator.curriculum}, "
      "expected False"
    )


def test_g1_velocity_has_correct_action_scale(g1_velocity_task_ids: list[str]) -> None:
  """G1 velocity tasks should use G1_ACTION_SCALE."""
  for task_id in g1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_go1_velocity_has_correct_action_scale(
  go1_velocity_task_ids: list[str],
) -> None:
  """Go1 velocity tasks should use GO1_ACTION_SCALE."""
  for task_id in go1_velocity_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == GO1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected GO1_ACTION_SCALE"
    )


def test_jljbot_velocity_uses_fixed_action_scale_by_default() -> None:
  """JLJBot velocity tasks should default to a fixed 0.5 action scale."""
  cfg = load_env_cfg("JLJBot-Velocity-Flat")

  assert "joint_pos" in cfg.actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == JLJBOT_FIXED_ACTION_SCALE


def test_jljbot_velocity_can_use_robot_adaptive_action_scale() -> None:
  """JLJBot can switch back to the original per-joint action scales."""
  cfg = jljbot_flat_env_cfg(use_fixed_action_scale=False)

  assert "joint_pos" in cfg.actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == JLJBOT_ACTION_SCALE


def test_jljbot_actor_base_lin_vel_is_configurable() -> None:
  """JLJBot can omit base linear velocity from actor observations."""
  cfg = load_env_cfg("JLJBot-Velocity-Flat")

  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms

  cfg_with_lin_vel = jljbot_flat_env_cfg(include_actor_base_lin_vel=True)

  assert "base_lin_vel" in cfg_with_lin_vel.observations["actor"].terms
  assert "base_lin_vel" in cfg_with_lin_vel.observations["critic"].terms


def test_jljbot_velocity_randomizes_link_pseudo_inertia() -> None:
  """JLJBot velocity tasks should randomize link mass and inertia consistently."""
  cfg = load_env_cfg("JLJBot-Velocity-Flat")

  event_names = list(cfg.events)
  assert "link_pseudo_inertia" in cfg.events
  assert event_names.index("link_pseudo_inertia") < event_names.index("base_com")

  event = cfg.events["link_pseudo_inertia"]
  assert event.mode == "startup"
  assert event.func is dr.pseudo_inertia
  assert event.params["asset_cfg"].body_names == (".*",)
  assert event.params["alpha_range"] == pytest.approx(
    (0.5 * math.log(0.8), 0.5 * math.log(1.2))
  )


def test_jljbot_velocity_has_arm_deviation_reward() -> None:
  """JLJBot velocity tasks should include the private arm-deviation reward."""
  cfg = load_env_cfg("JLJBot-Velocity-Flat")

  assert "arm_deviation" in cfg.rewards

  reward = cfg.rewards["arm_deviation"]
  assert reward.func.__name__ == "arm_initial_deviation_l2"
  assert reward.weight == pytest.approx(-0.05)
  assert reward.params["std"] == pytest.approx(0.35)


def test_jljlowbody_velocity_uses_fixed_action_scale_by_default() -> None:
  """JLJLowBody velocity tasks should default to a fixed 0.5 action scale."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert "joint_pos" in cfg.actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == JLJLOWBODY_FIXED_ACTION_SCALE


def test_jljlowbody_velocity_can_use_robot_adaptive_action_scale() -> None:
  """JLJLowBody can switch to the per-joint adaptive action scales."""
  cfg = jljlowbody_flat_env_cfg(use_fixed_action_scale=False)

  assert "joint_pos" in cfg.actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == JLJLOWBODY_ACTION_SCALE


def test_jljlowbody_action_scale_is_independent_per_joint() -> None:
  """JLJLowBody adaptive action scales should target exact joint names."""
  assert len(JLJLOWBODY_ACTION_SCALE) == 12
  assert all(".*" not in name for name in JLJLOWBODY_ACTION_SCALE)


def test_jljlowbody_actor_base_lin_vel_is_configurable() -> None:
  """JLJLowBody can omit base linear velocity from actor observations."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert "base_lin_vel" not in cfg.observations["actor"].terms
  assert "base_lin_vel" in cfg.observations["critic"].terms

  cfg_with_lin_vel = jljlowbody_flat_env_cfg(include_actor_base_lin_vel=True)

  assert "base_lin_vel" in cfg_with_lin_vel.observations["actor"].terms
  assert "base_lin_vel" in cfg_with_lin_vel.observations["critic"].terms


def test_jljlowbody_velocity_omits_jljbot_upper_body_rewards() -> None:
  """JLJLowBody should not carry over arm or waist-only deviation rewards."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert "arm_deviation" not in cfg.rewards
  assert "waist_roll_pitch_deviation" not in cfg.rewards


def test_jljlowbody_velocity_cfg_is_task_local() -> None:
  """JLJLowBody should not delegate its config to JLJBot config factories."""
  source = inspect.getsource(jljlowbody_env_cfgs)

  assert "jljbot_flat_env_cfg" not in source
  assert "jljbot_rough_env_cfg" not in source


def test_jljlowbody_velocity_retains_lowbody_domain_randomization() -> None:
  """JLJLowBody should keep its own foot friction and pseudo-inertia DR."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == (
    JLJLOWBODY_FOOT_COLLISION_NAMES
  )
  assert cfg.events["foot_friction"].params["shared_random"] is True

  event_names = list(cfg.events)
  assert "link_pseudo_inertia" in cfg.events
  assert event_names.index("link_pseudo_inertia") < event_names.index("base_com")

  assert "pd_gains" in cfg.events
  pd_event = cfg.events["pd_gains"]
  assert pd_event.func is dr.pd_gains
  assert pd_event.params["asset_cfg"].actuator_names == ".*"
  assert (
    pd_event.params["kp_range"]
    == lowbody_randomization.JLJLOWBODY_PD_RANDOMIZATION_KP_RANGE
  )
  assert (
    pd_event.params["kd_range"]
    == lowbody_randomization.JLJLOWBODY_PD_RANDOMIZATION_KD_RANGE
  )
  assert pd_event.params["operation"] == "scale"


def test_jljlowbody_pd_randomization_can_be_disabled() -> None:
  """JLJLowBody PD gain DR should be configurable."""
  train_cfg = jljlowbody_flat_env_cfg(randomize_pd_gains=False)
  play_cfg = jljlowbody_flat_env_cfg(play=True)

  assert "pd_gains" not in train_cfg.events
  assert "pd_gains" not in play_cfg.events


def test_jljlowbody_actuator_delay_can_be_disabled() -> None:
  """JLJLowBody command-delay randomization should be configurable."""
  cfg = jljlowbody_flat_env_cfg(use_actuator_delay=False)
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None

  for actuator_cfg in robot_cfg.articulation.actuators:
    assert actuator_cfg.delay_min_lag == 0
    assert actuator_cfg.delay_max_lag == 0


def test_jljlowbody_velocity_penalizes_action_jumps() -> None:
  """JLJLowBody penalizes first- and second-order action jumps."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert "action_rate_l2" in cfg.rewards
  assert cfg.rewards["action_rate_l2"].func is velocity_mdp.action_rate_l2
  assert cfg.rewards["action_rate_l2"].weight < 0.0
  assert "action_acc_l2" in cfg.rewards
  assert cfg.rewards["action_acc_l2"].func is velocity_mdp.action_acc_l2
  assert cfg.rewards["action_acc_l2"].weight == pytest.approx(
    lowbody_rewards.JLJLOWBODY_ACTION_ACC_WEIGHT
  )


def test_jljlowbody_private_terms_are_not_exported_from_common_velocity_mdp() -> None:
  """JLJLowBody-only curricula should stay outside the shared velocity MDP."""
  assert not hasattr(velocity_mdp, "standing_commands_vel")
  assert not hasattr(velocity_mdp, "terrain_levels_by_step")
  assert not hasattr(velocity_mdp, "StandingVelocityStage")
  assert not hasattr(velocity_mdp, "TerrainLevelStage")


def test_jljlowbody_hip_deviation_rewards_are_private() -> None:
  """JLJLowBody hip-shaping terms should use JLJLowBody reward helpers."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")

  assert (
    cfg.rewards["hip_roll_deviation"].func
    is lowbody_rewards.hip_roll_initial_deviation_l2
  )
  assert (
    cfg.rewards["hip_pitch_deviation"].func
    is lowbody_rewards.hip_pitch_initial_deviation_l2
  )
  assert (
    cfg.rewards["hip_yaw_deviation"].func
    is lowbody_rewards.hip_yaw_initial_deviation_l2
  )


def test_jljlowbody_capsule_velocity_uses_capsule_foot_geoms() -> None:
  """Capsule-foot JLJLowBody tasks should target the capsule foot collisions."""
  cfg = load_env_cfg("JLJLowBodyCapsule-Velocity-Flat")

  assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == (
    JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES
  )
  assert cfg.events["foot_friction"].params["shared_random"] is True


def test_jljlowbody_capsule_flat_randomizes_ankle_encoder_bias() -> None:
  """Capsule flat JLJLowBody should randomize ankle encoder bias during training."""
  cfg = load_env_cfg("JLJLowBodyCapsule-Velocity-Flat")

  assert "ankle_encoder_bias" in cfg.events
  event = cfg.events["ankle_encoder_bias"]
  assert event.mode == "startup"
  assert event.func is dr.encoder_bias
  assert (
    event.params["bias_range"]
    == lowbody_randomization.JLJLOWBODY_ANKLE_ENCODER_BIAS_RANGE
  )
  assert (
    event.params["asset_cfg"].joint_names
    == lowbody_randomization.JLJLOWBODY_ANKLE_JOINT_NAMES
  )

  play_cfg = load_env_cfg("JLJLowBodyCapsule-Velocity-Flat", play=True)
  assert "ankle_encoder_bias" not in play_cfg.events

  disabled_cfg = jljlowbody_env_cfgs.jljlowbody_capsule_flat_env_cfg(
    randomize_ankle_encoder_bias=False
  )
  assert "ankle_encoder_bias" not in disabled_cfg.events


def test_jljlowbody_capsule_flat_randomizes_foot_contact_softness() -> None:
  """Capsule flat JLJLowBody should randomize foot contact softness."""
  cfg = load_env_cfg("JLJLowBodyCapsule-Velocity-Flat")

  assert "foot_contact_softness" in cfg.events
  event = cfg.events["foot_contact_softness"]
  assert event.mode == "startup"
  assert event.func is dr.geom_solref
  assert event.params["asset_cfg"].geom_names == (
    JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES
  )
  assert (
    event.params["ranges"]
    == lowbody_randomization.JLJLOWBODY_FOOT_SOLREF_TIMECONST_RANGE
  )
  assert event.params["operation"] == "abs"
  assert event.params["axes"] == [0]
  assert event.params["shared_random"] is True

  play_cfg = load_env_cfg("JLJLowBodyCapsule-Velocity-Flat", play=True)
  assert "foot_contact_softness" not in play_cfg.events

  disabled_cfg = jljlowbody_env_cfgs.jljlowbody_capsule_flat_env_cfg(
    randomize_foot_contact_softness=False
  )
  assert "foot_contact_softness" not in disabled_cfg.events


def test_jljlowbody_actor_observation_noise_is_task_local() -> None:
  """JLJLowBody actor observation noise should be configurable per task."""
  cfg = jljlowbody_flat_env_cfg(include_actor_base_lin_vel=True)
  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms

  for (
    term_name,
    noise_range,
  ) in lowbody_randomization.JLJLOWBODY_ACTOR_NOISE_RANGES.items():
    if term_name not in actor_terms:
      continue

    actor_term = actor_terms[term_name]
    assert noise_range is not None
    assert isinstance(actor_term.noise, UniformNoiseCfg)
    assert actor_term.noise.n_min == pytest.approx(noise_range[0])
    assert actor_term.noise.n_max == pytest.approx(noise_range[1])

    if term_name in critic_terms:
      assert actor_term is not critic_terms[term_name]


def test_jljlowbody_foot_swing_height_reward_is_task_local() -> None:
  """JLJLowBody should override foot swing reward params in its own config."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")
  reward = cfg.rewards["foot_swing_height"]

  for (
    param_name,
    param_value,
  ) in lowbody_rewards.JLJLOWBODY_FOOT_SWING_HEIGHT_PARAMS.items():
    if isinstance(param_value, float):
      assert reward.params[param_name] == pytest.approx(param_value)
    else:
      assert reward.params[param_name] == param_value


def test_jljlowbody_velocity_disables_standing_command_curriculum_by_default() -> None:
  """JLJLowBody should use the base command distribution by default."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Flat")
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  assert twist_cmd.rel_standing_envs == pytest.approx(0.1)
  assert twist_cmd.rel_low_speed_envs == pytest.approx(0.0)
  assert "standing_commands" not in cfg.curriculum


def test_jljlowbody_velocity_can_enable_standing_command_curriculum() -> None:
  """JLJLowBody should preserve the optional standing curriculum behavior."""
  cfg = jljlowbody_flat_env_cfg(use_standing_command_curriculum=True)
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)

  first_stage = lowbody_curriculums.JLJLOWBODY_STANDING_COMMAND_STAGES[0]
  assert twist_cmd.rel_standing_envs == pytest.approx(first_stage["rel_standing_envs"])
  assert twist_cmd.rel_low_speed_envs == pytest.approx(
    first_stage["rel_low_speed_envs"]
  )
  assert twist_cmd.low_speed_ranges.lin_vel_x == first_stage["low_speed_lin_vel_x"]
  assert twist_cmd.low_speed_ranges.lin_vel_y == first_stage["low_speed_lin_vel_y"]
  assert twist_cmd.low_speed_ranges.ang_vel_z == first_stage["low_speed_ang_vel_z"]

  assert "standing_commands" in cfg.curriculum
  curriculum = cfg.curriculum["standing_commands"]
  assert curriculum.func is lowbody_curriculums.standing_commands_vel
  assert curriculum.params["stages"] == (
    lowbody_curriculums.JLJLOWBODY_STANDING_COMMAND_STAGES
  )


def test_jljlowbody_blind_rough_task_uses_local_blind_terrain_curriculum() -> None:
  """Blind rough JLJLowBody should use step-based blind terrain progression."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Blind-Rough")

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  assert cfg.scene.terrain.terrain_generator.curriculum is True
  assert tuple(cfg.scene.terrain.terrain_generator.sub_terrains) == (
    lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_TYPES
  )
  assert (
    cfg.scene.terrain.terrain_generator.num_rows
    == lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_NUM_ROWS
  )
  assert (
    cfg.scene.terrain.terrain_generator.size
    == lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_SIZE
  )
  assert "terrain_levels" in cfg.curriculum
  assert (
    cfg.curriculum["terrain_levels"].func is lowbody_curriculums.terrain_levels_by_step
  )
  assert cfg.curriculum["terrain_levels"].params["stages"] == (
    lowbody_curriculums.JLJLOWBODY_BLIND_ROUGH_TERRAIN_STAGES
  )
  assert cfg.scene.terrain.max_init_terrain_level == 0


def test_jljlowbody_blind_rough_is_blind_to_height_scan() -> None:
  """Blind rough JLJLowBody should remove terrain scan observations."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Blind-Rough")

  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  assert "terrain_scan" not in sensor_names
  assert "height_scan" not in cfg.observations["actor"].terms
  assert "height_scan" not in cfg.observations["critic"].terms


def test_jljlowbody_blind_rough_play_randomizes_without_curriculum() -> None:
  """Blind rough play mode should disable terrain curriculum for inspection."""
  cfg = load_env_cfg("JLJLowBody-Velocity-Blind-Rough", play=True)

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  assert cfg.scene.terrain.terrain_generator.curriculum is False


def test_jljlowbody_blind_rough_uses_separate_experiment_name() -> None:
  """Blind rough JLJLowBody should log to its own experiment directory."""
  rl_cfg = load_rl_cfg("JLJLowBody-Velocity-Blind-Rough")

  assert rl_cfg.experiment_name == "jljlowbody_blind_rough_velocity"


def test_jljlowbody_capsule_tasks_use_separate_experiment_names() -> None:
  """Capsule-foot JLJLowBody tasks should log separately from sphere-foot tasks."""
  flat_rl_cfg = load_rl_cfg("JLJLowBodyCapsule-Velocity-Flat")
  blind_rl_cfg = load_rl_cfg("JLJLowBodyCapsule-Velocity-Blind-Rough")

  assert flat_rl_cfg.experiment_name == "jljlowbody_capsule_velocity"
  assert blind_rl_cfg.experiment_name == "jljlowbody_capsule_blind_rough_velocity"


def test_lowbodynormal_uses_capsule_feet_and_flat_setup() -> None:
  """LowBodyNormal should mirror the reference flat task on the capsule model."""
  cfg = load_env_cfg("lowbodynormal")

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.terrain.terrain_generator is None

  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  assert "terrain_scan" not in sensor_names
  assert "feet_ground_contact" in sensor_names
  assert "self_collision" in sensor_names

  assert cfg.events["foot_friction"].params["asset_cfg"].geom_names == (
    JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES
  )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == JLJLOWBODY_ACTION_SCALE

  rl_cfg = load_rl_cfg("lowbodynormal")
  assert rl_cfg.experiment_name == "lowbodynormal_velocity"


def test_lowbodynormal_body_mass_randomization_is_configurable() -> None:
  """LowBodyNormal can opt into link mass randomization with a scale range."""
  cfg = load_env_cfg("lowbodynormal")

  assert "link_pseudo_inertia" not in cfg.events

  custom_range = (0.7, 1.4)
  randomized_cfg = lowbodynormal_flat_env_cfg(
    randomize_body_mass=True,
    body_mass_scale_range=custom_range,
  )

  event_names = list(randomized_cfg.events)
  assert "link_pseudo_inertia" in randomized_cfg.events
  assert event_names.index("link_pseudo_inertia") < event_names.index("base_com")

  event = randomized_cfg.events["link_pseudo_inertia"]
  assert event.mode == "startup"
  assert event.func is dr.pseudo_inertia
  assert event.params["asset_cfg"].body_names == (".*",)
  assert event.params["alpha_range"] == pytest.approx(
    (0.5 * math.log(custom_range[0]), 0.5 * math.log(custom_range[1]))
  )


def test_lowbodynormal_body_mass_randomization_uses_default_range() -> None:
  """LowBodyNormal mass randomization uses its named default scale range."""
  cfg = lowbodynormal_flat_env_cfg(randomize_body_mass=True)
  event = cfg.events["link_pseudo_inertia"]

  assert event.params["alpha_range"] == pytest.approx(
    (
      0.5 * math.log(LOWBODYNORMAL_BODY_MASS_SCALE_RANGE[0]),
      0.5 * math.log(LOWBODYNORMAL_BODY_MASS_SCALE_RANGE[1]),
    )
  )


def test_step_based_terrain_curriculum_progresses_by_training_step() -> None:
  """Step-based terrain curriculum should widen the sampled terrain levels."""
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(
      terrain_generator=SimpleNamespace(
        sub_terrains={"flat": object(), "wave_low": object()}
      )
    ),
    terrain_origins=torch.zeros(10, 2, 3),
    env_origins=torch.zeros(4, 3),
    terrain_levels=torch.zeros(4, dtype=torch.long),
    terrain_types=torch.tensor([0, 1, 0, 1], dtype=torch.long),
  )
  env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain), common_step_counter=0)
  term_cfg = CurriculumTermCfg(
    func=lowbody_curriculums.terrain_levels_by_step,
    params={
      "stages": [
        {"step": 0, "min_level": 0, "max_level": 0},
        {"step": 12, "min_level": 0, "max_level": 3},
      ]
    },
  )

  mock_env = cast(Any, env)
  term = lowbody_curriculums.terrain_levels_by_step(term_cfg, mock_env)
  state = term(mock_env, torch.arange(4), stages=term_cfg.params["stages"])

  assert int(state["stage_max_level"].item()) == 0
  assert torch.all(terrain.terrain_levels == 0)

  env.common_step_counter = 12
  state = term(mock_env, torch.arange(4), stages=term_cfg.params["stages"])

  assert int(state["stage_max_level"].item()) == 3
  assert torch.all((terrain.terrain_levels >= 0) & (terrain.terrain_levels <= 3))


def test_standing_command_curriculum_progresses_by_training_step() -> None:
  """Standing command curriculum should update zero and near-zero ratios."""
  command_cfg = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-0.5, 0.5),
    ),
  )
  command_term = SimpleNamespace(cfg=command_cfg)
  env = SimpleNamespace(
    common_step_counter=0,
    command_manager=SimpleNamespace(get_term=lambda _name: command_term),
  )
  stages: list[lowbody_curriculums.StandingVelocityStage] = [
    {
      "step": 0,
      "rel_standing_envs": 0.4,
      "rel_low_speed_envs": 0.2,
      "low_speed_lin_vel_x": (-0.03, 0.03),
      "low_speed_lin_vel_y": (-0.02, 0.02),
      "low_speed_ang_vel_z": (-0.01, 0.01),
    },
    {
      "step": 10,
      "rel_standing_envs": 0.15,
      "rel_low_speed_envs": 0.1,
      "low_speed_lin_vel_x": (-0.06, 0.06),
      "low_speed_lin_vel_y": (-0.05, 0.05),
      "low_speed_ang_vel_z": (-0.04, 0.04),
    },
  ]

  mock_env = cast(Any, env)
  state = lowbody_curriculums.standing_commands_vel(
    mock_env,
    torch.arange(4),
    command_name="twist",
    stages=stages,
  )

  assert command_cfg.rel_standing_envs == pytest.approx(0.4)
  assert command_cfg.rel_low_speed_envs == pytest.approx(0.2)
  assert command_cfg.low_speed_ranges.lin_vel_x == (-0.03, 0.03)
  assert state["rel_standing_envs"].item() == pytest.approx(0.4)

  env.common_step_counter = 10
  state = lowbody_curriculums.standing_commands_vel(
    mock_env,
    torch.arange(4),
    command_name="twist",
    stages=stages,
  )

  assert command_cfg.rel_standing_envs == pytest.approx(0.15)
  assert command_cfg.rel_low_speed_envs == pytest.approx(0.1)
  assert command_cfg.low_speed_ranges.ang_vel_z == (-0.04, 0.04)
  assert state["stage_step"].item() == 10
