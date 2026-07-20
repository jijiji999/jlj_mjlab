"""Tests for jljlowbody_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.jljbot import jljlowbody_constants
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def jljlowbody_entity() -> Entity:
  return Entity(jljlowbody_constants.get_jljlowbody_robot_cfg())


@pytest.fixture(scope="module")
def jljlowbody_model(jljlowbody_entity: Entity) -> mujoco.MjModel:
  return jljlowbody_entity.spec.compile()


@pytest.fixture(scope="module")
def jljlowbody_capsule_entity() -> Entity:
  return Entity(jljlowbody_constants.get_jljlowbody_capsule_robot_cfg())


@pytest.fixture(scope="module")
def jljlowbody_capsule_model(
  jljlowbody_capsule_entity: Entity,
) -> mujoco.MjModel:
  return jljlowbody_capsule_entity.spec.compile()


@pytest.mark.parametrize(
  "actuator_config",
  jljlowbody_constants.JLJLOWBODY_ACTUATORS,
)
def test_actuator_parameters(jljlowbody_model, actuator_config) -> None:
  (actuator_name,) = actuator_config.target_names_expr
  actuator = jljlowbody_model.actuator(actuator_name)
  assert actuator.gainprm[0] == 1.0
  np.testing.assert_array_equal(actuator.biasprm[:3], np.zeros(3))
  assert actuator.forcerange[0] == -actuator_config.effort_limit
  assert actuator.forcerange[1] == actuator_config.effort_limit


def test_jljlowbody_actuator_configs_are_per_joint() -> None:
  assert len(jljlowbody_constants.JLJLOWBODY_ACTUATORS) == len(
    jljlowbody_constants.JLJLOWBODY_JOINT_SDK_NAMES
  )

  configured_joint_names = []
  for actuator_cfg in jljlowbody_constants.JLJLOWBODY_ACTUATORS:
    assert len(actuator_cfg.target_names_expr) == 1
    configured_joint_names.append(actuator_cfg.target_names_expr[0])

  assert (
    tuple(configured_joint_names) == jljlowbody_constants.JLJLOWBODY_JOINT_SDK_NAMES
  )


def test_jljlowbody_action_scale_is_per_joint() -> None:
  assert tuple(jljlowbody_constants.JLJLOWBODY_ACTION_SCALE) == (
    jljlowbody_constants.JLJLOWBODY_JOINT_SDK_NAMES
  )


def test_jljlowbody_action_scale_is_computed_from_pd() -> None:
  for actuator_cfg in jljlowbody_constants.JLJLOWBODY_ACTUATORS:
    (joint_name,) = actuator_cfg.target_names_expr
    expected = (
      jljlowbody_constants.JLJLOWBODY_ACTION_SCALE_GAIN
      * actuator_cfg.effort_limit
      / actuator_cfg.stiffness
    )
    assert jljlowbody_constants.JLJLOWBODY_ACTION_SCALE[joint_name] == pytest.approx(
      expected
    )


def test_jljlowbody_actuators_have_uniform_small_command_delay() -> None:
  for actuator_cfg in jljlowbody_constants.JLJLOWBODY_ACTUATORS:
    assert actuator_cfg.delay_min_lag == jljlowbody_constants.JLJLOWBODY_DELAY_MIN_LAG
    assert actuator_cfg.delay_max_lag == jljlowbody_constants.JLJLOWBODY_DELAY_MAX_LAG


def test_jljlowbody_actuator_delay_can_be_disabled() -> None:
  robot_cfg = jljlowbody_constants.get_jljlowbody_robot_cfg(enable_actuator_delay=False)
  assert robot_cfg.articulation is not None

  for actuator_cfg in robot_cfg.articulation.actuators:
    assert actuator_cfg.delay_min_lag == 0
    assert actuator_cfg.delay_max_lag == 0


def test_keyframe_base_position(jljlowbody_model) -> None:
  data = mujoco.MjData(jljlowbody_model)
  mujoco.mj_resetDataKeyframe(jljlowbody_model, data, 0)
  mujoco.mj_forward(jljlowbody_model, data)
  np.testing.assert_array_equal(data.qpos[:3], jljlowbody_constants.INIT_STATE.pos)
  np.testing.assert_array_equal(data.qpos[3:7], jljlowbody_constants.INIT_STATE.rot)


def test_keyframe_joint_positions(jljlowbody_entity, jljlowbody_model) -> None:
  key = jljlowbody_model.key("init_state")
  expected_joint_pos = jljlowbody_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, jljlowbody_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    jljlowbody_entity.joint_names, expected_values, strict=True
  ):
    joint = jljlowbody_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch",
    )


def test_jljlowbody_entity_creation(jljlowbody_entity) -> None:
  assert jljlowbody_entity.num_actuators == 12
  assert jljlowbody_entity.num_joints == 12
  assert jljlowbody_entity.is_actuated
  assert not jljlowbody_entity.is_fixed_base


def test_jljlowbody_sites_and_collision_geoms(jljlowbody_model) -> None:
  site_names = {jljlowbody_model.site(i).name for i in range(jljlowbody_model.nsite)}
  assert {
    "left_foot",
    "right_foot",
    jljlowbody_constants.JLJLOWBODY_IMU_SITE,
  }.issubset(site_names)

  foot_pattern = re.compile(jljlowbody_constants.FOOT_COLLISION_REGEX)
  foot_geoms = []
  for i in range(jljlowbody_model.ngeom):
    geom = jljlowbody_model.geom(i)
    if foot_pattern.match(geom.name):
      foot_geoms.append(geom.name)
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6
      np.testing.assert_array_equal(
        geom.solref,
        np.array(jljlowbody_constants.JLJLOWBODY_FOOT_SOLREF),
      )
      np.testing.assert_array_equal(
        geom.solimp,
        np.array(jljlowbody_constants.JLJLOWBODY_FOOT_SOLIMP),
      )

  assert set(foot_geoms) == set(jljlowbody_constants.JLJLOWBODY_FOOT_COLLISION_NAMES)


def test_jljlowbody_foot_collision_geoms_are_spheres(jljlowbody_model) -> None:
  for geom_name in jljlowbody_constants.JLJLOWBODY_FOOT_COLLISION_NAMES:
    geom = jljlowbody_model.geom(geom_name)
    assert jljlowbody_model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_SPHERE


def test_jljlowbody_capsule_foot_collision_geoms_are_capsules(
  jljlowbody_capsule_model,
) -> None:
  for geom_name in jljlowbody_constants.JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES:
    geom = jljlowbody_capsule_model.geom(geom_name)
    assert jljlowbody_capsule_model.geom_type[geom.id] == (
      mujoco.mjtGeom.mjGEOM_CAPSULE
    )


def test_jljlowbody_capsule_foot_collision_parameters(
  jljlowbody_capsule_model,
) -> None:
  foot_pattern = re.compile(jljlowbody_constants.FOOT_COLLISION_REGEX)
  foot_geoms = []
  for i in range(jljlowbody_capsule_model.ngeom):
    geom = jljlowbody_capsule_model.geom(i)
    if foot_pattern.match(geom.name):
      foot_geoms.append(geom.name)
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6
      np.testing.assert_array_equal(
        geom.solref,
        np.array(jljlowbody_constants.JLJLOWBODY_FOOT_SOLREF),
      )
      np.testing.assert_array_equal(
        geom.solimp,
        np.array(jljlowbody_constants.JLJLOWBODY_FOOT_SOLIMP),
      )

  assert set(foot_geoms) == set(
    jljlowbody_constants.JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES
  )


def test_jljlowbody_builtin_sensor_targets(jljlowbody_model) -> None:
  imu_site_id = jljlowbody_model.site(jljlowbody_constants.JLJLOWBODY_IMU_SITE).id
  base_body_id = jljlowbody_model.body("base_link").id
  sensor_specs = {
    "imu_ang_vel": (mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE),
    "imu_lin_vel": (mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE),
    "imu_lin_acc": (mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE),
  }

  for sensor_name, (sensor_type, obj_type) in sensor_specs.items():
    sensor = jljlowbody_model.sensor(sensor_name)
    assert sensor.type[0] == sensor_type
    assert sensor.objtype[0] == obj_type
    assert sensor.objid[0] == imu_site_id

  upvector = jljlowbody_model.sensor("imu_upvector")
  assert upvector.type[0] == mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  assert upvector.objtype[0] == mujoco.mjtObj.mjOBJ_BODY
  assert upvector.objid[0] == 0
  assert upvector.reftype[0] == mujoco.mjtObj.mjOBJ_SITE
  assert upvector.refid[0] == imu_site_id

  angmom = jljlowbody_model.sensor("root_angmom")
  assert angmom.type[0] == mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  assert angmom.objtype[0] == mujoco.mjtObj.mjOBJ_BODY
  assert angmom.objid[0] == base_body_id


def test_jljlowbody_scene_exposes_velocity_task_imu_sensors() -> None:
  scene = Scene(
    SceneCfg(entities={"robot": jljlowbody_constants.get_jljlowbody_robot_cfg()}),
    device="cpu",
  )
  expected_sensors = {
    "robot/imu_ang_vel",
    "robot/imu_lin_vel",
    "robot/imu_lin_acc",
    "robot/imu_upvector",
    "robot/root_angmom",
  }
  assert expected_sensors.issubset(scene.sensors)
