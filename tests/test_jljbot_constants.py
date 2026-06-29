"""Tests for jljbot_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.jljbot import jljbot_constants
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def jljbot_entity() -> Entity:
  return Entity(jljbot_constants.get_jljbot_robot_cfg())


@pytest.fixture(scope="module")
def jljbot_model(jljbot_entity: Entity) -> mujoco.MjModel:
  return jljbot_entity.spec.compile()


@pytest.mark.parametrize(
  "actuator_config",
  [
    jljbot_constants.JLJBOT_ACTUATOR_HIP_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_HIP_ROLL_KNEE,
    jljbot_constants.JLJBOT_ACTUATOR_HIP_YAW_WAIST_YAW,
    jljbot_constants.JLJBOT_ACTUATOR_WAIST_ROLL_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_ROLL,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_YAW_ANKLE,
    jljbot_constants.JLJBOT_ACTUATOR_ELBOW,
    jljbot_constants.JLJBOT_ACTUATOR_WRIST,
  ],
)
def test_actuator_parameters(jljbot_model, actuator_config) -> None:
  for i in range(jljbot_model.nu):
    actuator = jljbot_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == actuator_config.stiffness
      assert actuator.biasprm[1] == -actuator_config.stiffness
      assert actuator.biasprm[2] == -actuator_config.damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_jljbot_actuators_have_uniform_small_command_delay() -> None:
  actuator_cfgs = (
    jljbot_constants.JLJBOT_ACTUATOR_HIP_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_HIP_ROLL_KNEE,
    jljbot_constants.JLJBOT_ACTUATOR_HIP_YAW_WAIST_YAW,
    jljbot_constants.JLJBOT_ACTUATOR_WAIST_ROLL_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_PITCH,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_ROLL,
    jljbot_constants.JLJBOT_ACTUATOR_SHOULDER_YAW_ANKLE,
    jljbot_constants.JLJBOT_ACTUATOR_ELBOW,
    jljbot_constants.JLJBOT_ACTUATOR_WRIST,
  )

  for actuator_cfg in actuator_cfgs:
    assert actuator_cfg.delay_min_lag == jljbot_constants.JLJBOT_DELAY_MIN_LAG
    assert actuator_cfg.delay_max_lag == jljbot_constants.JLJBOT_DELAY_MAX_LAG


def test_keyframe_base_position(jljbot_model) -> None:
  data = mujoco.MjData(jljbot_model)
  mujoco.mj_resetDataKeyframe(jljbot_model, data, 0)
  mujoco.mj_forward(jljbot_model, data)
  np.testing.assert_array_equal(data.qpos[:3], jljbot_constants.INIT_STATE.pos)
  np.testing.assert_array_equal(data.qpos[3:7], jljbot_constants.INIT_STATE.rot)


def test_keyframe_joint_positions(jljbot_entity, jljbot_model) -> None:
  key = jljbot_model.key("init_state")
  expected_joint_pos = jljbot_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, jljbot_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    jljbot_entity.joint_names, expected_values, strict=True
  ):
    joint = jljbot_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch",
    )


def test_jljbot_entity_creation(jljbot_entity) -> None:
  assert jljbot_entity.num_actuators == 29
  assert jljbot_entity.num_joints == 29
  assert jljbot_entity.is_actuated
  assert not jljbot_entity.is_fixed_base


def test_jljbot_sites_and_collision_geoms(jljbot_model) -> None:
  site_names = {jljbot_model.site(i).name for i in range(jljbot_model.nsite)}
  assert {"left_foot", "right_foot", jljbot_constants.JLJBOT_IMU_SITE}.issubset(
    site_names
  )

  foot_pattern = re.compile(jljbot_constants.FOOT_COLLISION_REGEX)
  foot_geoms = []
  for i in range(jljbot_model.ngeom):
    geom = jljbot_model.geom(i)
    if foot_pattern.match(geom.name):
      foot_geoms.append(geom.name)
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6

  assert set(foot_geoms) == set(jljbot_constants.JLJBOT_FOOT_COLLISION_NAMES)


def test_jljbot_foot_collision_geoms_are_spheres(jljbot_model) -> None:
  for geom_name in jljbot_constants.JLJBOT_FOOT_COLLISION_NAMES:
    geom = jljbot_model.geom(geom_name)
    assert jljbot_model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_SPHERE


def test_jljbot_builtin_sensor_targets(jljbot_model) -> None:
  imu_site_id = jljbot_model.site(jljbot_constants.JLJBOT_IMU_SITE).id
  base_body_id = jljbot_model.body("base_link").id
  sensor_specs = {
    "imu_ang_vel": (mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE),
    "imu_lin_vel": (mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE),
    "imu_lin_acc": (mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE),
  }

  for sensor_name, (sensor_type, obj_type) in sensor_specs.items():
    sensor = jljbot_model.sensor(sensor_name)
    assert sensor.type[0] == sensor_type
    assert sensor.objtype[0] == obj_type
    assert sensor.objid[0] == imu_site_id

  upvector = jljbot_model.sensor("imu_upvector")
  assert upvector.type[0] == mujoco.mjtSensor.mjSENS_FRAMEZAXIS
  assert upvector.objtype[0] == mujoco.mjtObj.mjOBJ_BODY
  assert upvector.objid[0] == 0  # world body
  assert upvector.reftype[0] == mujoco.mjtObj.mjOBJ_SITE
  assert upvector.refid[0] == imu_site_id

  angmom = jljbot_model.sensor("root_angmom")
  assert angmom.type[0] == mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
  assert angmom.objtype[0] == mujoco.mjtObj.mjOBJ_BODY
  assert angmom.objid[0] == base_body_id


def test_jljbot_scene_exposes_velocity_task_imu_sensors() -> None:
  scene = Scene(
    SceneCfg(entities={"robot": jljbot_constants.get_jljbot_robot_cfg()}),
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
