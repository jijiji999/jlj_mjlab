"""JLJLowBody constants."""

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

JLJLOWBODY_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "jljbot" / "xml" / "lowbody.xml"
)
JLJLOWBODY_CAPSULE_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "jljbot" / "xml" / "lowbody_capsule.xml"
)
JLJLOWBODY_ASSET_DIR = JLJLOWBODY_XML.parent / "assets"
assert JLJLOWBODY_XML.exists()
assert JLJLOWBODY_CAPSULE_XML.exists()
assert JLJLOWBODY_ASSET_DIR.exists()

JLJLOWBODY_IMU_SITE = "imu_in_base_link"


def _prepare_jljlowbody_xml(xml_path: Path = JLJLOWBODY_XML) -> str:
  """Return MJCF XML adjusted for mjlab-owned actuator config."""
  tree = ET.parse(xml_path)
  root = tree.getroot()

  compiler = root.find("compiler")
  if compiler is not None:
    compiler.set("meshdir", str(JLJLOWBODY_ASSET_DIR))

  for actuator in root.findall("actuator"):
    root.remove(actuator)

  return ET.tostring(root, encoding="unicode")


def get_spec(xml_path: Path = JLJLOWBODY_XML) -> mujoco.MjSpec:
  xml = _prepare_jljlowbody_xml(xml_path)
  with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8") as mjcf:
    mjcf.write(xml)
    mjcf.flush()
    return mujoco.MjSpec.from_file(mjcf.name)


def get_capsule_spec() -> mujoco.MjSpec:
  """Return the JLJLowBody MJCF with capsule foot collision geoms."""
  return get_spec(JLJLOWBODY_CAPSULE_XML)


##
# Actuator config.
##

JLJLOWBODY_DELAY_MIN_LAG = 0
JLJLOWBODY_DELAY_MAX_LAG = 2


def _make_jljlowbody_actuator(
  joint_name: str,
  *,
  stiffness: float,
  damping: float,
  effort_limit: float,
  armature: float,
  velocity_limit: float,
) -> DcMotorActuatorCfg:
  """Create a single-joint JLJLowBody DC motor actuator config."""
  return DcMotorActuatorCfg(
    target_names_expr=(joint_name,),
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    saturation_effort=effort_limit,
    armature=armature,
    velocity_limit=velocity_limit,
    delay_min_lag=JLJLOWBODY_DELAY_MIN_LAG,
    delay_max_lag=JLJLOWBODY_DELAY_MAX_LAG,
  )


JLJLOWBODY_ACTUATOR_LEFT_HIP_PITCH = _make_jljlowbody_actuator(
  "left_hip_pitch_joint",
  stiffness=180.0,
  damping=10.0,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_LEFT_HIP_ROLL = _make_jljlowbody_actuator(
  "left_hip_roll_joint",
  stiffness=80.0,
  damping=5.5,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_LEFT_HIP_YAW = _make_jljlowbody_actuator(
  "left_hip_yaw_joint",
  stiffness=140.0,
  damping=3.0,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_LEFT_KNEE = _make_jljlowbody_actuator(
  "left_knee_joint",
  stiffness=300.0,
  damping=4.5,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_LEFT_ANKLE_PITCH = _make_jljlowbody_actuator(
  "left_ankle_pitch_joint",
  stiffness=150.0,  # 120
  damping=5.0,
  effort_limit=15.0,  # 12
  armature=0.064,  # 0.032
  velocity_limit=4.0,
)
JLJLOWBODY_ACTUATOR_LEFT_ANKLE_ROLL = _make_jljlowbody_actuator(
  "left_ankle_roll_joint",
  stiffness=150.0,  # 120
  damping=5.0,
  effort_limit=15.0,  # 12
  armature=0.064,  # 0.032
  velocity_limit=4.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_HIP_PITCH = _make_jljlowbody_actuator(
  "right_hip_pitch_joint",
  stiffness=180.0,
  damping=10.0,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_HIP_ROLL = _make_jljlowbody_actuator(
  "right_hip_roll_joint",
  stiffness=80.0,
  damping=5.5,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_HIP_YAW = _make_jljlowbody_actuator(
  "right_hip_yaw_joint",
  stiffness=140.0,
  damping=3.0,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_KNEE = _make_jljlowbody_actuator(
  "right_knee_joint",
  stiffness=300.0,
  damping=4.5,
  effort_limit=45.0,
  armature=0.055,
  velocity_limit=12.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_ANKLE_PITCH = _make_jljlowbody_actuator(
  "right_ankle_pitch_joint",
  stiffness=150.0,  # 120
  damping=5.0,
  effort_limit=15.0,  # 12
  armature=0.064,  # 0.032
  velocity_limit=4.0,
)
JLJLOWBODY_ACTUATOR_RIGHT_ANKLE_ROLL = _make_jljlowbody_actuator(
  "right_ankle_roll_joint",
  stiffness=150.0,  # 120
  damping=5.0,
  effort_limit=15.0,  # 12
  armature=0.064,  # 0.032
  velocity_limit=4.0,
)

JLJLOWBODY_ACTUATORS: tuple[DcMotorActuatorCfg, ...] = (
  JLJLOWBODY_ACTUATOR_LEFT_HIP_PITCH,
  JLJLOWBODY_ACTUATOR_LEFT_HIP_ROLL,
  JLJLOWBODY_ACTUATOR_LEFT_HIP_YAW,
  JLJLOWBODY_ACTUATOR_LEFT_KNEE,
  JLJLOWBODY_ACTUATOR_LEFT_ANKLE_PITCH,
  JLJLOWBODY_ACTUATOR_LEFT_ANKLE_ROLL,
  JLJLOWBODY_ACTUATOR_RIGHT_HIP_PITCH,
  JLJLOWBODY_ACTUATOR_RIGHT_HIP_ROLL,
  JLJLOWBODY_ACTUATOR_RIGHT_HIP_YAW,
  JLJLOWBODY_ACTUATOR_RIGHT_KNEE,
  JLJLOWBODY_ACTUATOR_RIGHT_ANKLE_PITCH,
  JLJLOWBODY_ACTUATOR_RIGHT_ANKLE_ROLL,
)

##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1),
  joint_pos={
    "left_hip_pitch_joint": -0.12,
    "right_hip_pitch_joint": 0.12,
    "left_hip_yaw_joint": 0.07,
    "right_hip_yaw_joint": -0.07,
    "left_knee_joint": -0.43,
    "right_knee_joint": 0.43,
    "left_ankle_pitch_joint": 0.24,
    "right_ankle_pitch_joint": -0.24,
  },
  joint_vel={".*": 0.0},
)

JLJLOWBODY_JOINT_SDK_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)

##
# Collision config.
##

JLJLOWBODY_FOOT_COLLISION_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{idx}_collision" for side in ("left", "right") for idx in range(1, 6)
)
JLJLOWBODY_CAPSULE_FOOT_COLLISION_NAMES = JLJLOWBODY_FOOT_COLLISION_NAMES
FOOT_COLLISION_REGEX = r"^(left|right)_foot[1-5]_collision$"
JLJLOWBODY_FOOT_FRICTION = (0.6,)
JLJLOWBODY_FOOT_SOLREF = (0.01, 1.0)
JLJLOWBODY_FOOT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_FRICTION},
  solref={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_SOLREF},
  solimp={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_SOLIMP},
  disable_other_geoms=False,
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_FRICTION},
  solref={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_SOLREF},
  solimp={FOOT_COLLISION_REGEX: JLJLOWBODY_FOOT_SOLIMP},
  disable_other_geoms=False,
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=JLJLOWBODY_FOOT_FRICTION,
  solref=JLJLOWBODY_FOOT_SOLREF,
  solimp=JLJLOWBODY_FOOT_SOLIMP,
)

##
# Final config.
##

JLJLOWBODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=JLJLOWBODY_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def _get_jljlowbody_articulation(
  enable_actuator_delay: bool = True,
) -> EntityArticulationInfoCfg:
  """Get JLJLowBody articulation config with optional command delay."""
  if enable_actuator_delay:
    return JLJLOWBODY_ARTICULATION

  return replace(
    JLJLOWBODY_ARTICULATION,
    actuators=tuple(
      replace(actuator, delay_min_lag=0, delay_max_lag=0)
      for actuator in JLJLOWBODY_ACTUATORS
    ),
  )


def get_jljlowbody_robot_cfg(enable_actuator_delay: bool = True) -> EntityCfg:
  """Get a fresh JLJLowBody robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=_get_jljlowbody_articulation(enable_actuator_delay),
  )


def get_jljlowbody_capsule_robot_cfg(enable_actuator_delay: bool = True) -> EntityCfg:
  """Get a fresh JLJLowBody robot config with capsule foot collisions."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_capsule_spec,
    articulation=_get_jljlowbody_articulation(enable_actuator_delay),
  )


JLJLOWBODY_ACTION_SCALE: dict[str, float] = {
  "left_hip_pitch_joint": 0.0625,
  "left_hip_roll_joint": 0.140625,
  "left_hip_yaw_joint": 0.08035714285714286,
  "left_knee_joint": 0.0375,
  "left_ankle_pitch_joint": 0.025,
  "left_ankle_roll_joint": 0.025,
  "right_hip_pitch_joint": 0.0625,
  "right_hip_roll_joint": 0.140625,
  "right_hip_yaw_joint": 0.08035714285714286,
  "right_knee_joint": 0.0375,
  "right_ankle_pitch_joint": 0.025,
  "right_ankle_roll_joint": 0.025,
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_jljlowbody_robot_cfg())

  viewer.launch(robot.spec.compile())
