"""JLJBot constants."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

JLJBOT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "jljbot" / "xml" / "JLJBot_collision.xml"
)
JLJBOT_ASSET_DIR = JLJBOT_XML.parent / "assets"
assert JLJBOT_XML.exists()
assert JLJBOT_ASSET_DIR.exists()

JLJBOT_IMU_SITE = "imu_in_base_link"


def _prepare_jljbot_xml() -> str:
  """Return MJCF XML adjusted for mjlab-owned actuator config."""
  tree = ET.parse(JLJBOT_XML)
  root = tree.getroot()

  compiler = root.find("compiler")
  if compiler is not None:
    compiler.set("meshdir", str(JLJBOT_ASSET_DIR))

  for actuator in root.findall("actuator"):
    root.remove(actuator)

  return ET.tostring(root, encoding="unicode")


def get_spec() -> mujoco.MjSpec:
  xml = _prepare_jljbot_xml()
  with tempfile.NamedTemporaryFile("w", suffix=".xml", encoding="utf-8") as mjcf:
    mjcf.write(xml)
    mjcf.flush()
    return mujoco.MjSpec.from_file(mjcf.name)


##
# Actuator config.
##

JLJBOT_ARMATURE = 0.01

JLJBOT_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_.*",),
  stiffness=65,
  damping=5.0,
  effort_limit=200.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_HIP_ROLL_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_roll_.*", ".*_knee_.*"),
  stiffness=70,
  damping=5,
  effort_limit=200.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_HIP_YAW_WAIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_.*", "waist_yaw_joint"),
  stiffness=50,
  damping=5,
  effort_limit=200.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_WAIST_ROLL_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
  stiffness=50.0,
  damping=3.7,
  effort_limit=54.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_pitch_.*",),
  stiffness=35.0,
  damping=3.7,
  effort_limit=54.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_SHOULDER_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_roll_.*",),
  stiffness=27.0,
  damping=3.7,
  effort_limit=54.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_SHOULDER_YAW_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_shoulder_yaw_.*", ".*_ankle_.*"),
  stiffness=20,
  damping=1.5,
  effort_limit=28.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_ELBOW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_.*",),
  stiffness=20.0,
  damping=2.0,
  effort_limit=28.0,
  armature=JLJBOT_ARMATURE,
)
JLJBOT_ACTUATOR_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist.*",),
  stiffness=10.0,
  damping=1.0,
  effort_limit=10.0,
  armature=JLJBOT_ARMATURE,
)

##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    "left_hip_pitch_joint": -0.3,
    "right_hip_pitch_joint": 0.3,
    "left_hip_yaw_joint": 0.15,
    "right_hip_yaw_joint": -0.15,
    "left_knee_joint": -0.4,
    "right_knee_joint": 0.4,
    "left_ankle_pitch_joint": 0.15,
    "right_ankle_pitch_joint": -0.15,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": -0.25,
    "right_shoulder_roll_joint": 0.25,
    ".*_elbow_joint": -0.4,
  },
  joint_vel={".*": 0.0},
)

JLJBOT_JOINT_SDK_NAMES: tuple[str, ...] = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
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

JLJBOT_FOOT_COLLISION_NAMES: tuple[str, ...] = tuple(
  f"{side}_foot{idx}_collision" for side in ("left", "right") for idx in range(1, 5)
)
FOOT_COLLISION_REGEX = r"^(left|right)_foot[1-4]_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: (0.6,)},
  disable_other_geoms=False,
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={FOOT_COLLISION_REGEX: 3, ".*_collision": 1},
  priority={FOOT_COLLISION_REGEX: 1},
  friction={FOOT_COLLISION_REGEX: (0.6,)},
  disable_other_geoms=False,
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(FOOT_COLLISION_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

JLJBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    JLJBOT_ACTUATOR_HIP_PITCH,
    JLJBOT_ACTUATOR_HIP_ROLL_KNEE,
    JLJBOT_ACTUATOR_HIP_YAW_WAIST_YAW,
    JLJBOT_ACTUATOR_WAIST_ROLL_PITCH,
    JLJBOT_ACTUATOR_SHOULDER_PITCH,
    JLJBOT_ACTUATOR_SHOULDER_ROLL,
    JLJBOT_ACTUATOR_SHOULDER_YAW_ANKLE,
    JLJBOT_ACTUATOR_ELBOW,
    JLJBOT_ACTUATOR_WRIST,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_jljbot_robot_cfg() -> EntityCfg:
  """Get a fresh JLJBot robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=JLJBOT_ARTICULATION,
  )


JLJBOT_ACTION_SCALE: dict[str, float] = {}
for actuator in JLJBOT_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  effort_limit = actuator.effort_limit
  stiffness = actuator.stiffness
  assert effort_limit is not None
  for name_expr in actuator.target_names_expr:
    JLJBOT_ACTION_SCALE[name_expr] = 0.25 * effort_limit / stiffness


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_jljbot_robot_cfg())

  viewer.launch(robot.spec.compile())
