"""JLJLowBody joint PD response tests in mjlab/MuJoCo."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import tyro

import mjlab
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.jljbot.jljlowbody_constants import (
  JLJLOWBODY_JOINT_SDK_NAMES,
  get_jljlowbody_robot_cfg,
)
from mjlab.entity import Entity

FIELDNAMES = [
  "time_s",
  "source",
  "test",
  "joint",
  "target",
  "position",
  "velocity",
  "error",
  "torque_command",
  "actuator_force",
  "kp",
  "kd",
]


@dataclass
class LowbodyPdResponseTestCfg:
  """Configuration for JLJLowBody PD response tests."""

  joint_regex: str = ".*"
  """Regex selecting joints to test. Use e.g. '.*ankle.*' or 'left_hip_pitch_joint'."""

  output_root: str = "logs/lowbody_pd_response"
  """Directory where a timestamped run folder is written."""

  tests: str = "step,sine"
  """Comma-separated tests to run. Supported values: step, sine."""

  kp: str = ""
  """Optional comma-separated gain overrides as REGEX=VALUE, e.g. '.*ankle.*=120'."""

  kd: str = ""
  """Optional comma-separated damping overrides as REGEX=VALUE, e.g. '.*ankle.*=4'."""

  step_amplitude: float = 0.05
  """Step offset in rad."""

  sine_amplitude: float = 0.05
  """Sine amplitude in rad."""

  sine_frequency_hz: float = 0.5
  """Sine frequency in Hz."""

  pre_signal_s: float = 0.5
  """Hold the center pose this long before the step/sine starts."""

  duration_s: float = 6.0
  """Signal duration after pre_signal_s."""

  physics_dt: float = 0.001
  """MuJoCo physics timestep."""

  sample_hz: float = 200.0
  """CSV sampling rate."""

  fixed_base: bool = True
  """Clamp the floating base qpos/qvel after every step."""

  disable_gravity: bool = False
  """Set gravity to zero. Default keeps gravity on to match suspended hardware tests."""

  hold_other_joints: bool = True
  """Apply PD hold torque to all non-tested joints at their initial positions."""

  hold_kp_scale: float = 2.0
  """Kp scale for non-tested joint holding."""

  hold_kd_scale: float = 2.0
  """Kd scale for non-tested joint holding."""


@dataclass
class JointMotorParams:
  kp: float
  kd: float
  effort_limit: float
  saturation_effort: float
  velocity_limit: float


def _split_items(value: str) -> tuple[str, ...]:
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_overrides(value: str, label: str) -> list[tuple[re.Pattern[str], float]]:
  overrides: list[tuple[re.Pattern[str], float]] = []
  for item in _split_items(value):
    if "=" not in item:
      raise ValueError(f"{label} override must use REGEX=VALUE, got: {item}")
    pattern, raw_value = item.split("=", 1)
    overrides.append((re.compile(pattern), float(raw_value)))
  return overrides


def _apply_overrides(
  joint_name: str,
  default_value: float,
  overrides: list[tuple[re.Pattern[str], float]],
) -> float:
  value = default_value
  for pattern, override_value in overrides:
    if pattern.fullmatch(joint_name) or pattern.search(joint_name):
      value = override_value
  return value


def _build_joint_motor_params(
  cfg: LowbodyPdResponseTestCfg,
) -> dict[str, JointMotorParams]:
  robot_cfg = get_jljlowbody_robot_cfg()
  assert robot_cfg.articulation is not None
  kp_overrides = _parse_overrides(cfg.kp, "kp")
  kd_overrides = _parse_overrides(cfg.kd, "kd")

  params: dict[str, JointMotorParams] = {}
  for actuator in robot_cfg.articulation.actuators:
    if not isinstance(actuator, DcMotorActuatorCfg):
      continue
    for joint_name in JLJLOWBODY_JOINT_SDK_NAMES:
      if any(
        re.fullmatch(expr, joint_name) or re.search(expr, joint_name)
        for expr in actuator.target_names_expr
      ):
        params[joint_name] = JointMotorParams(
          kp=_apply_overrides(joint_name, actuator.stiffness, kp_overrides),
          kd=_apply_overrides(joint_name, actuator.damping, kd_overrides),
          effort_limit=actuator.effort_limit,
          saturation_effort=actuator.saturation_effort,
          velocity_limit=actuator.velocity_limit,
        )
  return params


def _joint_names(model: mujoco.MjModel) -> list[str]:
  names: list[str] = []
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name and model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
      names.append(name)
  return names


def _actuator_by_joint(model: mujoco.MjModel) -> dict[str, int]:
  mapping: dict[str, int] = {}
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
      continue
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name:
      mapping[joint_name] = actuator_id
  return mapping


def _initial_qpos(model: mujoco.MjModel) -> np.ndarray:
  if model.nkey > 0:
    return np.array(model.key("init_state").qpos, dtype=np.float64)
  return np.array(model.qpos0, dtype=np.float64)


def _clip_dc_motor_torque(
  params: JointMotorParams, velocity: float, torque: float
) -> float:
  vel_at_effort_limit = params.velocity_limit * (
    1.0 + params.effort_limit / params.saturation_effort
  )
  clipped_velocity = min(max(velocity, -vel_at_effort_limit), vel_at_effort_limit)
  upper = params.saturation_effort * (1.0 - clipped_velocity / params.velocity_limit)
  lower = params.saturation_effort * (-1.0 - clipped_velocity / params.velocity_limit)
  upper = min(upper, params.effort_limit)
  lower = max(lower, -params.effort_limit)
  return min(max(torque, lower), upper)


def _joint_position(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> float:
  return float(data.qpos[model.jnt_qposadr[joint_id]])


def _joint_velocity(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int) -> float:
  return float(data.qvel[model.jnt_dofadr[joint_id]])


def _target(
  test_name: str,
  center: float,
  elapsed: float,
  cfg: LowbodyPdResponseTestCfg,
) -> float:
  signal_t = max(0.0, elapsed - cfg.pre_signal_s)
  if elapsed < cfg.pre_signal_s:
    return center
  if test_name == "step":
    return center + cfg.step_amplitude
  if test_name == "sine":
    return center + cfg.sine_amplitude * math.sin(
      2.0 * math.pi * cfg.sine_frequency_hz * signal_t
    )
  raise ValueError(f"Unsupported test: {test_name}")


def _compute_summary(
  rows: list[dict[str, float | str]], test_name: str
) -> dict[str, float | str]:
  errors = np.array([float(row["error"]) for row in rows], dtype=np.float64)
  positions = np.array([float(row["position"]) for row in rows], dtype=np.float64)
  targets = np.array([float(row["target"]) for row in rows], dtype=np.float64)
  result: dict[str, float | str] = {
    "test": test_name,
    "samples": float(len(rows)),
    "rmse_rad": float(math.sqrt(float(np.mean(errors * errors))))
    if len(errors)
    else math.nan,
    "max_abs_error_rad": float(np.max(np.abs(errors))) if len(errors) else math.nan,
  }
  if test_name == "step" and len(rows):
    initial = targets[0]
    final_target = targets[-1]
    amplitude = final_target - initial
    final_position = positions[-1]
    result["final_error_rad"] = float(final_target - final_position)
    if abs(amplitude) > 1e-9:
      peak = np.max(positions) if amplitude > 0.0 else np.min(positions)
      result["overshoot_ratio"] = float((peak - final_target) / amplitude)
  if test_name == "sine" and len(rows):
    target_amp = 0.5 * (float(np.max(targets)) - float(np.min(targets)))
    position_amp = 0.5 * (float(np.max(positions)) - float(np.min(positions)))
    result["amplitude_gain"] = (
      float(position_amp / target_amp) if target_amp > 1e-9 else math.nan
    )
  return result


def _run_single_test(
  model: mujoco.MjModel,
  joint_name: str,
  test_name: str,
  params_by_joint: dict[str, JointMotorParams],
  cfg: LowbodyPdResponseTestCfg,
) -> tuple[list[dict[str, float | str]], dict[str, float | str]]:
  data = mujoco.MjData(model)
  data.qpos[:] = _initial_qpos(model)
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  actuator_ids = _actuator_by_joint(model)
  all_joint_names = _joint_names(model)
  joint_id_by_name = {
    name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in all_joint_names
  }
  tested_joint_id = joint_id_by_name[joint_name]
  tested_actuator_id = actuator_ids[joint_name]
  center_by_joint = {
    name: _joint_position(model, data, joint_id_by_name[name])
    for name in all_joint_names
  }

  root_qpos = data.qpos[:7].copy()
  root_qvel = data.qvel[:6].copy()
  total_s = cfg.pre_signal_s + cfg.duration_s
  sample_period = 1.0 / cfg.sample_hz
  next_sample_s = 0.0
  rows: list[dict[str, float | str]] = []

  while data.time <= total_s + 0.5 * cfg.physics_dt:
    data.ctrl[:] = 0.0
    target_by_joint: dict[str, float] = {}
    for current_joint_name in all_joint_names:
      if current_joint_name == joint_name:
        target = _target(test_name, center_by_joint[joint_name], data.time, cfg)
      elif cfg.hold_other_joints:
        target = center_by_joint[current_joint_name]
      else:
        continue

      joint_id = joint_id_by_name[current_joint_name]
      actuator_id = actuator_ids[current_joint_name]
      motor_params = params_by_joint[current_joint_name]
      pos = _joint_position(model, data, joint_id)
      vel = _joint_velocity(model, data, joint_id)
      kp = motor_params.kp
      kd = motor_params.kd
      if current_joint_name != joint_name:
        kp *= cfg.hold_kp_scale
        kd *= cfg.hold_kd_scale
      raw_torque = kp * (target - pos) - kd * vel
      data.ctrl[actuator_id] = _clip_dc_motor_torque(motor_params, vel, raw_torque)
      target_by_joint[current_joint_name] = target

    mujoco.mj_step(model, data)
    if cfg.fixed_base:
      data.qpos[:7] = root_qpos
      data.qvel[:6] = root_qvel
      mujoco.mj_forward(model, data)

    if data.time + 1e-12 >= next_sample_s:
      pos = _joint_position(model, data, tested_joint_id)
      vel = _joint_velocity(model, data, tested_joint_id)
      target = target_by_joint.get(joint_name, center_by_joint[joint_name])
      dof_id = model.jnt_dofadr[tested_joint_id]
      motor_params = params_by_joint[joint_name]
      rows.append(
        {
          "time_s": float(data.time),
          "source": "mjlab_mujoco",
          "test": test_name,
          "joint": joint_name,
          "target": float(target),
          "position": float(pos),
          "velocity": float(vel),
          "error": float(target - pos),
          "torque_command": float(data.ctrl[tested_actuator_id]),
          "actuator_force": float(data.qfrc_actuator[dof_id]),
          "kp": float(motor_params.kp),
          "kd": float(motor_params.kd),
        }
      )
      next_sample_s += sample_period

  summary = _compute_summary(rows, test_name)
  summary["joint"] = joint_name
  summary["kp"] = params_by_joint[joint_name].kp
  summary["kd"] = params_by_joint[joint_name].kd
  return rows, summary


def run_response_tests(cfg: LowbodyPdResponseTestCfg) -> Path:
  if cfg.physics_dt <= 0.0:
    raise ValueError("physics_dt must be > 0")
  if cfg.sample_hz <= 0.0:
    raise ValueError("sample_hz must be > 0")

  robot = Entity(get_jljlowbody_robot_cfg())
  model = robot.compile()
  model.opt.timestep = cfg.physics_dt
  if cfg.disable_gravity:
    model.opt.gravity[:] = 0.0

  joint_pattern = re.compile(cfg.joint_regex)
  selected_joints = [
    name
    for name in _joint_names(model)
    if joint_pattern.fullmatch(name) or joint_pattern.search(name)
  ]
  if not selected_joints:
    raise ValueError(f"No joints matched joint_regex={cfg.joint_regex!r}")

  requested_tests = tuple(test.lower() for test in _split_items(cfg.tests))
  for test_name in requested_tests:
    if test_name not in {"step", "sine"}:
      raise ValueError(f"Unsupported test: {test_name}")

  params_by_joint = _build_joint_motor_params(cfg)
  missing = [name for name in selected_joints if name not in params_by_joint]
  if missing:
    raise ValueError(f"Selected joints are missing mjlab actuator params: {missing}")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  output_dir = Path(cfg.output_root).expanduser() / timestamp
  output_dir.mkdir(parents=True, exist_ok=False)

  csv_path = output_dir / "joint_response.csv"
  summaries: list[dict[str, float | str]] = []
  with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
    writer.writeheader()
    for joint_name in selected_joints:
      for test_name in requested_tests:
        rows, summary = _run_single_test(
          model, joint_name, test_name, params_by_joint, cfg
        )
        writer.writerows(rows)
        summaries.append(summary)

  metadata = {
    "source": "mjlab_mujoco",
    "robot": "JLJLowBody",
    "joint_regex": cfg.joint_regex,
    "joints": selected_joints,
    "tests": requested_tests,
    "physics_dt": cfg.physics_dt,
    "sample_hz": cfg.sample_hz,
    "fixed_base": cfg.fixed_base,
    "disable_gravity": cfg.disable_gravity,
    "hold_other_joints": cfg.hold_other_joints,
    "hold_kp_scale": cfg.hold_kp_scale,
    "hold_kd_scale": cfg.hold_kd_scale,
    "step_amplitude": cfg.step_amplitude,
    "sine_amplitude": cfg.sine_amplitude,
    "sine_frequency_hz": cfg.sine_frequency_hz,
    "pre_signal_s": cfg.pre_signal_s,
    "duration_s": cfg.duration_s,
    "artifacts": {"csv": "joint_response.csv", "summary": "summary.json"},
    "runs": summaries,
  }
  (output_dir / "summary.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
  )
  return output_dir


def main() -> None:
  cfg = tyro.cli(LowbodyPdResponseTestCfg, config=mjlab.TYRO_FLAGS)
  output_dir = run_response_tests(cfg)
  print(f"Wrote lowbody PD response run: {output_dir}")


if __name__ == "__main__":
  main()
