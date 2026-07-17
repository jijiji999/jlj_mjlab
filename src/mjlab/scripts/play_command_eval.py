"""Fixed-command play evaluation for velocity policies."""

from __future__ import annotations

import csv
import json
import math
import numbers
import os
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def _get_pyplot():
  mplconfigdir = Path(tempfile.gettempdir()) / "matplotlib"
  mplconfigdir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  return plt


def default_play_command_eval_dir(
  task_id: str,
  *,
  log_dir: Path | None,
  log_root: str | Path,
) -> Path:
  """Return the default output directory for a fixed-command play evaluation."""
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  safe_task_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_id)
  if log_dir is not None:
    base_dir = Path(log_dir) / "play_command_evals"
  else:
    base_dir = Path(log_root).resolve() / "play_command_evals" / safe_task_id
  return base_dir / f"play_command_eval_{timestamp}"


def run_play_command_eval(
  *,
  env: Any,
  policy: Callable[[Any], torch.Tensor],
  task_id: str,
  output_dir: str | Path,
  command_velocity: tuple[float, float, float],
  duration_s: float,
  entity_name: str = "robot",
  command_name: str = "twist",
  env_id: int = 0,
  interval: int = 1,
  checkpoint_file: str | Path | None = None,
) -> Path:
  """Run a policy under a fixed velocity command and save joint curves."""
  recorder = PlayCommandEvalRecorder(
    env=env,
    output_dir=output_dir,
    task_id=task_id,
    entity_name=entity_name,
    command_name=command_name,
    command_velocity=command_velocity,
    duration_s=duration_s,
    env_id=env_id,
    interval=interval,
    checkpoint_file=checkpoint_file,
  )

  recorder.pin_command()
  obs = env.get_observations()
  recorder.record_sample(step=0, time_s=0.0, reward=None, done=False)
  recorder.record_physics_sample(physics_step=0, policy_step=0, time_s=0.0)

  num_steps = recorder.num_steps
  for step in range(num_steps):
    recorder.pin_command()
    with torch.no_grad():
      actions = policy(obs)
    policy_action = actions.detach().clone()
    applied_action = _applied_action(env, actions).detach().clone()
    obs, reward, done, *_ = env.step(actions)
    recorder.pin_command()
    recorder.record_sample(
      step=step + 1,
      time_s=(step + 1) * recorder.step_dt,
      reward=reward,
      done=done,
      policy_action=policy_action,
      applied_action=applied_action,
    )
    if step + 1 < num_steps:
      obs = env.get_observations()

  recorder.close()
  return recorder.output_dir


class PlayCommandEvalEnvWrapper:
  """Environment wrapper that pins a command and records joint curves in viewers."""

  def __init__(
    self,
    env: Any,
    *,
    output_dir: str | Path,
    task_id: str,
    command_velocity: tuple[float, float, float],
    duration_s: float,
    entity_name: str = "robot",
    command_name: str = "twist",
    env_id: int = 0,
    interval: int = 1,
    checkpoint_file: str | Path | None = None,
  ) -> None:
    self.env = env
    self.num_envs: int = int(env.num_envs)
    self.recorder = PlayCommandEvalRecorder(
      env=env,
      output_dir=output_dir,
      task_id=task_id,
      entity_name=entity_name,
      command_name=command_name,
      command_velocity=command_velocity,
      duration_s=duration_s,
      env_id=env_id,
      interval=interval,
      checkpoint_file=checkpoint_file,
    )
    self._closed = False
    self.recorder.pin_command()
    self.recorder.record_sample(step=0, time_s=0.0, reward=None, done=False)
    self.recorder.record_physics_sample(physics_step=0, policy_step=0, time_s=0.0)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  @property
  def device(self) -> torch.device | str:
    return self.env.device

  @property
  def cfg(self) -> Any:
    return self.env.cfg

  @property
  def unwrapped(self) -> Any:
    return self.env.unwrapped

  def get_observations(self) -> Any:
    self.recorder.pin_command()
    return self.env.get_observations()

  def reset(self, *args: Any, **kwargs: Any) -> Any:
    result = self.env.reset(*args, **kwargs)
    self.recorder.pin_command()
    return result

  def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
    self.recorder.pin_command()
    policy_action = actions.detach().clone()
    applied_action = _applied_action(self.env, actions).detach().clone()
    step_result = self.env.step(actions)
    self.recorder.pin_command()
    reward, done = self.recorder.reward_done(step_result)
    self.recorder.advance_step()
    self.recorder.record_sample(
      step=self.recorder.step_count,
      time_s=self.recorder.step_count * self.recorder.step_dt,
      reward=reward,
      done=done,
      policy_action=policy_action,
      applied_action=applied_action,
    )
    return step_result

  def close(self) -> None:
    if self._closed:
      return
    try:
      self.recorder.close()
    finally:
      self._closed = True
      self.env.close()


class PlayCommandEvalRecorder:
  """Collects joint state under a fixed command and writes report artifacts."""

  def __init__(
    self,
    *,
    env: Any,
    output_dir: str | Path,
    task_id: str,
    entity_name: str,
    command_name: str,
    command_velocity: tuple[float, float, float],
    duration_s: float,
    env_id: int,
    interval: int,
    checkpoint_file: str | Path | None,
  ) -> None:
    if duration_s <= 0.0:
      raise ValueError(f"eval_duration must be > 0, got {duration_s}.")
    if interval < 1:
      raise ValueError(f"eval_interval must be >= 1, got {interval}.")

    self.env = env
    self.output_dir = Path(output_dir).expanduser()
    self.task_id = task_id
    self.entity_name = entity_name
    self.command_name = command_name
    self.command_velocity = tuple(float(v) for v in command_velocity)
    self.duration_s = float(duration_s)
    self.env_id = int(env_id)
    self.interval = int(interval)
    self.checkpoint_file = (
      str(Path(checkpoint_file).expanduser()) if checkpoint_file is not None else None
    )
    self.step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.0))
    if self.step_dt <= 0.0:
      raise ValueError(f"Environment step_dt must be > 0, got {self.step_dt}.")
    self.physics_dt = _physics_dt(self.env, self.step_dt)
    self.decimation = _decimation(self.env, self.step_dt, self.physics_dt)
    self.num_steps = max(1, int(math.ceil(self.duration_s / self.step_dt)))
    self.step_count = 0
    self.physics_step_count = 0

    num_envs = _num_envs(env)
    if self.env_id < 0 or self.env_id >= num_envs:
      raise ValueError(
        f"eval_env_id={self.env_id} is out of range for num_envs={num_envs}"
      )

    self._entity()
    self._command_term()
    command = self._command_tensor()
    if command.ndim != 2 or command.shape[1] < 3:
      raise ValueError(
        f"Command '{self.command_name}' must have shape (num_envs, >=3), "
        f"got {tuple(command.shape)}"
      )

    self._samples: list[dict[str, Any]] = []
    self._physics_samples: list[dict[str, Any]] = []
    self._remove_physics_step_callback: Callable[[], None] | None = (
      self._install_physics_step_callback()
    )
    self._closed = False

  def advance_step(self) -> None:
    self.step_count += 1

  def reward_done(self, step_result: tuple[Any, ...]) -> tuple[Any, Any]:
    if len(step_result) >= 3:
      return step_result[1], step_result[2]
    return None, False

  def pin_command(self) -> None:
    """Force the command term to keep the requested velocity command."""
    term = self._command_term()
    command_tensor = self._command_tensor()
    fixed = torch.tensor(
      self.command_velocity,
      dtype=command_tensor.dtype,
      device=command_tensor.device,
    )
    command_tensor[:, :3] = fixed

    for attr_name in ("vel_command_b", "vel_command_w"):
      value = getattr(term, attr_name, None)
      if isinstance(value, torch.Tensor) and value.ndim == 2 and value.shape[1] >= 3:
        value[:, :3] = fixed

    for attr_name in (
      "is_heading_env",
      "is_standing_env",
      "is_world_env",
      "is_forward_env",
    ):
      value = getattr(term, attr_name, None)
      if isinstance(value, torch.Tensor) and value.dtype == torch.bool:
        value[:] = False

    time_left = getattr(term, "time_left", None)
    if isinstance(time_left, torch.Tensor):
      time_left[:] = self.duration_s + self.step_dt * (self.num_steps + 1)

  def record_sample(
    self,
    *,
    step: int,
    time_s: float,
    reward: Any,
    done: Any,
    policy_action: torch.Tensor | None = None,
    applied_action: torch.Tensor | None = None,
  ) -> None:
    """Record one sampled state from the selected environment."""
    if step % self.interval != 0 and step != self.num_steps:
      return

    entity = self._entity()
    data = entity.data
    command = self._command_tensor()[self.env_id].detach().cpu()
    lin_vel = data.root_link_lin_vel_b[self.env_id].detach().cpu()
    ang_vel = data.root_link_ang_vel_b[self.env_id].detach().cpu()

    sample = {
      "step": int(step),
      "time_s": float(time_s),
      "command_vx": float(command[0].item()),
      "command_vy": float(command[1].item()),
      "command_wz": float(command[2].item()),
      "actual_vx": float(lin_vel[0].item()),
      "actual_vy": float(lin_vel[1].item()),
      "actual_wz": float(ang_vel[2].item()),
      "reward": _optional_env_scalar(reward, self.env_id, _num_envs(self.env)),
      "done": bool(_optional_env_scalar(done, self.env_id, _num_envs(self.env))),
      "policy_action": _optional_env_tensor(policy_action, self.env_id),
      "applied_action": _optional_env_tensor(applied_action, self.env_id),
      "joint_pos": _env_tensor(data.joint_pos, self.env_id),
      "joint_pos_target": _env_tensor(data.joint_pos_target, self.env_id),
      "joint_vel": _env_tensor(data.joint_vel, self.env_id),
      "joint_torque": _env_tensor(data.qfrc_actuator, self.env_id),
      "joint_power": _env_tensor(data.qfrc_actuator * data.joint_vel, self.env_id),
    }
    self._samples.append(sample)

  def record_physics_substep(self, _sim_step: int, _sim_time_s: float) -> None:
    """Record one MuJoCo physics substep from the selected environment."""
    self.physics_step_count += 1
    max_physics_steps = self.num_steps * self.decimation
    if self.physics_step_count > max_physics_steps:
      return
    policy_step = math.ceil(self.physics_step_count / self.decimation)
    self.record_physics_sample(
      physics_step=self.physics_step_count,
      policy_step=policy_step,
      time_s=self.physics_step_count * self.physics_dt,
    )

  def record_physics_sample(
    self,
    *,
    physics_step: int,
    policy_step: int,
    time_s: float,
  ) -> None:
    """Record high-frequency joint state at the MuJoCo physics rate."""
    entity = self._entity()
    data = entity.data
    command = self._command_tensor()[self.env_id].detach().cpu()
    sample = {
      "physics_step": int(physics_step),
      "policy_step": int(policy_step),
      "time_s": float(time_s),
      "command_vx": float(command[0].item()),
      "command_vy": float(command[1].item()),
      "command_wz": float(command[2].item()),
      "joint_pos": _env_tensor(data.joint_pos, self.env_id),
      "joint_pos_target": _env_tensor(data.joint_pos_target, self.env_id),
      "joint_vel": _env_tensor(data.joint_vel, self.env_id),
      "joint_torque": _env_tensor(data.qfrc_actuator, self.env_id),
      "joint_power": _env_tensor(data.qfrc_actuator * data.joint_vel, self.env_id),
    }
    self._physics_samples.append(sample)

  def close(self) -> None:
    if self._closed:
      return
    self._uninstall_physics_step_callback()
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._write_outputs()
    self._closed = True

  def _install_physics_step_callback(self) -> Callable[[], None] | None:
    add_callback = getattr(self.env.unwrapped, "add_physics_step_callback", None)
    if not callable(add_callback):
      return None
    typed_add_callback = cast(
      Callable[[Callable[[int, float], None]], Callable[[], None]],
      add_callback,
    )
    return typed_add_callback(self.record_physics_substep)

  def _uninstall_physics_step_callback(self) -> None:
    if self._remove_physics_step_callback is None:
      return
    self._remove_physics_step_callback()
    self._remove_physics_step_callback = None

  def _entity(self) -> Any:
    scene = self.env.unwrapped.scene
    try:
      return scene[self.entity_name]
    except KeyError as err:
      available = ", ".join(scene.entities.keys())
      raise KeyError(
        f"Eval entity '{self.entity_name}' was not found. "
        f"Available entities: {available}"
      ) from err

  def _command_term(self) -> Any:
    command_manager = getattr(self.env.unwrapped, "command_manager", None)
    if command_manager is None:
      raise RuntimeError("Command evaluation requires a command manager.")
    term = command_manager.get_term(self.command_name)
    if term is None:
      raise RuntimeError(f"Command term '{self.command_name}' was not found.")
    return term

  def _command_tensor(self) -> torch.Tensor:
    command_manager = self.env.unwrapped.command_manager
    command = command_manager.get_command(self.command_name)
    if not isinstance(command, torch.Tensor):
      raise TypeError(
        f"Command '{self.command_name}' must be a torch.Tensor, "
        f"got {type(command).__name__}."
      )
    return command

  def _write_outputs(self) -> None:
    summary = self._build_summary()
    self._write_summary(summary)
    self._write_readme(summary)
    self._write_joint_timeseries_csv()
    self._write_policy_timeseries_csv()
    self._write_joint_npz()
    self._write_policy_npz()
    self._write_joint_plot(
      field="joint_pos",
      title="Joint Position",
      ylabel="rad or m",
      path=self.output_dir / "joint_positions.png",
      samples=self._physics_samples,
    )
    self._write_joint_plot(
      field="joint_pos_target",
      title="Joint Position Target",
      ylabel="rad or m",
      path=self.output_dir / "joint_position_targets.png",
      samples=self._physics_samples,
    )
    self._write_joint_plot(
      field="joint_vel",
      title="Joint Velocity",
      ylabel="rad/s or m/s",
      path=self.output_dir / "joint_velocities.png",
      samples=self._physics_samples,
    )
    self._write_joint_plot(
      field="joint_torque",
      title="Joint Torque",
      ylabel="N*m or N",
      path=self.output_dir / "joint_torques.png",
      samples=self._physics_samples,
    )
    self._write_joint_plot(
      field="joint_power",
      title="Joint Power",
      ylabel="W",
      path=self.output_dir / "joint_powers.png",
      samples=self._physics_samples,
    )
    self._write_joint_plot(
      field="policy_action",
      title="Policy Action",
      ylabel="raw action",
      path=self.output_dir / "policy_actions.png",
      samples=self._samples,
    )
    self._write_joint_plot(
      field="applied_action",
      title="Applied Action",
      ylabel="clipped action",
      path=self.output_dir / "applied_actions.png",
      samples=self._samples,
    )
    self._write_interactive_plot(
      field="joint_pos",
      title="Joint Position",
      ylabel="rad or m",
      path=self.output_dir / "joint_positions.html",
      samples=self._physics_samples,
    )
    self._write_interactive_plot(
      field="joint_pos_target",
      title="Joint Position Target",
      ylabel="rad or m",
      path=self.output_dir / "joint_position_targets.html",
      samples=self._physics_samples,
    )
    self._write_joint_position_compare_plot(samples=self._physics_samples)
    self._write_interactive_plot(
      field="joint_vel",
      title="Joint Velocity",
      ylabel="rad/s or m/s",
      path=self.output_dir / "joint_velocities.html",
      samples=self._physics_samples,
    )
    self._write_interactive_plot(
      field="joint_torque",
      title="Joint Torque",
      ylabel="N*m or N",
      path=self.output_dir / "joint_torques.html",
      samples=self._physics_samples,
    )
    self._write_interactive_plot(
      field="joint_power",
      title="Joint Power",
      ylabel="W",
      path=self.output_dir / "joint_powers.html",
      samples=self._physics_samples,
    )
    self._write_interactive_plot(
      field="policy_action",
      title="Policy Action",
      ylabel="raw action",
      path=self.output_dir / "policy_actions.html",
      samples=self._samples,
    )
    self._write_interactive_plot(
      field="applied_action",
      title="Applied Action",
      ylabel="clipped action",
      path=self.output_dir / "applied_actions.html",
      samples=self._samples,
    )

  def _build_summary(self) -> dict[str, Any]:
    entity = self._entity()
    if not self._samples:
      duration_s = 0.0
      done_count = 0
      reward_mean = 0.0
    else:
      duration_s = float(self._samples[-1]["time_s"])
      done_count = sum(1 for sample in self._samples if sample["done"])
      rewards = [
        sample["reward"] for sample in self._samples if sample["reward"] is not None
      ]
      reward_mean = float(np.mean(rewards)) if rewards else 0.0
    joint_duration_s = (
      float(self._physics_samples[-1]["time_s"]) if self._physics_samples else 0.0
    )
    policy_sample_dt_s = self.step_dt * self.interval

    return {
      "task_id": self.task_id,
      "entity_name": self.entity_name,
      "command_name": self.command_name,
      "command_velocity": {
        "vx": self.command_velocity[0],
        "vy": self.command_velocity[1],
        "wz": self.command_velocity[2],
      },
      "env_id": self.env_id,
      "requested_duration_s": self.duration_s,
      "actual_duration_s": duration_s,
      "joint_duration_s": joint_duration_s,
      "physics_dt": self.physics_dt,
      "physics_frequency_hz": 1.0 / self.physics_dt,
      "decimation": self.decimation,
      "step_dt": self.step_dt,
      "control_frequency_hz": 1.0 / self.step_dt,
      "sample_dt_s": policy_sample_dt_s,
      "sample_frequency_hz": 1.0 / policy_sample_dt_s,
      "policy_sample_dt_s": policy_sample_dt_s,
      "policy_sample_frequency_hz": 1.0 / policy_sample_dt_s,
      "joint_sample_dt_s": self.physics_dt,
      "joint_sample_frequency_hz": 1.0 / self.physics_dt,
      "num_policy_steps": self.num_steps,
      "sample_interval": self.interval,
      "num_samples": len(self._samples),
      "num_policy_samples": len(self._samples),
      "num_joint_samples": len(self._physics_samples),
      "done_count": int(done_count),
      "mean_reward": reward_mean,
      "joint_names": list(entity.joint_names),
      "units": {
        "policy_action": "raw policy output before action clipping",
        "applied_action": "policy output after RSL-RL wrapper action clipping",
        "joint_pos": "rad for hinge joints, m for slide joints",
        "joint_pos_target": (
          "policy-derived target position in rad for hinge joints, m for slide joints"
        ),
        "joint_vel": "rad/s for hinge joints, m/s for slide joints",
        "joint_torque": "qfrc_actuator in N*m for hinge joints, N for slide joints",
        "joint_power": "joint_torque * joint_vel in W",
      },
      "artifacts": {
        "csv": "joint_timeseries.csv",
        "npz": "joint_timeseries.npz",
        "joint_csv": "joint_timeseries.csv",
        "joint_npz": "joint_timeseries.npz",
        "policy_csv": "policy_timeseries.csv",
        "policy_npz": "policy_timeseries.npz",
        "joint_positions": "joint_positions.png",
        "joint_position_targets": "joint_position_targets.png",
        "joint_velocities": "joint_velocities.png",
        "joint_torques": "joint_torques.png",
        "joint_powers": "joint_powers.png",
        "policy_actions": "policy_actions.png",
        "applied_actions": "applied_actions.png",
        "joint_positions_html": "joint_positions.html",
        "joint_position_targets_html": "joint_position_targets.html",
        "joint_position_compare_html": "joint_position_compare.html",
        "joint_velocities_html": "joint_velocities.html",
        "joint_torques_html": "joint_torques.html",
        "joint_powers_html": "joint_powers.html",
        "policy_actions_html": "policy_actions.html",
        "applied_actions_html": "applied_actions.html",
        "readme": "README.md",
      },
    }

  def _write_summary(self, summary: dict[str, Any]) -> None:
    path = self.output_dir / "summary.json"
    path.write_text(
      json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
      encoding="utf-8",
    )

  def _write_readme(self, summary: dict[str, Any]) -> None:
    path = self.output_dir / "README.md"
    path.write_text(_readme_text(summary, self.checkpoint_file), encoding="utf-8")

  def _write_joint_timeseries_csv(self) -> None:
    path = self.output_dir / "joint_timeseries.csv"
    field_names = [
      "physics_step",
      "policy_step",
      "time_s",
      "command_vx",
      "command_vy",
      "command_wz",
    ]
    for field in (
      "joint_pos",
      "joint_pos_target",
      "joint_vel",
      "joint_torque",
      "joint_power",
    ):
      for name in self._series_names(field, self._physics_samples):
        field_names.append(f"{field}/{name}")

    with path.open("w", encoding="utf-8", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=field_names)
      writer.writeheader()
      for sample in self._physics_samples:
        row = {name: sample.get(name, "") for name in field_names}
        for field in (
          "joint_pos",
          "joint_pos_target",
          "joint_vel",
          "joint_torque",
          "joint_power",
        ):
          for idx, name in enumerate(self._series_names(field, self._physics_samples)):
            row[f"{field}/{name}"] = self._sample_series_value(sample, field, idx)
        writer.writerow(row)

  def _write_policy_timeseries_csv(self) -> None:
    path = self.output_dir / "policy_timeseries.csv"
    field_names = [
      "step",
      "time_s",
      "command_vx",
      "command_vy",
      "command_wz",
      "actual_vx",
      "actual_vy",
      "actual_wz",
      "reward",
      "done",
    ]
    for field in (
      "policy_action",
      "applied_action",
      "joint_pos",
      "joint_pos_target",
      "joint_vel",
      "joint_torque",
      "joint_power",
    ):
      for name in self._series_names(field, self._samples):
        field_names.append(f"{field}/{name}")

    with path.open("w", encoding="utf-8", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=field_names)
      writer.writeheader()
      for sample in self._samples:
        row = {name: sample.get(name, "") for name in field_names}
        for field in (
          "policy_action",
          "applied_action",
          "joint_pos",
          "joint_pos_target",
          "joint_vel",
          "joint_torque",
          "joint_power",
        ):
          for idx, name in enumerate(self._series_names(field, self._samples)):
            row[f"{field}/{name}"] = self._sample_series_value(sample, field, idx)
        writer.writerow(row)

  def _write_joint_npz(self) -> None:
    path = self.output_dir / "joint_timeseries.npz"
    arrays = self._arrays(self._physics_samples)
    np.savez(
      path,
      physics_step=np.asarray(
        [sample["physics_step"] for sample in self._physics_samples],
        dtype=np.int64,
      ),
      policy_step=np.asarray(
        [sample["policy_step"] for sample in self._physics_samples],
        dtype=np.int64,
      ),
      time_s=arrays["time_s"],
      command_velocity=arrays["command_velocity"],
      joint_pos=arrays["joint_pos"],
      joint_pos_target=arrays["joint_pos_target"],
      joint_vel=arrays["joint_vel"],
      joint_torque=arrays["joint_torque"],
      joint_power=arrays["joint_power"],
      joint_names=np.asarray(list(self._entity().joint_names), dtype=str),
    )

  def _write_policy_npz(self) -> None:
    path = self.output_dir / "policy_timeseries.npz"
    arrays = self._arrays(self._samples)
    np.savez(
      path,
      step=np.asarray([sample["step"] for sample in self._samples], dtype=np.int64),
      time_s=arrays["time_s"],
      command_velocity=arrays["command_velocity"],
      actual_velocity=arrays["actual_velocity"],
      policy_action=arrays["policy_action"],
      applied_action=arrays["applied_action"],
      joint_pos=arrays["joint_pos"],
      joint_pos_target=arrays["joint_pos_target"],
      joint_vel=arrays["joint_vel"],
      joint_torque=arrays["joint_torque"],
      joint_power=arrays["joint_power"],
      joint_names=np.asarray(list(self._entity().joint_names), dtype=str),
    )

  def _write_joint_plot(
    self,
    *,
    field: str,
    title: str,
    ylabel: str,
    path: Path,
    samples: list[dict[str, Any]],
  ) -> None:
    if not samples:
      return

    arrays = self._arrays(samples)
    values = arrays[field]
    if values.size == 0:
      return

    plt = _get_pyplot()
    time_s = arrays["time_s"]
    names = self._series_names(field, samples)
    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, name in enumerate(names):
      ax.plot(time_s, values[:, idx], linewidth=1.2, label=name)
    ax.set_title(
      f"{title} | cmd vx={self.command_velocity[0]:+.2f}, "
      f"vy={self.command_velocity[1]:+.2f}, wz={self.command_velocity[2]:+.2f}"
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if len(names) <= 16:
      ax.legend(loc="upper right", ncol=2)
    else:
      ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
      fig.subplots_adjust(right=0.78)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

  def _write_interactive_plot(
    self,
    *,
    field: str,
    title: str,
    ylabel: str,
    path: Path,
    samples: list[dict[str, Any]],
  ) -> None:
    if not samples:
      return

    arrays = self._arrays(samples)
    values = arrays[field]
    if values.size == 0:
      return

    payload = {
      "title": (
        f"{title} | cmd vx={self.command_velocity[0]:+.2f}, "
        f"vy={self.command_velocity[1]:+.2f}, wz={self.command_velocity[2]:+.2f}"
      ),
      "ylabel": ylabel,
      "time": arrays["time_s"].tolist(),
      "names": self._series_names(field, samples),
      "values": _json_safe_value(values.T.tolist()),
    }
    html = _interactive_plot_html(payload)
    path.write_text(html, encoding="utf-8")

  def _write_joint_position_compare_plot(
    self,
    *,
    samples: list[dict[str, Any]],
  ) -> None:
    if not samples:
      return

    arrays = self._arrays(samples)
    joint_pos = arrays["joint_pos"]
    joint_pos_target = arrays["joint_pos_target"]
    if joint_pos.size == 0 or joint_pos_target.size == 0:
      return

    width = min(joint_pos.shape[1], joint_pos_target.shape[1])
    if width == 0:
      return

    joint_names = self._series_names("joint_pos", samples)[:width]
    names: list[str] = []
    values: list[list[float]] = []
    for idx, joint_name in enumerate(joint_names):
      names.append(f"{joint_name} measured")
      values.append(joint_pos[:, idx].tolist())
      names.append(f"{joint_name} target")
      values.append(joint_pos_target[:, idx].tolist())

    payload = {
      "title": (
        "Joint Position vs Target | "
        f"cmd vx={self.command_velocity[0]:+.2f}, "
        f"vy={self.command_velocity[1]:+.2f}, wz={self.command_velocity[2]:+.2f}"
      ),
      "ylabel": "rad or m",
      "time": arrays["time_s"].tolist(),
      "names": names,
      "values": _json_safe_value(values),
      "initial_visible_count": min(8, len(names)),
    }
    html = _interactive_plot_html(payload)
    (self.output_dir / "joint_position_compare.html").write_text(
      html,
      encoding="utf-8",
    )

  def _arrays(self, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    actual_velocity = np.empty((len(samples), 0), dtype=float)
    if samples and "actual_vx" in samples[0]:
      actual_velocity = np.asarray(
        [
          [sample["actual_vx"], sample["actual_vy"], sample["actual_wz"]]
          for sample in samples
        ],
        dtype=float,
      )

    return {
      "time_s": np.asarray(
        [sample["time_s"] for sample in samples],
        dtype=float,
      ),
      "command_velocity": np.asarray(
        [
          [sample["command_vx"], sample["command_vy"], sample["command_wz"]]
          for sample in samples
        ],
        dtype=float,
      ),
      "actual_velocity": actual_velocity,
      "joint_pos": np.asarray(
        self._series_rows("joint_pos", samples),
        dtype=float,
      ),
      "joint_pos_target": np.asarray(
        self._series_rows("joint_pos_target", samples),
        dtype=float,
      ),
      "joint_vel": np.asarray(
        self._series_rows("joint_vel", samples),
        dtype=float,
      ),
      "joint_torque": np.asarray(
        self._series_rows("joint_torque", samples),
        dtype=float,
      ),
      "joint_power": np.asarray(
        self._series_rows("joint_power", samples),
        dtype=float,
      ),
      "policy_action": np.asarray(
        self._series_rows("policy_action", samples),
        dtype=float,
      ),
      "applied_action": np.asarray(
        self._series_rows("applied_action", samples),
        dtype=float,
      ),
    }

  def _series_names(self, field: str, samples: list[dict[str, Any]]) -> list[str]:
    width = self._series_width(field, samples)
    if width == 0:
      return []
    if field in {"policy_action", "applied_action"}:
      return [f"action_{idx}" for idx in range(width)]
    joint_names = list(self._entity().joint_names)
    if width == len(joint_names):
      return joint_names
    return [f"dof_{idx}" for idx in range(width)]

  def _series_width(self, field: str, samples: list[dict[str, Any]]) -> int:
    for sample in samples:
      values = sample.get(field)
      if values is not None:
        return len(values)
    return 0

  def _series_rows(
    self,
    field: str,
    samples: list[dict[str, Any]],
  ) -> list[list[float]]:
    width = self._series_width(field, samples)
    rows: list[list[float]] = []
    for sample in samples:
      values = sample.get(field)
      if values is None:
        rows.append([math.nan] * width)
      else:
        row = [float(value) for value in values]
        rows.append(row + [math.nan] * max(0, width - len(row)))
    return rows

  def _sample_series_value(self, sample: dict[str, Any], field: str, idx: int) -> Any:
    values = sample.get(field)
    if values is None or idx >= len(values):
      return ""
    return values[idx]


def _num_envs(env: Any) -> int:
  num_envs = getattr(env, "num_envs", None)
  if num_envs is None:
    num_envs = env.unwrapped.num_envs
  return int(num_envs)


def _physics_dt(env: Any, step_dt: float) -> float:
  value = getattr(env.unwrapped, "physics_dt", None)
  if value is not None:
    physics_dt = float(value)
    if physics_dt > 0.0:
      return physics_dt

  cfg = getattr(env.unwrapped, "cfg", None)
  sim_cfg = getattr(cfg, "sim", None)
  mujoco_cfg = getattr(sim_cfg, "mujoco", None)
  timestep = getattr(mujoco_cfg, "timestep", None)
  if timestep is not None:
    physics_dt = float(timestep)
    if physics_dt > 0.0:
      return physics_dt

  decimation = getattr(cfg, "decimation", None)
  if decimation is not None and int(decimation) > 0:
    return step_dt / int(decimation)

  return step_dt


def _decimation(env: Any, step_dt: float, physics_dt: float) -> int:
  cfg = getattr(env.unwrapped, "cfg", None)
  decimation = getattr(cfg, "decimation", None)
  if decimation is not None and int(decimation) > 0:
    return int(decimation)
  if physics_dt > 0.0:
    return max(1, int(round(step_dt / physics_dt)))
  return 1


def _env_tensor(value: torch.Tensor, env_id: int) -> list[float]:
  return value[env_id].detach().cpu().to(dtype=torch.float32).tolist()


def _optional_env_tensor(value: torch.Tensor | None, env_id: int) -> list[float] | None:
  if value is None:
    return None
  selected = value[env_id] if value.ndim > 1 else value
  cpu_value = selected.detach().cpu().to(dtype=torch.float32)
  if cpu_value.ndim == 0:
    return [float(cpu_value.item())]
  return cpu_value.tolist()


def _applied_action(env: Any, actions: torch.Tensor) -> torch.Tensor:
  clip_actions = getattr(env, "clip_actions", None)
  if clip_actions is None:
    return actions
  return torch.clamp(actions, -clip_actions, clip_actions)


def _json_safe_value(value: Any) -> Any:
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, np.floating):
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.ndarray):
    return _json_safe_value(value.tolist())
  if isinstance(value, list):
    return [_json_safe_value(item) for item in value]
  if isinstance(value, tuple):
    return [_json_safe_value(item) for item in value]
  if isinstance(value, dict):
    return {str(key): _json_safe_value(item) for key, item in value.items()}
  return value


def _optional_env_scalar(value: Any, env_id: int, num_envs: int) -> float | bool | None:
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    if value.ndim > 0 and value.shape[0] == num_envs:
      value = value[env_id]
    if value.numel() != 1:
      raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
    return value.item()
  if isinstance(value, np.ndarray):
    if value.ndim > 0 and value.shape[0] == num_envs:
      value = value[env_id]
    if value.size != 1:
      raise ValueError(f"Expected scalar ndarray, got shape {value.shape}")
    return value.item()
  if isinstance(value, (list, tuple)) and len(value) == num_envs:
    return _optional_env_scalar(value[env_id], env_id, num_envs)
  if isinstance(value, np.generic):
    value = value.item()
  if isinstance(value, bool):
    return value
  if isinstance(value, numbers.Real):
    return float(value)
  raise ValueError(f"Expected scalar value, got {type(value).__name__}.")


def _readme_text(summary: dict[str, Any], checkpoint_file: str | None) -> str:
  command = summary["command_velocity"]
  checkpoint_arg = (
    f" \\\n  --checkpoint-file '{checkpoint_file}'" if checkpoint_file else ""
  )
  example_command = (
    f"uv run play {summary['task_id']}{checkpoint_arg} \\\n"
    "  --command-eval True \\\n"
    f"  --command-eval-velocity '({command['vx']},{command['vy']},{command['wz']})' "
    "\\\n"
    f"  --command-eval-duration {summary['requested_duration_s']} \\\n"
    "  --no-terminations True"
  )
  return f"""# Play Command Evaluation Report

This directory was generated by `play --command-eval`.

## Command

```sh
{example_command}
```

`--command-eval-velocity` is `(vx, vy, wz)` in the robot body frame. The units are
meters per second for `vx` and `vy`, and radians per second for `wz`.

`--no-terminations True` is optional but recommended when you want one continuous
curve without episode resets.

## Sampling

- `physics_dt`: {summary["physics_dt"]} s
- `physics_frequency_hz`: {summary["physics_frequency_hz"]} Hz
- `decimation`: {summary["decimation"]} MuJoCo step(s) per policy step
- `step_dt`: {summary["step_dt"]} s
- `control_frequency_hz`: {summary["control_frequency_hz"]} Hz
- `sample_interval`: every {summary["sample_interval"]} policy step(s)
- `policy_sample_dt_s`: {summary["policy_sample_dt_s"]} s
- `policy_sample_frequency_hz`: {summary["policy_sample_frequency_hz"]} Hz
- `joint_sample_dt_s`: {summary["joint_sample_dt_s"]} s
- `joint_sample_frequency_hz`: {summary["joint_sample_frequency_hz"]} Hz

Joint state is sampled after each MuJoCo physics substep. Policy outputs are still
sampled only on policy/control steps. For the velocity tasks that use
`physics_dt = 0.002` and `decimation = 10`, joint curves are saved at 500 Hz while
policy actions are saved at 50 Hz.

## Output Files

- `summary.json`: metadata, command, sample frequency, units, and artifact names.
- `joint_timeseries.csv`: physics-rate joint position, target, velocity, torque,
  and power values in a flat table.
- `joint_timeseries.npz`: the same physics-rate joint data as NumPy arrays.
- `policy_timeseries.csv`: policy-rate command, actual velocity, reward, done,
  policy action, applied action, and policy-step joint snapshot values.
- `policy_timeseries.npz`: the same policy-rate data as NumPy arrays.
- `*.png`: static quick-look plots. Joint plots use physics-rate samples; action
  plots use policy-rate samples.
- `*.html`: interactive plots. Click legend rows to show/hide curves; double-click
  a legend row to solo one curve. Joint plots use physics-rate samples; action
  plots use policy-rate samples.
- `joint_position_compare.html`: measured joint position and target joint position
  in the same physics-rate interactive plot for easier tracking comparison.

## Data Fields

- `time_s`: elapsed evaluation time in seconds. In `joint_timeseries.*` this is
  physics-step time; in `policy_timeseries.*` this is policy-step time.
- `physics_step`: MuJoCo physics substep index in `joint_timeseries.*`.
- `policy_step`: policy/control step index associated with a physics sample.
- `command_vx`, `command_vy`, `command_wz`: fixed commanded body-frame velocity.
- `actual_vx`, `actual_vy`, `actual_wz`: measured robot body-frame velocity.
- `policy_action/action_i`: raw policy output before action clipping.
- `applied_action/action_i`: action after RSL-RL wrapper clipping, before the env
  action manager maps it to actuator targets.
- `joint_pos/<joint>`: measured joint position during play. This is simulation
  state feedback, not the policy target.
- `joint_pos_target/<joint>`: policy-derived target joint position written by the
  action/actuator pipeline.
- `joint_vel/<joint>`: measured joint velocity during play.
- `joint_torque/<joint>`: actuator generalized force in joint space
  (`qfrc_actuator`). For hinge joints this is N*m; for slide joints this is N.
- `joint_power/<joint>`: signed mechanical joint power, computed as
  `joint_torque * joint_vel`. For hinge and slide joints this is W. Positive
  values mean force/torque and velocity have the same sign; negative values mean
  the actuator is opposing the joint motion.
- `reward`: scalar environment reward for the selected `env_id`.
- `done`: whether the selected environment terminated or timed out on that step.

The first row at `time_s = 0` is the initial state before the first policy action,
so `policy_action/*` and `applied_action/*` are intentionally empty in CSV and
`NaN` in `policy_timeseries.npz` for that row.
"""


def _interactive_plot_html(payload: dict[str, Any]) -> str:
  data_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
  return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload["title"]}</title>
<style>
  :root {{
    color-scheme: light;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  body {{
    margin: 0;
    background: #f7f8fa;
    color: #1f2937;
  }}
  header {{
    padding: 16px 20px 8px;
  }}
  h1 {{
    margin: 0 0 6px;
    font-size: 20px;
    font-weight: 650;
  }}
  p {{
    margin: 0;
    color: #667085;
    font-size: 13px;
  }}
  main {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 280px;
    gap: 12px;
    height: calc(100vh - 78px);
    padding: 12px 20px 20px;
    box-sizing: border-box;
  }}
  .plot-panel,
  .legend-panel {{
    background: #ffffff;
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  }}
  .plot-panel {{
    min-height: 360px;
    padding: 10px;
  }}
  canvas {{
    display: block;
    width: 100%;
    height: 100%;
  }}
  .legend-panel {{
    display: flex;
    min-height: 0;
    flex-direction: column;
  }}
  .legend-actions {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    padding: 10px;
    border-bottom: 1px solid #eaecf0;
  }}
  button {{
    border: 1px solid #cfd4dc;
    border-radius: 6px;
    background: #ffffff;
    color: #344054;
    font: inherit;
    font-size: 12px;
    padding: 6px 8px;
    cursor: pointer;
  }}
  button:hover {{
    background: #f2f4f7;
  }}
  #legend {{
    overflow: auto;
    padding: 6px;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 28px;
    padding: 4px 6px;
    border-radius: 6px;
    cursor: pointer;
    color: #98a2b3;
    font-size: 13px;
    user-select: none;
  }}
  .legend-item:hover {{
    background: #f2f4f7;
  }}
  .legend-item.active {{
    color: #1f2937;
    font-weight: 600;
  }}
  .swatch {{
    width: 18px;
    height: 3px;
    border-radius: 999px;
    flex: 0 0 auto;
    opacity: 0.35;
  }}
  .legend-item.active .swatch {{
    opacity: 1;
  }}
  @media (max-width: 900px) {{
    main {{
      grid-template-columns: 1fr;
      height: auto;
    }}
    .plot-panel {{
      height: 62vh;
    }}
    .legend-panel {{
      max-height: 36vh;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <p>Click legend rows to show or hide curves. Double-click a row to solo it.</p>
</header>
<main>
  <section class="plot-panel">
    <canvas id="plot"></canvas>
  </section>
  <aside class="legend-panel">
    <div class="legend-actions">
      <button id="showAll">Show all</button>
      <button id="showFirst">First 8</button>
      <button id="clearAll">Clear</button>
    </div>
    <div id="legend"></div>
  </aside>
</main>
<script id="plot-data" type="application/json">{data_json}</script>
<script>
const data = JSON.parse(document.getElementById("plot-data").textContent);
const palette = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#4e79a7", "#f28e2b",
  "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7",
  "#9c755f", "#bab0ab"
];
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
const legend = document.getElementById("legend");
const title = document.getElementById("title");
const initialVisibleCount = data.initial_visible_count ?? Math.min(8, data.names.length);
const visible = data.names.map((_, i) => i < Math.min(initialVisibleCount, data.names.length));
title.textContent = data.title;

function colorFor(index) {{
  return palette[index % palette.length];
}}

function setVisible(mode) {{
  visible.fill(false);
  if (mode === "all") {{
    visible.fill(true);
  }} else if (mode === "first") {{
    for (let i = 0; i < Math.min(8, visible.length); i += 1) {{
      visible[i] = true;
    }}
  }}
  renderLegend();
  draw();
}}

function renderLegend() {{
  legend.innerHTML = "";
  data.names.forEach((name, index) => {{
    const row = document.createElement("div");
    row.className = "legend-item" + (visible[index] ? " active" : "");
    row.title = "Click to toggle. Double-click to solo.";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = colorFor(index);
    const label = document.createElement("span");
    label.textContent = name;
    row.append(swatch, label);
    row.addEventListener("click", () => {{
      visible[index] = !visible[index];
      renderLegend();
      draw();
    }});
    row.addEventListener("dblclick", () => {{
      visible.fill(false);
      visible[index] = true;
      renderLegend();
      draw();
    }});
    legend.appendChild(row);
  }});
}}

function resizeCanvas() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function drawText(text, x, y, align = "left") {{
  ctx.fillStyle = "#475467";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = align;
  ctx.fillText(text, x, y);
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const left = 72;
  const right = 20;
  const top = 24;
  const bottom = 48;
  const plotW = Math.max(1, width - left - right);
  const plotH = Math.max(1, height - top - bottom);
  const active = visible.map((v, i) => v ? i : -1).filter(i => i >= 0);
  const tMin = data.time[0] ?? 0;
  const tMax = data.time[data.time.length - 1] ?? 1;
  const xSpan = Math.max(1e-9, tMax - tMin);

  if (active.length === 0) {{
    drawText("Select one or more legend rows to draw curves.", left + 12, top + 24);
    return;
  }}

  let yMin = Infinity;
  let yMax = -Infinity;
  active.forEach(seriesIdx => {{
    data.values[seriesIdx].forEach(v => {{
      if (Number.isFinite(v)) {{
        yMin = Math.min(yMin, v);
        yMax = Math.max(yMax, v);
      }}
    }});
  }});
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {{
    yMin = -1;
    yMax = 1;
  }}
  if (Math.abs(yMax - yMin) < 1e-9) {{
    yMin -= 1;
    yMax += 1;
  }}
  const pad = 0.05 * (yMax - yMin);
  yMin -= pad;
  yMax += pad;
  const ySpan = yMax - yMin;

  const x = t => left + ((t - tMin) / xSpan) * plotW;
  const y = v => top + (1 - ((v - yMin) / ySpan)) * plotH;

  ctx.strokeStyle = "#e4e7ec";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {{
    const gx = left + (i / 5) * plotW;
    const gy = top + (i / 5) * plotH;
    ctx.beginPath();
    ctx.moveTo(gx, top);
    ctx.lineTo(gx, top + plotH);
    ctx.moveTo(left, gy);
    ctx.lineTo(left + plotW, gy);
    ctx.stroke();
    drawText((tMin + (i / 5) * xSpan).toFixed(2), gx, top + plotH + 22, "center");
    drawText((yMax - (i / 5) * ySpan).toPrecision(4), left - 8, gy + 4, "right");
  }}

  ctx.strokeStyle = "#667085";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.lineTo(left + plotW, top + plotH);
  ctx.stroke();
  drawText("time [s]", left + plotW / 2, height - 12, "center");
  ctx.save();
  ctx.translate(16, top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  drawText(data.ylabel, 0, 0, "center");
  ctx.restore();

  active.forEach(seriesIdx => {{
    const series = data.values[seriesIdx];
    ctx.strokeStyle = colorFor(seriesIdx);
    ctx.lineWidth = active.length === 1 ? 2.4 : 1.35;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < data.time.length; i += 1) {{
      const value = series[i];
      if (!Number.isFinite(value)) {{
        started = false;
        continue;
      }}
      const px = x(data.time[i]);
      const py = y(value);
      if (!started) {{
        ctx.moveTo(px, py);
        started = true;
      }} else {{
        ctx.lineTo(px, py);
      }}
    }}
    ctx.stroke();
  }});
}}

document.getElementById("showAll").addEventListener("click", () => setVisible("all"));
document.getElementById("showFirst").addEventListener("click", () => setVisible("first"));
document.getElementById("clearAll").addEventListener("click", () => setVisible("none"));
window.addEventListener("resize", resizeCanvas);
renderLegend();
resizeCanvas();
</script>
</body>
</html>
"""
