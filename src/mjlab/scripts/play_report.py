"""Play-time evaluation report generation for velocity tasks."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mjlab.utils.lab_api.math import euler_xyz_from_quat


def _get_pyplot():
  mplconfigdir = Path(tempfile.gettempdir()) / "matplotlib"
  mplconfigdir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  return plt


def default_play_report_dir(
  task_id: str,
  *,
  log_dir: Path | None,
  log_root: str | Path,
) -> Path:
  """Return the default output directory for a play evaluation report."""
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  safe_task_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_id)
  if log_dir is not None:
    base_dir = Path(log_dir) / "play_reports"
  else:
    base_dir = Path(log_root).resolve() / "play_reports" / safe_task_id
  return base_dir / f"play_report_{timestamp}"


class PlayReportRecorder:
  """Collects play-time evaluation metrics and writes summary artifacts."""

  def __init__(
    self,
    *,
    env: Any,
    output_dir: str | Path,
    task_id: str,
    entity_name: str = "robot",
    command_name: str = "twist",
    env_id: int = 0,
    interval: int = 1,
  ) -> None:
    if interval < 1:
      raise ValueError(f"report interval must be >= 1, got {interval}")

    self.env = env
    self.output_dir = Path(output_dir).expanduser()
    self.task_id = task_id
    self.entity_name = entity_name
    self.command_name = command_name
    self.env_id = int(env_id)
    self.interval = int(interval)
    self.step_count = 0
    self._closed = False
    self._samples: list[dict[str, Any]] = []

    num_envs = _num_envs(env)
    if self.env_id < 0 or self.env_id >= num_envs:
      raise ValueError(
        f"report_env_id={self.env_id} is out of range for num_envs={num_envs}"
      )

    # Resolve these once so invalid names fail before the viewer starts.
    self._entity()
    command = self._command()
    if command.ndim != 2 or command.shape[1] < 3:
      raise ValueError(
        f"Command '{self.command_name}' must have shape (num_envs, >=3), "
        f"got {tuple(command.shape)}"
      )

  @property
  def should_record(self) -> bool:
    return self.step_count % self.interval == 0

  def capture_command(self) -> torch.Tensor:
    """Capture the current command before the environment step."""
    return self._command().detach().clone()

  def write_step(
    self,
    *,
    pre_command: torch.Tensor,
    step_result: tuple[Any, ...],
  ) -> None:
    """Record one sampled play step."""
    entity = self._entity()
    data = entity.data
    step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.0))

    lin_vel = data.root_link_lin_vel_b[self.env_id].detach().cpu()
    ang_vel = data.root_link_ang_vel_b[self.env_id].detach().cpu()
    quat = data.root_link_quat_w[self.env_id : self.env_id + 1].detach().cpu()
    roll, pitch, _ = euler_xyz_from_quat(quat)

    command = pre_command[self.env_id].detach().cpu()
    reward, done = self._reward_done(step_result)
    reward_terms = self._reward_terms()
    metrics_terms = self._metrics_terms()

    sample = {
      "step": self.step_count,
      "time_s": (self.step_count + 1) * step_dt,
      "command_vx": float(command[0].item()),
      "command_vy": float(command[1].item()),
      "command_wz": float(command[2].item()),
      "actual_vx": float(lin_vel[0].item()),
      "actual_vy": float(lin_vel[1].item()),
      "actual_wz": float(ang_vel[2].item()),
      "roll_rad": float(roll[0].item()),
      "pitch_rad": float(pitch[0].item()),
      "reward": float(reward),
      "done": bool(done),
      "reward_terms": reward_terms,
      "metrics": metrics_terms,
    }
    self._samples.append(sample)

  def advance_step(self) -> None:
    self.step_count += 1

  def close(self) -> None:
    if self._closed:
      return
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self._write_outputs()
    self._closed = True

  def _entity(self) -> Any:
    scene = self.env.unwrapped.scene
    try:
      return scene[self.entity_name]
    except KeyError as err:
      available = ", ".join(scene.entities.keys())
      raise KeyError(
        f"Report entity '{self.entity_name}' was not found. "
        f"Available entities: {available}"
      ) from err

  def _command(self) -> torch.Tensor:
    command_manager = getattr(self.env.unwrapped, "command_manager", None)
    if command_manager is None:
      raise RuntimeError("Play report requires an environment with a command manager.")
    return command_manager.get_command(self.command_name)

  def _reward_done(self, step_result: tuple[Any, ...]) -> tuple[float, bool]:
    if len(step_result) < 3:
      return 0.0, False
    reward = _slice_env_value(step_result[1], self.env_id, _num_envs(self.env))
    done = _slice_env_value(step_result[2], self.env_id, _num_envs(self.env))
    return float(_scalar_value(reward)), bool(_scalar_value(done))

  def _reward_terms(self) -> dict[str, float]:
    reward_manager = getattr(self.env.unwrapped, "reward_manager", None)
    if reward_manager is None or not hasattr(
      reward_manager, "get_active_iterable_terms"
    ):
      return {}
    return _iterable_terms_to_dict(
      reward_manager.get_active_iterable_terms(self.env_id)
    )

  def _metrics_terms(self) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics_manager = getattr(self.env.unwrapped, "metrics_manager", None)
    if metrics_manager is not None and hasattr(
      metrics_manager, "get_active_iterable_terms"
    ):
      metrics.update(
        _iterable_terms_to_dict(metrics_manager.get_active_iterable_terms(self.env_id))
      )
    if "mean_action_acc" not in metrics:
      action_manager = getattr(self.env.unwrapped, "action_manager", None)
      if action_manager is not None:
        action_acc = (
          action_manager.action
          - 2 * action_manager.prev_action
          + action_manager.prev_prev_action
        )
        metrics["mean_action_acc"] = float(
          torch.mean(torch.abs(action_acc[self.env_id])).item()
        )
    return metrics

  def _write_outputs(self) -> None:
    summary = self._build_summary()
    self._write_summary(summary)
    self._write_timeseries_csv()
    self._write_overview_plot(summary)
    self._write_reward_terms_plot(summary)

  def _build_summary(self) -> dict[str, Any]:
    if not self._samples:
      return {
        "task_id": self.task_id,
        "entity_name": self.entity_name,
        "command_name": self.command_name,
        "env_id": self.env_id,
        "step_dt": float(getattr(self.env.unwrapped, "step_dt", 0.0)),
        "sample_interval": self.interval,
        "num_steps": 0,
        "duration_s": 0.0,
        "done_count": 0,
        "tracking": {},
        "stability": {},
        "control": {},
        "reward": {},
        "reward_term_means": {},
        "metric_means": {},
      }

    arrays = self._to_numpy()
    done_mask = arrays["done"].astype(bool)
    valid_mask = ~done_mask
    if not np.any(valid_mask):
      valid_mask = np.ones_like(done_mask, dtype=bool)

    err_vx = arrays["command_vx"] - arrays["actual_vx"]
    err_vy = arrays["command_vy"] - arrays["actual_vy"]
    err_wz = arrays["command_wz"] - arrays["actual_wz"]
    err_lin_xy = np.sqrt(np.square(err_vx) + np.square(err_vy))

    reward_term_means = {
      name: _nanmean(values) for name, values in arrays["reward_terms"].items()
    }
    metric_means = {
      name: _nanmean(values) for name, values in arrays["metrics"].items()
    }

    return {
      "task_id": self.task_id,
      "entity_name": self.entity_name,
      "command_name": self.command_name,
      "env_id": self.env_id,
      "step_dt": float(getattr(self.env.unwrapped, "step_dt", 0.0)),
      "sample_interval": self.interval,
      "num_steps": int(len(self._samples)),
      "duration_s": float(arrays["time_s"][-1]),
      "done_count": int(np.sum(done_mask)),
      "tracking": {
        "rmse_vx": _rmse(err_vx[valid_mask]),
        "rmse_vy": _rmse(err_vy[valid_mask]),
        "rmse_wz": _rmse(err_wz[valid_mask]),
        "mae_vx": _mae(err_vx[valid_mask]),
        "mae_vy": _mae(err_vy[valid_mask]),
        "mae_wz": _mae(err_wz[valid_mask]),
        "rmse_linear_xy": _rmse(err_lin_xy[valid_mask]),
        "mae_linear_xy": _mae(err_lin_xy[valid_mask]),
        "peak_linear_xy_error": _nanmax(err_lin_xy[valid_mask]),
      },
      "stability": {
        "mean_abs_roll_deg": _rad_to_deg(_mae(arrays["roll_rad"][valid_mask])),
        "mean_abs_pitch_deg": _rad_to_deg(_mae(arrays["pitch_rad"][valid_mask])),
        "max_abs_roll_deg": _rad_to_deg(
          _nanmax(np.abs(arrays["roll_rad"][valid_mask]))
        ),
        "max_abs_pitch_deg": _rad_to_deg(
          _nanmax(np.abs(arrays["pitch_rad"][valid_mask]))
        ),
      },
      "control": {
        "mean_action_acc": float(metric_means.get("mean_action_acc", 0.0)),
      },
      "reward": {
        "mean_total_reward": _nanmean(arrays["reward"]),
        "min_total_reward": _nanmin(arrays["reward"]),
        "max_total_reward": _nanmax(arrays["reward"]),
      },
      "reward_term_means": reward_term_means,
      "metric_means": metric_means,
    }

  def _to_numpy(self) -> dict[str, Any]:
    reward_term_names = sorted(
      {name for sample in self._samples for name in sample["reward_terms"]}
    )
    metric_names = sorted(
      {name for sample in self._samples for name in sample["metrics"]}
    )

    arrays: dict[str, Any] = {
      "time_s": np.asarray([sample["time_s"] for sample in self._samples], dtype=float),
      "command_vx": np.asarray(
        [sample["command_vx"] for sample in self._samples], dtype=float
      ),
      "command_vy": np.asarray(
        [sample["command_vy"] for sample in self._samples], dtype=float
      ),
      "command_wz": np.asarray(
        [sample["command_wz"] for sample in self._samples], dtype=float
      ),
      "actual_vx": np.asarray(
        [sample["actual_vx"] for sample in self._samples], dtype=float
      ),
      "actual_vy": np.asarray(
        [sample["actual_vy"] for sample in self._samples], dtype=float
      ),
      "actual_wz": np.asarray(
        [sample["actual_wz"] for sample in self._samples], dtype=float
      ),
      "roll_rad": np.asarray(
        [sample["roll_rad"] for sample in self._samples], dtype=float
      ),
      "pitch_rad": np.asarray(
        [sample["pitch_rad"] for sample in self._samples], dtype=float
      ),
      "reward": np.asarray([sample["reward"] for sample in self._samples], dtype=float),
      "done": np.asarray([sample["done"] for sample in self._samples], dtype=bool),
      "reward_terms": {
        name: np.asarray(
          [sample["reward_terms"].get(name, np.nan) for sample in self._samples],
          dtype=float,
        )
        for name in reward_term_names
      },
      "metrics": {
        name: np.asarray(
          [sample["metrics"].get(name, np.nan) for sample in self._samples],
          dtype=float,
        )
        for name in metric_names
      },
    }
    return arrays

  def _write_summary(self, summary: dict[str, Any]) -> None:
    summary_path = self.output_dir / "summary.json"
    summary_path.write_text(
      json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
      encoding="utf-8",
    )

  def _write_timeseries_csv(self) -> None:
    csv_path = self.output_dir / "timeseries.csv"
    reward_term_names = sorted(
      {name for sample in self._samples for name in sample["reward_terms"]}
    )
    metric_names = sorted(
      {name for sample in self._samples for name in sample["metrics"]}
    )
    fieldnames = [
      "step",
      "time_s",
      "command_vx",
      "command_vy",
      "command_wz",
      "actual_vx",
      "actual_vy",
      "actual_wz",
      "roll_rad",
      "pitch_rad",
      "reward",
      "done",
      *[f"reward_term_{name}" for name in reward_term_names],
      *[f"metric_{name}" for name in metric_names],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
      writer = csv.DictWriter(file, fieldnames=fieldnames)
      writer.writeheader()
      for sample in self._samples:
        row = {
          "step": sample["step"],
          "time_s": sample["time_s"],
          "command_vx": sample["command_vx"],
          "command_vy": sample["command_vy"],
          "command_wz": sample["command_wz"],
          "actual_vx": sample["actual_vx"],
          "actual_vy": sample["actual_vy"],
          "actual_wz": sample["actual_wz"],
          "roll_rad": sample["roll_rad"],
          "pitch_rad": sample["pitch_rad"],
          "reward": sample["reward"],
          "done": sample["done"],
        }
        for name in reward_term_names:
          row[f"reward_term_{name}"] = sample["reward_terms"].get(name, "")
        for name in metric_names:
          row[f"metric_{name}"] = sample["metrics"].get(name, "")
        writer.writerow(row)

  def _write_overview_plot(self, summary: dict[str, Any]) -> None:
    if not self._samples:
      return

    plt = _get_pyplot()
    arrays = self._to_numpy()
    time_s = arrays["time_s"]
    done_times = time_s[arrays["done"]]
    err_vx = arrays["command_vx"] - arrays["actual_vx"]
    err_vy = arrays["command_vy"] - arrays["actual_vy"]
    err_wz = arrays["command_wz"] - arrays["actual_wz"]
    err_lin_xy = np.sqrt(np.square(err_vx) + np.square(err_vy))

    actual_vx = _mask_done_steps(arrays["actual_vx"], arrays["done"])
    actual_vy = _mask_done_steps(arrays["actual_vy"], arrays["done"])
    actual_wz = _mask_done_steps(arrays["actual_wz"], arrays["done"])
    roll_deg = _mask_done_steps(np.rad2deg(arrays["roll_rad"]), arrays["done"])
    pitch_deg = _mask_done_steps(np.rad2deg(arrays["pitch_rad"]), arrays["done"])
    reward = _mask_done_steps(arrays["reward"], arrays["done"])
    mean_action_acc = _mask_done_steps(
      arrays["metrics"].get("mean_action_acc", np.full_like(time_s, np.nan)),
      arrays["done"],
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(
      f"{self.task_id} play report | env {self.env_id} | command '{self.command_name}'",
      fontsize=14,
    )

    ax = axes[0, 0]
    ax.plot(time_s, arrays["command_vx"], label="cmd vx", linewidth=2)
    ax.plot(time_s, actual_vx, label="actual vx", linewidth=1.8)
    ax.set_ylabel("m/s")
    ax.set_title("Forward Velocity Tracking")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(time_s, arrays["command_vy"], label="cmd vy", linewidth=2)
    ax.plot(time_s, actual_vy, label="actual vy", linewidth=1.8)
    ax.set_ylabel("m/s")
    ax.set_title("Lateral Velocity Tracking")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.plot(time_s, arrays["command_wz"], label="cmd wz", linewidth=2)
    ax.plot(time_s, actual_wz, label="actual wz", linewidth=1.8)
    ax.set_ylabel("rad/s")
    ax.set_title("Yaw-Rate Tracking")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.plot(time_s, np.abs(err_vx), label="|vx error|", linewidth=1.4)
    ax.plot(time_s, np.abs(err_vy), label="|vy error|", linewidth=1.4)
    ax.plot(time_s, np.abs(err_wz), label="|wz error|", linewidth=1.4)
    ax.plot(time_s, err_lin_xy, label="xy error norm", linewidth=2.2)
    ax.set_ylabel("error")
    ax.set_title("Velocity Tracking Error")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[2, 0]
    ax.plot(time_s, roll_deg, label="roll", linewidth=1.8)
    ax.plot(time_s, pitch_deg, label="pitch", linewidth=1.8)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deg")
    ax.set_title("Base Attitude")
    ax.legend(loc="upper right")

    ax = axes[2, 1]
    ax.plot(time_s, reward, label="reward", linewidth=1.8)
    if not np.all(np.isnan(mean_action_acc)):
      ax2 = ax.twinx()
      ax2.plot(
        time_s,
        mean_action_acc,
        label="mean_action_acc",
        color="tab:orange",
        linewidth=1.6,
      )
      ax2.set_ylabel("action acc")
      lines, labels = ax.get_legend_handles_labels()
      lines2, labels2 = ax2.get_legend_handles_labels()
      ax.legend(lines + lines2, labels + labels2, loc="upper right")
    else:
      ax.legend(loc="upper right")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reward")
    ax.set_title("Reward And Action Smoothness")

    summary_lines = [
      f"Duration: {summary['duration_s']:.2f} s",
      f"Samples: {summary['num_steps']}",
      f"Done count: {summary['done_count']}",
      "",
      "Tracking:",
      f"  RMSE vx: {summary['tracking'].get('rmse_vx', 0.0):.3f}",
      f"  RMSE vy: {summary['tracking'].get('rmse_vy', 0.0):.3f}",
      f"  RMSE wz: {summary['tracking'].get('rmse_wz', 0.0):.3f}",
      f"  MAE xy:  {summary['tracking'].get('mae_linear_xy', 0.0):.3f}",
      "",
      "Stability:",
      f"  Mean |roll|:  {summary['stability'].get('mean_abs_roll_deg', 0.0):.2f} deg",
      f"  Mean |pitch|: {summary['stability'].get('mean_abs_pitch_deg', 0.0):.2f} deg",
      "",
      "Control:",
      f"  Mean action acc: {summary['control'].get('mean_action_acc', 0.0):.4f}",
      f"  Mean reward:     {summary['reward'].get('mean_total_reward', 0.0):.4f}",
    ]
    fig.text(
      0.76,
      0.50,
      "\n".join(summary_lines),
      va="top",
      ha="left",
      family="monospace",
      fontsize=10,
      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    for ax_row in axes:
      for ax in ax_row:
        _mark_done_steps(ax, done_times)
        ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.96))
    fig.savefig(self.output_dir / "overview.png", dpi=180)
    plt.close(fig)

  def _write_reward_terms_plot(self, summary: dict[str, Any]) -> None:
    reward_term_means = summary.get("reward_term_means", {})
    if not reward_term_means:
      return

    plt = _get_pyplot()
    items = sorted(
      reward_term_means.items(),
      key=lambda item: abs(item[1]),
      reverse=True,
    )
    names = [name for name, _ in items[:10]]
    values = np.asarray([value for _, value in items[:10]], dtype=float)
    colors = ["tab:green" if value >= 0.0 else "tab:red" for value in values]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.55 * len(names) + 1.5)))
    ypos = np.arange(len(names))
    ax.barh(ypos, values, color=colors, alpha=0.85)
    ax.set_yticks(ypos, labels=names)
    ax.invert_yaxis()
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("mean reward rate (weighted, before dt scaling)")
    ax.set_title("Average Reward-Term Contribution During Play")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(self.output_dir / "reward_terms.png", dpi=180)
    plt.close(fig)


class PlayReportEnvWrapper:
  """Environment wrapper that records metrics and writes play reports."""

  def __init__(
    self,
    env: Any,
    *,
    output_dir: str | Path,
    task_id: str,
    entity_name: str = "robot",
    command_name: str = "twist",
    env_id: int = 0,
    interval: int = 1,
  ) -> None:
    self.env = env
    self.num_envs: int = int(env.num_envs)
    self.recorder = PlayReportRecorder(
      env=env,
      output_dir=output_dir,
      task_id=task_id,
      entity_name=entity_name,
      command_name=command_name,
      env_id=env_id,
      interval=interval,
    )
    self._closed = False

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
    return self.env.get_observations()

  def reset(self, *args: Any, **kwargs: Any) -> Any:
    return self.env.reset(*args, **kwargs)

  def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
    pre_command = None
    if self.recorder.should_record:
      pre_command = self.recorder.capture_command()
    step_result = self.env.step(actions)
    if pre_command is not None:
      self.recorder.write_step(pre_command=pre_command, step_result=step_result)
    self.recorder.advance_step()
    return step_result

  def close(self) -> None:
    if self._closed:
      return
    try:
      self.recorder.close()
    finally:
      self._closed = True
      self.env.close()


def _iterable_terms_to_dict(
  terms: list[tuple[str, list[float]]] | tuple[tuple[str, list[float]], ...] | Any,
) -> dict[str, float]:
  result: dict[str, float] = {}
  for name, values in terms:
    if not values:
      continue
    result[str(name)] = float(values[0])
  return result


def _num_envs(env: Any) -> int:
  num_envs = getattr(env, "num_envs", None)
  if num_envs is None:
    num_envs = env.unwrapped.num_envs
  return int(num_envs)


def _slice_env_value(value: Any, env_id: int, num_envs: int) -> Any:
  if isinstance(value, torch.Tensor):
    if value.ndim > 0 and value.shape[0] == num_envs:
      return value[env_id]
    return value
  if isinstance(value, np.ndarray):
    if value.ndim > 0 and value.shape[0] == num_envs:
      return value[env_id]
    return value
  if isinstance(value, (list, tuple)) and len(value) == num_envs:
    return value[env_id]
  return value


def _scalar_value(value: Any) -> float | bool:
  if isinstance(value, torch.Tensor):
    if value.numel() != 1:
      raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
    return value.item()
  if isinstance(value, np.ndarray):
    if value.size != 1:
      raise ValueError(f"Expected scalar ndarray, got shape {value.shape}")
    return value.item()
  if isinstance(value, np.generic):
    return value.item()
  return value


def _rmse(values: np.ndarray) -> float:
  if values.size == 0:
    return 0.0
  return float(np.sqrt(np.nanmean(np.square(values))))


def _mae(values: np.ndarray) -> float:
  if values.size == 0:
    return 0.0
  return float(np.nanmean(np.abs(values)))


def _nanmean(values: np.ndarray) -> float:
  if values.size == 0 or np.all(np.isnan(values)):
    return 0.0
  return float(np.nanmean(values))


def _nanmax(values: np.ndarray) -> float:
  if values.size == 0 or np.all(np.isnan(values)):
    return 0.0
  return float(np.nanmax(values))


def _nanmin(values: np.ndarray) -> float:
  if values.size == 0 or np.all(np.isnan(values)):
    return 0.0
  return float(np.nanmin(values))


def _rad_to_deg(value: float) -> float:
  return float(np.rad2deg(value))


def _mask_done_steps(values: np.ndarray, done: np.ndarray) -> np.ndarray:
  masked = values.astype(float).copy()
  masked[done.astype(bool)] = np.nan
  return masked


def _mark_done_steps(ax: Any, done_times: np.ndarray) -> None:
  for time_s in done_times:
    ax.axvline(float(time_s), color="tab:red", linestyle="--", linewidth=1, alpha=0.2)
