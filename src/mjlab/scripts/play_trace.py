"""Per-step trace logging for the play script."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

_STATE_FIELDS = (
  "joint_pos",
  "joint_vel",
  "joint_pos_target",
  "joint_vel_target",
  "joint_effort_target",
  "actuator_force",
  "qfrc_actuator",
  "ctrl",
)


def default_play_trace_path(
  task_id: str,
  *,
  log_dir: Path | None,
  log_root: str | Path,
) -> Path:
  """Return the default JSONL trace path for a play session."""
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  safe_task_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in task_id)
  if log_dir is not None:
    base_dir = Path(log_dir) / "play_traces"
  else:
    base_dir = Path(log_root).resolve() / "play_traces" / safe_task_id
  return base_dir / f"play_trace_{timestamp}.jsonl"


class PlayTraceRecorder:
  """Writes play-step observations, actions, and actuator state as JSONL."""

  def __init__(
    self,
    *,
    env: Any,
    path: str | Path,
    task_id: str,
    entity_name: str = "robot",
    env_ids: Sequence[int] = (0,),
    interval: int = 1,
  ) -> None:
    if interval < 1:
      raise ValueError(f"trace interval must be >= 1, got {interval}")

    self.env = env
    self.path = Path(path).expanduser()
    self.task_id = task_id
    self.entity_name = entity_name
    self.env_ids = tuple(int(env_id) for env_id in env_ids)
    self.interval = int(interval)
    self.step_count = 0
    self._closed = False
    self._rows_since_flush = 0

    if not self.env_ids:
      raise ValueError("trace_env_ids must contain at least one environment id")

    num_envs = _num_envs(env)
    invalid_env_ids = [
      env_id for env_id in self.env_ids if env_id < 0 or env_id >= num_envs
    ]
    if invalid_env_ids:
      raise ValueError(
        f"trace_env_ids {invalid_env_ids} are out of range for num_envs={num_envs}"
      )

    # Resolve the entity now so an invalid entity name fails before the viewer starts.
    self._entity()

    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._file = self.path.open("w", encoding="utf-8")
    self._write_metadata(num_envs)

  @property
  def should_record(self) -> bool:
    return self.step_count % self.interval == 0

  def capture_state(self) -> dict[str, torch.Tensor]:
    """Capture the selected entity's current joint and actuator state."""
    entity_data = self._entity().data
    state = {
      "joint_pos": entity_data.joint_pos,
      "joint_vel": entity_data.joint_vel,
      "joint_pos_target": entity_data.joint_pos_target,
      "joint_vel_target": entity_data.joint_vel_target,
      "joint_effort_target": entity_data.joint_effort_target,
      "actuator_force": entity_data.actuator_force,
      "qfrc_actuator": entity_data.qfrc_actuator,
    }
    ctrl_ids = entity_data.indexing.ctrl_ids
    state["ctrl"] = entity_data.data.ctrl[:, ctrl_ids]
    return {name: value.detach().clone() for name, value in state.items()}

  def write_step(
    self,
    *,
    obs: Any,
    policy_output: torch.Tensor,
    applied_action: torch.Tensor,
    pre_state: dict[str, torch.Tensor],
    post_state: dict[str, torch.Tensor],
    step_result: tuple[Any, ...],
  ) -> None:
    """Write one JSONL row per selected environment for the current play step."""
    reward, done = self._reward_done(step_result)
    num_envs = _num_envs(self.env)
    step_dt = float(getattr(self.env.unwrapped, "step_dt", 0.0))

    for env_id in self.env_ids:
      row = {
        "type": "step",
        "step": self.step_count,
        "time_s": self.step_count * step_dt,
        "env_id": env_id,
        "policy_input": _json_from_batch(obs, env_id, num_envs),
        "policy_output": _json_from_batch(policy_output, env_id, num_envs),
        "applied_action": _json_from_batch(applied_action, env_id, num_envs),
        "pre": _json_from_batch(pre_state, env_id, num_envs),
        "post": _json_from_batch(post_state, env_id, num_envs),
        "reward": _json_from_batch(reward, env_id, num_envs),
        "done": bool(_json_from_batch(done, env_id, num_envs)),
      }
      self._write_jsonl(row)

  def advance_step(self) -> None:
    self.step_count += 1

  def close(self) -> None:
    if self._closed:
      return
    self._file.flush()
    self._file.close()
    self._closed = True

  def _entity(self) -> Any:
    scene = self.env.unwrapped.scene
    try:
      return scene[self.entity_name]
    except KeyError as err:
      available = ", ".join(scene.entities.keys())
      raise KeyError(
        f"Trace entity '{self.entity_name}' was not found. "
        f"Available entities: {available}"
      ) from err

  def _write_metadata(self, num_envs: int) -> None:
    entity = self._entity()
    metadata = {
      "type": "metadata",
      "created_at": datetime.now().isoformat(timespec="seconds"),
      "task_id": self.task_id,
      "entity_name": self.entity_name,
      "num_envs": num_envs,
      "env_ids": list(self.env_ids),
      "interval": self.interval,
      "step_dt": getattr(self.env.unwrapped, "step_dt", None),
      "joint_names": list(entity.joint_names),
      "actuator_names": list(entity.actuator_names),
      "state_fields": list(_STATE_FIELDS),
      "observation_terms": self._observation_terms(),
      "observation_dims": self._observation_dims(),
      "action_terms": self._action_terms(),
      "units": {
        "joint_pos": "rad for hinge joints, m for slide joints",
        "joint_vel": "rad/s for hinge joints, m/s for slide joints",
        "joint_pos_target": "rad for hinge joints, m for slide joints",
        "joint_vel_target": "rad/s for hinge joints, m/s for slide joints",
        "joint_effort_target": "N*m for hinge joints, N for slide joints",
        "actuator_force": "actuator force output in actuator space",
        "qfrc_actuator": "N*m for hinge joints, N for slide joints",
        "ctrl": "raw MuJoCo actuator ctrl units",
      },
    }
    self._write_jsonl(metadata)

  def _observation_terms(self) -> dict[str, list[str]]:
    obs_manager = getattr(self.env.unwrapped, "observation_manager", None)
    terms = getattr(obs_manager, "active_terms", {}) if obs_manager else {}
    return {
      str(group_name): [str(term) for term in group_terms]
      for group_name, group_terms in terms.items()
    }

  def _observation_dims(self) -> dict[str, Any]:
    obs_manager = getattr(self.env.unwrapped, "observation_manager", None)
    dims = getattr(obs_manager, "group_obs_dim", {}) if obs_manager else {}
    return _json_value(dims)

  def _action_terms(self) -> list[str]:
    action_manager = getattr(self.env.unwrapped, "action_manager", None)
    terms = getattr(action_manager, "active_terms", []) if action_manager else []
    return [str(term) for term in terms]

  def _reward_done(self, step_result: tuple[Any, ...]) -> tuple[Any, Any]:
    if len(step_result) >= 3:
      return step_result[1], step_result[2]
    return None, False

  def _write_jsonl(self, row: Mapping[str, Any]) -> None:
    self._file.write(json.dumps(_json_value(row), ensure_ascii=True) + "\n")
    self._rows_since_flush += 1
    if self._rows_since_flush >= 100:
      self._file.flush()
      self._rows_since_flush = 0


class PlayTraceEnvWrapper:
  """Environment wrapper that records the exact obs/action cycle used by viewers."""

  def __init__(
    self,
    env: Any,
    *,
    path: str | Path,
    task_id: str,
    entity_name: str = "robot",
    env_ids: Sequence[int] = (0,),
    interval: int = 1,
  ) -> None:
    self.env = env
    self.num_envs: int = int(env.num_envs)
    self.recorder = PlayTraceRecorder(
      env=env,
      path=path,
      task_id=task_id,
      entity_name=entity_name,
      env_ids=env_ids,
      interval=interval,
    )
    self._pending_obs: Any | None = None

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
    obs = self.env.get_observations()
    self._pending_obs = obs
    return obs

  def reset(self, *args: Any, **kwargs: Any) -> Any:
    self._pending_obs = None
    return self.env.reset(*args, **kwargs)

  def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
    obs = self._pending_obs
    if obs is None:
      obs = self.env.get_observations()

    should_record = self.recorder.should_record
    record_payload = None
    if should_record:
      policy_output = actions.detach().clone()
      applied_action = self._applied_action(actions).detach().clone()
      pre_state = self.recorder.capture_state()
      record_payload = (policy_output, applied_action, pre_state)

    step_result = self.env.step(actions)

    if record_payload is not None:
      policy_output, applied_action, pre_state = record_payload
      post_state = self.recorder.capture_state()
      self.recorder.write_step(
        obs=obs,
        policy_output=policy_output,
        applied_action=applied_action,
        pre_state=pre_state,
        post_state=post_state,
        step_result=step_result,
      )
    self.recorder.advance_step()
    self._pending_obs = None
    return step_result

  def close(self) -> None:
    self.recorder.close()
    self.env.close()

  def _applied_action(self, actions: torch.Tensor) -> torch.Tensor:
    clip_actions = getattr(self.env, "clip_actions", None)
    if clip_actions is None:
      return actions
    return torch.clamp(actions, -clip_actions, clip_actions)


def _json_from_batch(value: Any, env_id: int, num_envs: int) -> Any:
  return _json_value(_slice_batch(value, env_id, num_envs))


def _num_envs(env: Any) -> int:
  num_envs = getattr(env, "num_envs", None)
  if num_envs is None:
    num_envs = env.unwrapped.num_envs
  return int(num_envs)


def _slice_batch(value: Any, env_id: int, num_envs: int) -> Any:
  if isinstance(value, torch.Tensor):
    if value.ndim > 0 and value.shape[0] == num_envs:
      return value[env_id]
    return value
  if isinstance(value, Mapping):
    return {
      str(key): _slice_batch(item, env_id, num_envs) for key, item in value.items()
    }
  if isinstance(value, tuple):
    return tuple(_slice_batch(item, env_id, num_envs) for item in value)
  if isinstance(value, list):
    return [_slice_batch(item, env_id, num_envs) for item in value]
  return value


def _json_value(value: Any) -> Any:
  if isinstance(value, torch.Tensor):
    cpu_value = value.detach().cpu()
    if cpu_value.ndim == 0:
      return cpu_value.item()
    return cpu_value.tolist()
  if isinstance(value, Mapping):
    return {str(key): _json_value(item) for key, item in value.items()}
  if isinstance(value, tuple):
    return [_json_value(item) for item in value]
  if isinstance(value, list):
    return [_json_value(item) for item in value]
  if isinstance(value, Path):
    return str(value)
  return value
