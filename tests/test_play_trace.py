from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts.play_trace import PlayTraceEnvWrapper


class FakeScene:
  def __init__(self, entity) -> None:
    self.entities = {"robot": entity}

  def __getitem__(self, key: str):
    return self.entities[key]


class FakeEntity:
  joint_names = ("hip", "knee")
  actuator_names = ("hip_motor", "knee_motor")

  def __init__(self) -> None:
    ctrl_ids = torch.tensor([0, 1])
    self.data = SimpleNamespace(
      joint_pos=torch.tensor([[0.1, 0.2], [1.1, 1.2]]),
      joint_vel=torch.tensor([[0.3, 0.4], [1.3, 1.4]]),
      joint_pos_target=torch.tensor([[0.5, 0.6], [1.5, 1.6]]),
      joint_vel_target=torch.tensor([[0.7, 0.8], [1.7, 1.8]]),
      joint_effort_target=torch.tensor([[0.9, 1.0], [1.9, 2.0]]),
      actuator_force=torch.tensor([[1.1, 1.2], [2.1, 2.2]]),
      qfrc_actuator=torch.tensor([[1.3, 1.4], [2.3, 2.4]]),
      indexing=SimpleNamespace(ctrl_ids=ctrl_ids),
      data=SimpleNamespace(ctrl=torch.tensor([[1.5, 1.6], [2.5, 2.6]])),
    )


class FakeEnv:
  def __init__(self) -> None:
    self.num_envs = 2
    self.device = torch.device("cpu")
    self.clip_actions = 0.5
    self.cfg = SimpleNamespace(viewer=SimpleNamespace())
    self.entity = FakeEntity()
    self.unwrapped = SimpleNamespace(
      step_dt=0.02,
      scene=FakeScene(self.entity),
      observation_manager=SimpleNamespace(
        active_terms={"actor": ["obs"]},
        group_obs_dim={"actor": (3,)},
      ),
      action_manager=SimpleNamespace(active_terms=["joint_action"]),
    )
    self.closed = False

  def get_observations(self):
    return {"actor": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}

  def reset(self):
    return self.get_observations(), {}

  def step(self, actions: torch.Tensor):
    del actions
    self.entity.data.joint_pos += 10.0
    self.entity.data.qfrc_actuator += 20.0
    obs = self.get_observations()
    rewards = torch.tensor([1.25, 2.25])
    dones = torch.tensor([0, 1])
    return obs, rewards, dones, {}

  def close(self) -> None:
    self.closed = True


def test_play_trace_records_policy_io_and_actuator_state(tmp_path: Path) -> None:
  trace_path = tmp_path / "trace.jsonl"
  env = FakeEnv()
  wrapped = PlayTraceEnvWrapper(
    env,
    path=trace_path,
    task_id="Fake-Task",
    env_ids=(0,),
  )

  _ = wrapped.get_observations()
  wrapped.step(torch.tensor([[2.0, -2.0], [0.25, 0.75]]))
  wrapped.close()

  rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
  assert rows[0]["type"] == "metadata"
  assert rows[0]["joint_names"] == ["hip", "knee"]
  assert rows[0]["actuator_names"] == ["hip_motor", "knee_motor"]

  step = rows[1]
  assert step["type"] == "step"
  assert step["step"] == 0
  assert step["time_s"] == 0.0
  assert step["env_id"] == 0
  assert step["policy_input"]["actor"] == [1.0, 2.0, 3.0]
  assert step["policy_output"] == [2.0, -2.0]
  assert step["applied_action"] == [0.5, -0.5]
  assert step["pre"]["joint_pos"] == pytest.approx([0.1, 0.2])
  assert step["post"]["joint_pos"] == pytest.approx([10.1, 10.2])
  assert step["pre"]["qfrc_actuator"] == pytest.approx([1.3, 1.4])
  assert step["post"]["qfrc_actuator"] == pytest.approx([21.3, 21.4])
  assert step["pre"]["ctrl"] == pytest.approx([1.5, 1.6])
  assert step["reward"] == pytest.approx(1.25)
  assert step["done"] is False
  assert env.closed is True
