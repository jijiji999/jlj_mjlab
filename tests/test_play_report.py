from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts.play_report import PlayReportEnvWrapper


class FakeScene:
  def __init__(self, entity) -> None:
    self.entities = {"robot": entity}

  def __getitem__(self, key: str):
    return self.entities[key]


class FakeCommandManager:
  def __init__(self) -> None:
    self._command = torch.tensor([[1.0, 0.0, 0.2], [0.5, -0.1, -0.3]])

  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return self._command


class FakeRewardManager:
  def get_active_iterable_terms(self, env_idx: int):
    values = [
      [0.8, -0.1],
      [0.4, -0.2],
    ]
    return [
      ("track_linear_velocity", [values[env_idx][0]]),
      ("track_angular_velocity", [values[env_idx][1]]),
    ]


class FakeMetricsManager:
  def get_active_iterable_terms(self, env_idx: int):
    values = [0.12, 0.34]
    return [("mean_action_acc", [values[env_idx]])]


class FakeEntity:
  joint_names = ("hip", "knee")
  actuator_names = ("hip_motor", "knee_motor")

  def __init__(self) -> None:
    self.data = SimpleNamespace(
      root_link_lin_vel_b=torch.tensor([[0.7, 0.1, 0.0], [0.4, -0.2, 0.0]]),
      root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.05], [0.0, 0.0, -0.2]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )


class FakeEnv:
  def __init__(self) -> None:
    self.num_envs = 2
    self.device = torch.device("cpu")
    self.cfg = SimpleNamespace(viewer=SimpleNamespace())
    self.entity = FakeEntity()
    self.unwrapped = SimpleNamespace(
      step_dt=0.02,
      scene=FakeScene(self.entity),
      command_manager=FakeCommandManager(),
      reward_manager=FakeRewardManager(),
      metrics_manager=FakeMetricsManager(),
      action_manager=SimpleNamespace(
        action=torch.zeros((2, 3)),
        prev_action=torch.zeros((2, 3)),
        prev_prev_action=torch.zeros((2, 3)),
      ),
    )
    self.closed = False

  def get_observations(self):
    return {"actor": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

  def reset(self):
    return self.get_observations(), {}

  def step(self, actions: torch.Tensor):
    del actions
    obs = self.get_observations()
    rewards = torch.tensor([1.25, 2.25])
    dones = torch.tensor([0, 0])
    return obs, rewards, dones, {}

  def close(self) -> None:
    self.closed = True


def test_play_report_generates_summary_and_plots(tmp_path: Path) -> None:
  output_dir = tmp_path / "report"
  env = FakeEnv()
  wrapped = PlayReportEnvWrapper(
    env,
    output_dir=output_dir,
    task_id="Fake-Velocity-Task",
    entity_name="robot",
    command_name="twist",
    env_id=0,
  )

  _ = wrapped.get_observations()
  wrapped.step(torch.zeros((2, 3)))
  wrapped.close()

  summary = json.loads((output_dir / "summary.json").read_text())
  assert summary["task_id"] == "Fake-Velocity-Task"
  assert summary["num_steps"] == 1
  assert summary["done_count"] == 0
  assert summary["tracking"]["rmse_vx"] == pytest.approx(0.3)
  assert summary["tracking"]["rmse_vy"] == pytest.approx(0.1)
  assert summary["tracking"]["rmse_wz"] == pytest.approx(0.15)
  assert summary["control"]["mean_action_acc"] == pytest.approx(0.12)
  assert summary["reward"]["mean_total_reward"] == pytest.approx(1.25)
  assert summary["reward_term_means"]["track_linear_velocity"] == pytest.approx(0.8)
  assert summary["reward_term_means"]["track_angular_velocity"] == pytest.approx(-0.1)

  with (output_dir / "timeseries.csv").open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
  assert len(rows) == 1
  assert float(rows[0]["command_vx"]) == pytest.approx(1.0)
  assert float(rows[0]["actual_vx"]) == pytest.approx(0.7)
  assert float(rows[0]["metric_mean_action_acc"]) == pytest.approx(0.12)

  assert (output_dir / "overview.png").exists()
  assert (output_dir / "reward_terms.png").exists()
  assert env.closed is True
