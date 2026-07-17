from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.scripts.play_command_eval import (
  PlayCommandEvalEnvWrapper,
  run_play_command_eval,
)


class FakeScene:
  def __init__(self, entity) -> None:
    self.entities = {"robot": entity}

  def __getitem__(self, key: str):
    return self.entities[key]


class FakeVelocityCommand:
  def __init__(self) -> None:
    self.vel_command_b = torch.zeros((2, 3))
    self.vel_command_w = torch.zeros((2, 3))
    self.is_heading_env = torch.ones(2, dtype=torch.bool)
    self.is_standing_env = torch.ones(2, dtype=torch.bool)
    self.is_world_env = torch.ones(2, dtype=torch.bool)
    self.is_forward_env = torch.ones(2, dtype=torch.bool)
    self.time_left = torch.zeros(2)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b


class FakeCommandManager:
  def __init__(self) -> None:
    self.term = FakeVelocityCommand()

  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return self.term.command

  def get_term(self, name: str) -> FakeVelocityCommand:
    assert name == "twist"
    return self.term


class FakeEntity:
  joint_names = ("hip", "knee")

  def __init__(self) -> None:
    self.data = SimpleNamespace(
      root_link_lin_vel_b=torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
      root_link_ang_vel_b=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]]),
      joint_pos=torch.tensor([[0.1, 0.2], [1.1, 1.2]]),
      joint_pos_target=torch.tensor([[0.15, 0.25], [1.15, 1.25]]),
      joint_vel=torch.tensor([[0.3, 0.4], [1.3, 1.4]]),
      qfrc_actuator=torch.tensor([[0.5, 0.6], [1.5, 1.6]]),
    )


class FakeEnv:
  def __init__(self) -> None:
    self.num_envs = 2
    self.device = torch.device("cpu")
    self.clip_actions = 0.5
    self.physics_dt = 0.02
    self.decimation = 5
    self.physics_step_count = 0
    self.physics_step_callbacks = []
    self.entity = FakeEntity()
    self.command_manager = FakeCommandManager()
    self.unwrapped = SimpleNamespace(
      step_dt=0.1,
      physics_dt=self.physics_dt,
      cfg=SimpleNamespace(decimation=self.decimation),
      scene=FakeScene(self.entity),
      command_manager=self.command_manager,
      add_physics_step_callback=self.add_physics_step_callback,
    )
    self.step_count = 0

  def get_observations(self):
    return {"actor": torch.zeros((2, 3))}

  def add_physics_step_callback(self, callback):
    self.physics_step_callbacks.append(callback)

    def remove_callback() -> None:
      if callback in self.physics_step_callbacks:
        self.physics_step_callbacks.remove(callback)

    return remove_callback

  def step(self, actions: torch.Tensor):
    del actions
    self.step_count += 1
    self.entity.data.root_link_lin_vel_b += 0.1
    self.entity.data.root_link_ang_vel_b[:, 2] += 0.05
    for _ in range(self.decimation):
      self.physics_step_count += 1
      self.entity.data.joint_pos += 2.0
      self.entity.data.joint_pos_target += 8.0
      self.entity.data.joint_vel += 4.0
      self.entity.data.qfrc_actuator += 6.0
      for callback in tuple(self.physics_step_callbacks):
        callback(self.physics_step_count, self.physics_step_count * self.physics_dt)
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([0, 0])
    return self.get_observations(), rewards, dones, {}

  def close(self) -> None:
    pass


def test_play_command_eval_generates_fixed_command_joint_outputs(
  tmp_path: Path,
) -> None:
  env = FakeEnv()

  def policy(obs) -> torch.Tensor:
    del obs
    return torch.tensor([[0.25, -0.25, 0.1, -0.1], [1.0, -1.0, 0.2, -0.2]])

  output_dir = run_play_command_eval(
    env=env,
    policy=policy,
    task_id="Fake-Velocity-Task",
    output_dir=tmp_path / "eval",
    command_velocity=(0.7, -0.2, 0.3),
    duration_s=0.2,
    entity_name="robot",
    command_name="twist",
    env_id=1,
  )

  assert env.step_count == 2
  assert env.command_manager.term.vel_command_b[0].tolist() == pytest.approx(
    [0.7, -0.2, 0.3]
  )
  assert env.command_manager.term.vel_command_b[1].tolist() == pytest.approx(
    [0.7, -0.2, 0.3]
  )
  assert torch.all(~env.command_manager.term.is_standing_env)
  assert torch.all(env.command_manager.term.time_left > 0.0)

  summary = json.loads((output_dir / "summary.json").read_text())
  assert summary["task_id"] == "Fake-Velocity-Task"
  assert summary["num_policy_steps"] == 2
  assert summary["num_samples"] == 3
  assert summary["num_policy_samples"] == 3
  assert summary["num_joint_samples"] == 11
  assert summary["command_velocity"]["vx"] == pytest.approx(0.7)
  assert summary["actual_duration_s"] == pytest.approx(0.2)
  assert summary["joint_duration_s"] == pytest.approx(0.2)
  assert summary["sample_frequency_hz"] == pytest.approx(10.0)
  assert summary["policy_sample_frequency_hz"] == pytest.approx(10.0)
  assert summary["joint_sample_frequency_hz"] == pytest.approx(50.0)
  assert summary["physics_dt"] == pytest.approx(0.02)
  assert summary["decimation"] == 5
  assert summary["joint_names"] == ["hip", "knee"]

  with (output_dir / "joint_timeseries.csv").open(newline="", encoding="utf-8") as f:
    joint_rows = list(csv.DictReader(f))
  assert len(joint_rows) == 11
  assert "policy_action/action_0" not in joint_rows[0]
  assert float(joint_rows[1]["time_s"]) == pytest.approx(0.02)
  assert int(joint_rows[1]["physics_step"]) == 1
  assert int(joint_rows[1]["policy_step"]) == 1
  assert float(joint_rows[-1]["joint_pos/hip"]) == pytest.approx(21.1)
  assert float(joint_rows[-1]["joint_pos_target/hip"]) == pytest.approx(81.15)
  assert float(joint_rows[-1]["joint_vel/knee"]) == pytest.approx(41.4)
  assert float(joint_rows[-1]["joint_torque/hip"]) == pytest.approx(61.5)
  assert float(joint_rows[-1]["joint_power/hip"]) == pytest.approx(2539.95)

  with (output_dir / "policy_timeseries.csv").open(newline="", encoding="utf-8") as f:
    policy_rows = list(csv.DictReader(f))
  assert len(policy_rows) == 3
  assert float(policy_rows[-1]["policy_action/action_0"]) == pytest.approx(1.0)
  assert float(policy_rows[-1]["applied_action/action_0"]) == pytest.approx(0.5)

  joint_arrays = np.load(output_dir / "joint_timeseries.npz")
  assert joint_arrays["joint_pos"].shape == (11, 2)
  assert joint_arrays["joint_pos_target"][-1, 0] == pytest.approx(81.15)
  assert joint_arrays["joint_torque"][-1, 1] == pytest.approx(61.6)
  assert joint_arrays["joint_power"][-1, 0] == pytest.approx(2539.95)
  assert joint_arrays["policy_step"][-1] == 2

  policy_arrays = np.load(output_dir / "policy_timeseries.npz")
  assert policy_arrays["policy_action"].shape == (3, 4)
  assert policy_arrays["policy_action"][-1, 0] == pytest.approx(1.0)
  assert policy_arrays["applied_action"][-1, 0] == pytest.approx(0.5)

  assert (output_dir / "joint_positions.png").exists()
  assert (output_dir / "joint_position_targets.png").exists()
  assert (output_dir / "joint_velocities.png").exists()
  assert (output_dir / "joint_torques.png").exists()
  assert (output_dir / "joint_powers.png").exists()
  assert (output_dir / "policy_actions.png").exists()
  assert (output_dir / "applied_actions.png").exists()
  assert (output_dir / "joint_positions.html").exists()
  assert (output_dir / "joint_position_targets.html").exists()
  assert (output_dir / "joint_position_compare.html").exists()
  assert (output_dir / "joint_velocities.html").exists()
  assert (output_dir / "joint_torques.html").exists()
  assert (output_dir / "joint_powers.html").exists()
  assert (output_dir / "policy_actions.html").exists()
  assert (output_dir / "applied_actions.html").exists()
  html = (output_dir / "joint_positions.html").read_text()
  assert "Click legend rows" in html
  assert "hip" in html
  compare_html = (output_dir / "joint_position_compare.html").read_text()
  assert "Joint Position vs Target" in compare_html
  assert "hip measured" in compare_html
  assert "hip target" in compare_html
  policy_html = (output_dir / "policy_actions.html").read_text()
  assert "NaN" not in policy_html
  policy_payload = policy_html.split('<script id="plot-data" type="application/json">')[
    1
  ].split("</script>")[0]
  assert json.loads(policy_payload)["names"] == [
    "action_0",
    "action_1",
    "action_2",
    "action_3",
  ]
  readme = (output_dir / "README.md").read_text()
  assert "uv run play Fake-Velocity-Task" in readme
  assert "policy_action/action_i" in readme
  assert "joint_power/<joint>" in readme
  assert "joint_sample_frequency_hz" in readme
  assert "policy_timeseries.csv" in readme


def test_play_command_eval_wrapper_records_while_stepping_viewer_env(
  tmp_path: Path,
) -> None:
  env = FakeEnv()
  wrapped = PlayCommandEvalEnvWrapper(
    env,
    output_dir=tmp_path / "viewer_eval",
    task_id="Fake-Velocity-Task",
    command_velocity=(0.4, 0.1, -0.2),
    duration_s=0.2,
    entity_name="robot",
    command_name="twist",
    env_id=0,
  )

  _ = wrapped.get_observations()
  wrapped.step(torch.zeros((2, 4)))
  wrapped.step(torch.zeros((2, 4)))
  wrapped.close()

  assert env.step_count == 2
  assert env.physics_step_count == 10
  assert env.physics_step_callbacks == []
  assert wrapped.recorder.step_count == 2
  assert env.command_manager.term.vel_command_b[0].tolist() == pytest.approx(
    [0.4, 0.1, -0.2]
  )

  summary = json.loads((tmp_path / "viewer_eval" / "summary.json").read_text())
  assert summary["num_policy_steps"] == 2
  assert summary["num_samples"] == 3
  assert summary["num_joint_samples"] == 11
  assert summary["artifacts"]["joint_positions_html"] == "joint_positions.html"
  assert (
    summary["artifacts"]["joint_position_targets_html"] == "joint_position_targets.html"
  )
  assert (
    summary["artifacts"]["joint_position_compare_html"] == "joint_position_compare.html"
  )
  assert summary["artifacts"]["readme"] == "README.md"
