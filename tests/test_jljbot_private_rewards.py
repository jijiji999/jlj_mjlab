"""Tests for JLJBot-private reward helpers."""

from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.tasks.velocity.config.jljbot import rewards as jljbot_rewards


def test_zero_reward_shape_and_device() -> None:
  env = cast(Any, SimpleNamespace(num_envs=3, device="cpu"))
  reward = jljbot_rewards.zero(env)
  assert reward.shape == (3,)
  assert reward.device == torch.device("cpu")
  assert torch.all(reward == 0.0)


def test_hip_roll_initial_deviation_l2_penalizes_default_pose_error() -> None:
  class FakeRobot:
    data = SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.20, 1.0, -0.40],
          [0.60, 0.0, -0.10],
        ]
      ),
      default_joint_pos=torch.tensor(
        [
          [0.10, 1.0, -0.30],
          [0.10, 0.0, -0.30],
        ]
      ),
    )

    def find_joints(self, joint_names):
      assert joint_names == (".*_hip_roll_joint",)
      return [0, 2], ["left_hip_roll_joint", "right_hip_roll_joint"]

  env = cast(Any, SimpleNamespace(scene={"robot": FakeRobot()}))
  reward = jljbot_rewards.hip_roll_initial_deviation_l2(env, std=0.5)
  torch.testing.assert_close(reward, torch.tensor([0.08, 1.16]))


def test_hip_pitch_initial_deviation_l2_penalizes_default_pose_error() -> None:
  class FakeRobot:
    data = SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.20, 1.0, -0.30],
          [0.70, 0.0, 0.10],
        ]
      ),
      default_joint_pos=torch.tensor(
        [
          [0.0, 1.0, -0.10],
          [0.20, 0.0, 0.10],
        ]
      ),
    )

    def find_joints(self, joint_names):
      assert joint_names == (".*_hip_pitch_joint",)
      return [0, 2], ["left_hip_pitch_joint", "right_hip_pitch_joint"]

  env = cast(Any, SimpleNamespace(scene={"robot": FakeRobot()}))
  reward = jljbot_rewards.hip_pitch_initial_deviation_l2(env, std=0.5)
  torch.testing.assert_close(reward, torch.tensor([0.32, 1.0]))


def test_hip_yaw_initial_deviation_l2_penalizes_default_pose_error() -> None:
  class FakeRobot:
    data = SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.25, 1.0, -0.15],
          [0.45, 0.0, 0.25],
        ]
      ),
      default_joint_pos=torch.tensor(
        [
          [0.15, 1.0, -0.15],
          [0.05, 0.0, -0.15],
        ]
      ),
    )

    def find_joints(self, joint_names):
      assert joint_names == (".*_hip_yaw_joint",)
      return [0, 2], ["left_hip_yaw_joint", "right_hip_yaw_joint"]

  env = cast(Any, SimpleNamespace(scene={"robot": FakeRobot()}))
  reward = jljbot_rewards.hip_yaw_initial_deviation_l2(env, std=0.5)
  torch.testing.assert_close(reward, torch.tensor([0.04, 1.28]))


def test_waist_roll_pitch_initial_deviation_l2_penalizes_default_pose_error() -> None:
  class FakeRobot:
    data = SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.10, -0.20, 1.0],
          [0.40, 0.60, 0.0],
        ]
      ),
      default_joint_pos=torch.tensor(
        [
          [0.0, 0.0, 1.0],
          [0.10, 0.20, 0.0],
        ]
      ),
    )

    def find_joints(self, joint_names):
      assert joint_names == ("waist_roll_joint", "waist_pitch_joint")
      return [0, 1], ["waist_roll_joint", "waist_pitch_joint"]

  env = cast(Any, SimpleNamespace(scene={"robot": FakeRobot()}))
  reward = jljbot_rewards.waist_roll_pitch_initial_deviation_l2(env, std=0.2)
  torch.testing.assert_close(reward, torch.tensor([1.25, 6.25]))


def test_arm_initial_deviation_l2_penalizes_default_pose_error() -> None:
  class FakeRobot:
    data = SimpleNamespace(
      joint_pos=torch.tensor(
        [
          [0.30, -0.10, 0.20, -0.50, 0.40],
          [0.10, -0.20, 0.40, -0.10, 0.00],
        ]
      ),
      default_joint_pos=torch.tensor(
        [
          [0.20, -0.20, 0.10, -0.40, 0.20],
          [0.00, -0.20, 0.10, -0.30, 0.10],
        ]
      ),
    )

    def find_joints(self, joint_names):
      assert joint_names == (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",
      )
      return [0, 1, 2, 3, 4], [
        "left_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
      ]

  env = cast(Any, SimpleNamespace(scene={"robot": FakeRobot()}))
  reward = jljbot_rewards.arm_initial_deviation_l2(env, std=0.2)
  torch.testing.assert_close(reward, torch.tensor([2.0, 3.75]))
