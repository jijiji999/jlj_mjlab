from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  jljbot_flat_env_cfg,
  jljbot_rough_env_cfg,
)
from .rl_cfg import jljbot_ppo_runner_cfg

register_mjlab_task(
  task_id="JLJBot-Velocity-Rough",
  env_cfg=jljbot_rough_env_cfg(),
  play_env_cfg=jljbot_rough_env_cfg(play=True),
  rl_cfg=jljbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJBot-Velocity-Flat",
  env_cfg=jljbot_flat_env_cfg(),
  play_env_cfg=jljbot_flat_env_cfg(play=True),
  rl_cfg=jljbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
