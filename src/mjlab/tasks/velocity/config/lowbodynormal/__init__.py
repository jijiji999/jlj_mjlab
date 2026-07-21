from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import lowbodynormal_flat_env_cfg
from .rl_cfg import lowbodynormal_ppo_runner_cfg

register_mjlab_task(
  task_id="lowbodynormal",
  env_cfg=lowbodynormal_flat_env_cfg(),
  play_env_cfg=lowbodynormal_flat_env_cfg(play=True),
  rl_cfg=lowbodynormal_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
