from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  jljlowbody_blind_rough_env_cfg,
  jljlowbody_capsule_blind_rough_env_cfg,
  jljlowbody_capsule_flat_env_cfg,
  jljlowbody_capsule_rough_env_cfg,
  jljlowbody_flat_env_cfg,
  jljlowbody_rough_env_cfg,
)
from .rl_cfg import (
  jljlowbody_blind_rough_ppo_runner_cfg,
  jljlowbody_capsule_blind_rough_ppo_runner_cfg,
  jljlowbody_capsule_ppo_runner_cfg,
  jljlowbody_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="JLJLowBody-Velocity-Rough",
  env_cfg=jljlowbody_rough_env_cfg(),
  play_env_cfg=jljlowbody_rough_env_cfg(play=True),
  rl_cfg=jljlowbody_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJLowBody-Velocity-Blind-Rough",
  env_cfg=jljlowbody_blind_rough_env_cfg(),
  play_env_cfg=jljlowbody_blind_rough_env_cfg(play=True),
  rl_cfg=jljlowbody_blind_rough_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJLowBody-Velocity-Flat",
  env_cfg=jljlowbody_flat_env_cfg(),
  play_env_cfg=jljlowbody_flat_env_cfg(play=True),
  rl_cfg=jljlowbody_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJLowBodyCapsule-Velocity-Rough",
  env_cfg=jljlowbody_capsule_rough_env_cfg(),
  play_env_cfg=jljlowbody_capsule_rough_env_cfg(play=True),
  rl_cfg=jljlowbody_capsule_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJLowBodyCapsule-Velocity-Blind-Rough",
  env_cfg=jljlowbody_capsule_blind_rough_env_cfg(),
  play_env_cfg=jljlowbody_capsule_blind_rough_env_cfg(play=True),
  rl_cfg=jljlowbody_capsule_blind_rough_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="JLJLowBodyCapsule-Velocity-Flat",
  env_cfg=jljlowbody_capsule_flat_env_cfg(),
  play_env_cfg=jljlowbody_capsule_flat_env_cfg(play=True),
  rl_cfg=jljlowbody_capsule_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
