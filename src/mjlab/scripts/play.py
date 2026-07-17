"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.scripts.play_command_eval import (
  PlayCommandEvalEnvWrapper,
  default_play_command_eval_dir,
)
from mjlab.scripts.play_report import (
  PlayReportEnvWrapper,
  default_play_report_dir,
)
from mjlab.scripts.play_trace import PlayTraceEnvWrapper, default_play_trace_path
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG, random_rough
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

PlayTerrainMode = Literal["task", "plane", "rough", "random_rough"]
PLAY_RANDOM_ROUGH_BASE_THICKNESS_RATIO = 10.0


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  play_terrain: PlayTerrainMode = "task"
  """Override terrain for playback. 'task' keeps the task's play config."""
  terrain_difficulty: float = 0.5
  """Fixed terrain difficulty for play terrain overrides, in [0, 1]."""
  terrain_rows: int = 1
  """Number of generated terrain rows for play terrain overrides."""
  terrain_cols: int = 5
  """Number of generated terrain columns for play terrain overrides."""
  terrain_seed: int | None = None
  """Optional seed for reproducible play terrain generation."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  trace: bool = False
  """Record per-step policy inputs, policy outputs, and robot actuator state."""
  trace_path: str | None = None
  """Optional JSONL output path for ``--trace``. Defaults under the play log dir."""
  trace_env_ids: tuple[int, ...] = (0,)
  """Environment ids to record when ``--trace`` is enabled."""
  trace_interval: int = 1
  """Record every N policy steps when ``--trace`` is enabled."""
  trace_entity: str = "robot"
  """Scene entity name whose joint and actuator state is recorded."""
  report: bool = False
  """Generate a play-time evaluation report with plots and summary metrics."""
  report_dir: str | None = None
  """Optional output directory for ``--report``. Defaults under the play log dir."""
  report_env_id: int = 0
  """Environment id to evaluate when ``--report`` is enabled."""
  report_interval: int = 1
  """Record every N policy steps when ``--report`` is enabled."""
  report_entity: str = "robot"
  """Scene entity name to evaluate when ``--report`` is enabled."""
  report_command_name: str = "twist"
  """Command term name to compare against robot motion in the report."""
  command_eval: bool = False
  """Record a fixed velocity-command evaluation during normal viewer playback."""
  command_eval_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Fixed ``(vx, vy, wz)`` command in body-frame m/s, m/s, and rad/s."""
  command_eval_duration: float = 5.0
  """Duration in seconds for the fixed velocity-command evaluation."""
  command_eval_dir: str | None = None
  """Optional output directory for ``--command-eval`` artifacts."""
  command_eval_env_id: int = 0
  """Environment id to evaluate when ``--command-eval`` is enabled."""
  command_eval_interval: int = 1
  """Record every N policy steps when ``--command-eval`` is enabled."""
  command_eval_entity: str = "robot"
  """Scene entity name whose joints are recorded during command evaluation."""
  command_eval_command_name: str = "twist"
  """Command term name to force during command evaluation."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _validate_play_terrain_cfg(cfg: PlayConfig) -> None:
  if not 0.0 <= cfg.terrain_difficulty <= 1.0:
    raise ValueError(
      f"terrain_difficulty must be within [0, 1], got {cfg.terrain_difficulty}."
    )
  if cfg.terrain_rows < 1:
    raise ValueError(f"terrain_rows must be >= 1, got {cfg.terrain_rows}.")
  if cfg.terrain_cols < 1:
    raise ValueError(f"terrain_cols must be >= 1, got {cfg.terrain_cols}.")


def _make_play_terrain_generator_cfg(cfg: PlayConfig) -> TerrainGeneratorCfg:
  _validate_play_terrain_cfg(cfg)
  difficulty_range = (cfg.terrain_difficulty, cfg.terrain_difficulty)

  if cfg.play_terrain == "rough":
    terrain_cfg = deepcopy(ROUGH_TERRAINS_CFG)
    terrain_cfg.curriculum = False
    terrain_cfg.num_rows = cfg.terrain_rows
    terrain_cfg.num_cols = cfg.terrain_cols
    terrain_cfg.border_width = 10.0
    terrain_cfg.difficulty_range = difficulty_range
    terrain_cfg.seed = cfg.terrain_seed
    return terrain_cfg

  if cfg.play_terrain == "random_rough":
    return TerrainGeneratorCfg(
      seed=cfg.terrain_seed,
      curriculum=False,
      size=(6.0, 6.0),
      border_width=10.0,
      num_rows=cfg.terrain_rows,
      num_cols=cfg.terrain_cols,
      difficulty_range=difficulty_range,
      sub_terrains={
        "random_rough": random_rough(
          proportion=1.0,
          noise_range=(0.005, 0.025),
          noise_step=0.005,
          horizontal_scale=0.15,
          downsampled_scale=0.3,
          border_width=0.25,
          base_thickness_ratio=PLAY_RANDOM_ROUGH_BASE_THICKNESS_RATIO,
          scale_with_difficulty=True,
        ),
      },
      add_lights=True,
    )

  raise ValueError(f"Unsupported play terrain override: {cfg.play_terrain}")


def _apply_play_terrain_override(
  env_cfg: ManagerBasedRlEnvCfg,
  cfg: PlayConfig,
) -> None:
  if cfg.play_terrain == "task":
    return

  terrain_cfg = env_cfg.scene.terrain
  if terrain_cfg is None:
    terrain_cfg = TerrainEntityCfg()
    env_cfg.scene.terrain = terrain_cfg

  if cfg.play_terrain == "plane":
    terrain_cfg.terrain_type = "plane"
    terrain_cfg.terrain_generator = None
    terrain_cfg.max_init_terrain_level = None
    print("[INFO]: Play terrain override: plane")
    return

  terrain_generator = _make_play_terrain_generator_cfg(cfg)
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = terrain_generator
  terrain_cfg.max_init_terrain_level = cfg.terrain_rows - 1
  print(
    "[INFO]: Play terrain override: "
    f"{cfg.play_terrain}, difficulty={cfg.terrain_difficulty:.3f}"
  )


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  _apply_play_terrain_override(env_cfg, cfg)

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  play_env = env
  viewer_num_steps: int | None = None
  if cfg.command_eval:
    command_eval_dir = (
      Path(cfg.command_eval_dir).expanduser()
      if cfg.command_eval_dir is not None
      else default_play_command_eval_dir(
        task_id,
        log_dir=log_dir,
        log_root=cfg.log_root,
      )
    )
    play_env = PlayCommandEvalEnvWrapper(
      play_env,
      output_dir=command_eval_dir,
      task_id=task_id,
      command_velocity=cfg.command_eval_velocity,
      duration_s=cfg.command_eval_duration,
      entity_name=cfg.command_eval_entity,
      command_name=cfg.command_eval_command_name,
      env_id=cfg.command_eval_env_id,
      interval=cfg.command_eval_interval,
      checkpoint_file=resume_path,
    )
    viewer_num_steps = play_env.recorder.num_steps
    print(
      "[INFO] Recording fixed command evaluation while playing: "
      f"vx={cfg.command_eval_velocity[0]:+.3f}, "
      f"vy={cfg.command_eval_velocity[1]:+.3f}, "
      f"wz={cfg.command_eval_velocity[2]:+.3f}, "
      f"duration={cfg.command_eval_duration:.3f}s"
    )
    print(
      "[INFO] Command evaluation sampling: "
      f"policy={1.0 / play_env.recorder.step_dt:.1f}Hz, "
      f"physics_joint={1.0 / play_env.recorder.physics_dt:.1f}Hz "
      f"(dt={play_env.recorder.physics_dt:.6f}s, "
      f"decimation={play_env.recorder.decimation})"
    )
    print(f"[INFO] Command evaluation output: {command_eval_dir}")

  if cfg.trace:
    trace_path = (
      Path(cfg.trace_path).expanduser()
      if cfg.trace_path is not None
      else default_play_trace_path(task_id, log_dir=log_dir, log_root=cfg.log_root)
    )
    play_env = PlayTraceEnvWrapper(
      env,
      path=trace_path,
      task_id=task_id,
      entity_name=cfg.trace_entity,
      env_ids=cfg.trace_env_ids,
      interval=cfg.trace_interval,
    )
    print(f"[INFO] Recording play trace to: {trace_path}")

  if cfg.report:
    report_dir = (
      Path(cfg.report_dir).expanduser()
      if cfg.report_dir is not None
      else default_play_report_dir(task_id, log_dir=log_dir, log_root=cfg.log_root)
    )
    play_env = PlayReportEnvWrapper(
      play_env,
      output_dir=report_dir,
      task_id=task_id,
      entity_name=cfg.report_entity,
      command_name=cfg.report_command_name,
      env_id=cfg.report_env_id,
      interval=cfg.report_interval,
    )
    print(f"[INFO] Recording play report to: {report_dir}")

  try:
    if resolved_viewer == "native":
      NativeMujocoViewer(play_env, policy).run(num_steps=viewer_num_steps)
    elif resolved_viewer == "viser":
      ViserPlayViewer(play_env, policy, checkpoint_manager=ckpt_manager).run(
        num_steps=viewer_num_steps
      )
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    play_env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
