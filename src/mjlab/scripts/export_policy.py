"""Export a trained RL policy checkpoint to ONNX."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass
class ExportPolicyCfg:
  task: tyro.conf.Positional[str]
  """Registered task ID, for example ``lowbodynormal``."""

  checkpoint_file: str
  """Path to the checkpoint to export."""

  output_file: str | None = None
  """Output ONNX path. Defaults to a sibling file with the same stem."""

  device: str | None = None
  """Torch device. Defaults to ``cuda:0`` when CUDA is available."""


def export_policy(cfg: ExportPolicyCfg) -> Path:
  """Load one checkpoint and export only that policy to ONNX."""
  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  output_path = (
    Path(cfg.output_file).expanduser().resolve()
    if cfg.output_file is not None
    else checkpoint_path.with_suffix(".onnx")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(cfg.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(vec_env, asdict(agent_cfg), device=device)

  try:
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.load(
      str(checkpoint_path),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
    metadata = get_base_metadata(env, str(checkpoint_path.parent))
    attach_metadata_to_onnx(str(output_path), metadata)
  finally:
    vec_env.close()

  print(f"[INFO] Exported ONNX policy: {output_path}")
  return output_path


def main() -> None:
  cfg = tyro.cli(ExportPolicyCfg, config=mjlab.TYRO_FLAGS)
  export_policy(cfg)


if __name__ == "__main__":
  main()
