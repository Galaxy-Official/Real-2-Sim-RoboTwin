"""Invoke Depth Anything V2 through configurable command or precomputed depth files."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

try:
    from .path_utils import REPO_ROOT, resolve_repo_path
except ImportError:
    from auto_init.path_utils import REPO_ROOT, resolve_repo_path


def run_depth_anything(config: dict, image_path: str, cache_dir: str, episode_index: int) -> Path:
    depth_cfg = config.get("auto_init", {}).get("depth_anything", {})
    mode = depth_cfg.get("mode", "command")
    output_template = depth_cfg.get(
        "output_template",
        "{cache_dir}/episode_{episode_index:06d}_depth.npy",
    )
    output_path = Path(
        output_template.format(
            cache_dir=cache_dir,
            episode_index=episode_index,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "precomputed":
        precomputed = resolve_repo_path(
            depth_cfg["path"].format(cache_dir=cache_dir, episode_index=episode_index)
        )
        if not precomputed.is_file():
            raise FileNotFoundError(f"Precomputed depth file not found: {precomputed}")
        return precomputed

    if mode != "command":
        raise ValueError(f"Unsupported depth_anything mode: {mode}")

    command = _format_command(
        depth_cfg.get("command", []),
        image_path=image_path,
        output_path=str(output_path),
        depth_path=str(output_path),
        cache_dir=cache_dir,
        episode_index=episode_index,
    )
    if not command:
        raise ValueError(
            "auto_init.depth_anything.command is empty. "
            "Fill it with a server-side command that writes a depth file."
        )
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))
    if not output_path.is_file():
        raise RuntimeError(f"Depth Anything command completed but no depth file was created: {output_path}")
    return output_path


def _format_command(command_cfg, **kwargs) -> list[str]:
    if isinstance(command_cfg, str):
        command_cfg = shlex.split(command_cfg)
    return [str(part).format(**kwargs) for part in command_cfg]
