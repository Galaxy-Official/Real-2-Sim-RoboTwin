"""Invoke FoundationPose through configurable command or precomputed pose files."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import numpy as np

try:
    from ..坐标系转换 import pose6d_to_matrix, pose7d_wxyz_to_matrix
except ImportError:
    from 坐标系转换 import pose6d_to_matrix, pose7d_wxyz_to_matrix


def run_foundationpose(
    config: dict,
    image_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    mesh_path: str,
    cache_dir: str,
    episode_index: int,
) -> tuple[np.ndarray, Path]:
    fp_cfg = config.get("auto_init", {}).get("foundationpose", {})
    mode = fp_cfg.get("mode", "command")
    output_template = fp_cfg.get(
        "output_template",
        "{cache_dir}/episode_{episode_index:06d}_foundationpose.json",
    )
    output_path = Path(
        output_template.format(
            cache_dir=cache_dir,
            episode_index=episode_index,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "precomputed":
        pose_file = Path(fp_cfg["path"].format(cache_dir=cache_dir, episode_index=episode_index))
        return _load_pose_matrix(pose_file), pose_file

    if mode != "command":
        raise ValueError(f"Unsupported foundationpose mode: {mode}")

    command = _format_command(
        fp_cfg.get("command", []),
        image_path=image_path,
        rgb_path=image_path,
        depth_path=depth_path,
        mask_path=mask_path,
        intrinsics_path=intrinsics_path,
        mesh_path=mesh_path,
        output_path=str(output_path),
        cache_dir=cache_dir,
        episode_index=episode_index,
    )
    if not command:
        raise ValueError(
            "auto_init.foundationpose.command is empty. "
            "Fill it with a server-side command that writes a pose json."
        )
    subprocess.run(command, check=True)
    return _load_pose_matrix(output_path), output_path


def _load_pose_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FoundationPose output not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    for key in ("matrix", "cam_T_obj", "T_cam_obj"):
        if key in payload:
            return np.asarray(payload[key], dtype=np.float64).reshape(4, 4)
    if "pose6d_rotvec" in payload:
        return pose6d_to_matrix(payload["pose6d_rotvec"])
    if "pose7d_wxyz" in payload:
        return pose7d_wxyz_to_matrix(payload["pose7d_wxyz"])
    raise ValueError(
        f"Unsupported FoundationPose output format in {path}. "
        "Expected one of: matrix / cam_T_obj / T_cam_obj / pose6d_rotvec / pose7d_wxyz."
    )


def _format_command(command_cfg, **kwargs) -> list[str]:
    if isinstance(command_cfg, str):
        command_cfg = shlex.split(command_cfg)
    return [str(part).format(**kwargs) for part in command_cfg]
