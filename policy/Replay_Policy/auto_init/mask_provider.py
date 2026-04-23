"""Resolve the first-frame mask required by FoundationPose."""

from __future__ import annotations

from pathlib import Path

try:
    from .path_utils import resolve_repo_path
except ImportError:
    from auto_init.path_utils import resolve_repo_path


def resolve_mask_path(config: dict, data_dir: str, episode_index: int) -> Path:
    mask_cfg = config.get("auto_init", {}).get("mask", {})
    mode = mask_cfg.get("mode", "file_template")
    if mode == "file_template":
        template = mask_cfg.get("template")
        if not template:
            raise ValueError("auto_init.mask.template must be set when mask mode is file_template")
        mask_path = Path(
            template.format(
                data_dir=data_dir,
                episode_index=episode_index,
            )
        )
    elif mode == "explicit":
        mask_path = resolve_repo_path(mask_cfg["path"])
    else:
        raise ValueError(
            f"Unsupported mask mode: {mode}. "
            "Use a precomputed mask path or extend mask_provider.py for automatic segmentation."
        )

    if not mask_path.is_file():
        raise FileNotFoundError(
            f"FoundationPose mask not found for episode {episode_index}: {mask_path}"
        )
    return mask_path
