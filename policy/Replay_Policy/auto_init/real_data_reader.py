"""Read first-frame inputs from a LeRobot dataset."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from ..replay_lerobot_loader import (
        extract_first_frame,
        load_first_frame_state,
        resolve_episode_video_path,
    )
except ImportError:
    from replay_lerobot_loader import extract_first_frame, load_first_frame_state, resolve_episode_video_path


def extract_first_frame_inputs(
    config: dict,
    data_dir: str,
    episode_index: int,
    frame_image_override: str | None = None,
) -> dict:
    auto_init_cfg = config.get("auto_init", {})
    cache_dir = Path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame_path, video_path = _resolve_first_frame(
        auto_init_cfg=auto_init_cfg,
        data_dir=data_dir,
        episode_index=episode_index,
        cache_dir=cache_dir,
        frame_image_override=frame_image_override,
    )

    state_column = auto_init_cfg.get("state_column", config.get("state_column", "observation.state"))
    real_eef_pose = load_first_frame_state(data_dir, episode_index, state_column=state_column)

    intrinsics_cfg = auto_init_cfg.get("intrinsics", {})
    intrinsics = {
        "fx": float(intrinsics_cfg.get("fx", 0.0)),
        "fy": float(intrinsics_cfg.get("fy", 0.0)),
        "cx": float(intrinsics_cfg.get("cx", 0.0)),
        "cy": float(intrinsics_cfg.get("cy", 0.0)),
    }

    return {
        "frame_path": str(frame_path),
        "video_path": str(video_path) if video_path is not None else None,
        "real_eef_pose6d": real_eef_pose.tolist(),
        "intrinsics": intrinsics,
        "intrinsics_path": _write_intrinsics(cache_dir / f"episode_{episode_index:06d}_intrinsics.json", intrinsics),
    }


def _write_intrinsics(path: Path, intrinsics: dict) -> str:
    path.write_text(json.dumps(intrinsics, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _resolve_first_frame(auto_init_cfg: dict, data_dir: str, episode_index: int, cache_dir: Path, frame_image_override: str | None):
    if frame_image_override:
        frame_path = Path(frame_image_override)
        if not frame_path.is_file():
            raise FileNotFoundError(f"Explicit first-frame image not found: {frame_path}")
        return frame_path, None

    image_template = auto_init_cfg.get("first_frame_image_template")
    if image_template:
        frame_path = Path(
            image_template.format(
                data_dir=data_dir,
                episode_index=episode_index,
            )
        )
        if not frame_path.is_file():
            raise FileNotFoundError(f"Configured first-frame image not found: {frame_path}")
        return frame_path, None

    wrist_video_key = auto_init_cfg.get("wrist_video_key", "observation.images.wrist")
    video_path = resolve_episode_video_path(data_dir, episode_index, video_key=wrist_video_key)
    frame_path = cache_dir / f"episode_{episode_index:06d}_wrist.png"
    extract_first_frame(video_path, frame_path)
    return frame_path, video_path
