"""Read first-frame inputs from a LeRobot dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

try:
    from .camera_calibration import prepare_frame_and_intrinsics
    from .path_utils import resolve_repo_path
    from ..replay_lerobot_loader import (
        extract_first_frame,
        load_first_frame_state,
        resolve_episode_video_path,
    )
except ImportError:
    from auto_init.camera_calibration import prepare_frame_and_intrinsics
    from auto_init.path_utils import resolve_repo_path
    from replay_lerobot_loader import extract_first_frame, load_first_frame_state, resolve_episode_video_path


def extract_first_frame_inputs(
    config: dict,
    data_dir: str,
    episode_index: int,
    frame_image_override: str | None = None,
) -> dict:
    auto_init_cfg = deepcopy(config.get("auto_init", {}))
    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    calib_cfg = auto_init_cfg.get("camera_calibration")
    if isinstance(calib_cfg, dict) and calib_cfg.get("path"):
        calib_cfg["path"] = str(resolve_repo_path(calib_cfg["path"]))

    frame_path, video_path = _resolve_first_frame(
        auto_init_cfg=auto_init_cfg,
        data_dir=data_dir,
        episode_index=episode_index,
        cache_dir=cache_dir,
        frame_image_override=frame_image_override,
    )

    state_column = auto_init_cfg.get("state_column", config.get("state_column", "observation.state"))
    real_eef_pose = load_first_frame_state(data_dir, episode_index, state_column=state_column)

    camera_inputs = prepare_frame_and_intrinsics(
        auto_init_cfg=auto_init_cfg,
        frame_path=frame_path,
        cache_dir=cache_dir,
        episode_index=episode_index,
    )

    return {
        "frame_path": camera_inputs["frame_path"],
        "raw_frame_path": camera_inputs["raw_frame_path"],
        "video_path": str(video_path) if video_path is not None else None,
        "real_eef_pose6d": real_eef_pose.tolist(),
        "intrinsics": camera_inputs["intrinsics"],
        "intrinsics_matrix": camera_inputs["intrinsics_matrix"],
        "intrinsics_path": camera_inputs["intrinsics_path"],
        "calibration": camera_inputs["calibration"],
        "undistorted": camera_inputs["undistorted"],
        "preprocessing_debug": camera_inputs.get("preprocessing_debug"),
    }


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
