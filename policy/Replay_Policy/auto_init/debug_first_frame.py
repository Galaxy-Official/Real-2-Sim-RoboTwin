#!/usr/bin/env python3
"""Debug step 1: verify LeRobot wrist video path resolution and first-frame extraction."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
if str(REPLAY_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_POLICY_DIR))

from replay_lerobot_loader import extract_first_frame, load_dataset_info, resolve_episode_video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="policy/Replay_Policy/deploy_policy.yml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--video-key", default=None)
    parser.add_argument(
        "--output-dir",
        default="policy/Replay_Policy/init_meta/cache/step1_debug",
        help="Directory for the extracted first frame and debug summary JSON.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def extract_first_frame_for_debug(video_path: Path, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "Failed to extract the exact first frame because `ffmpeg` is unavailable. "
            "Install ffmpeg before running this debug step or the main build_init_meta pipeline."
        )
    extract_first_frame(video_path, output_path)
    return "ffmpeg"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    info = load_dataset_info(args.data_dir)
    auto_init_cfg = config.get("auto_init", {})
    video_key = args.video_key or auto_init_cfg.get("wrist_video_key", "observation.images.wrist")

    video_path = resolve_episode_video_path(args.data_dir, args.episode_index, video_key=video_key)
    output_dir = Path(args.output_dir)
    frame_path = output_dir / f"episode_{args.episode_index:06d}_wrist_first_frame.png"
    extractor = extract_first_frame_for_debug(video_path=video_path, output_path=frame_path)

    with Image.open(frame_path) as image:
        image_width, image_height = image.size

    feature_info = info.get("features", {}).get(video_key, {})
    expected_width = feature_info.get("info", {}).get("video.width")
    expected_height = feature_info.get("info", {}).get("video.height")
    if expected_width is None or expected_height is None:
        feature_shape = feature_info.get("shape", [])
        if len(feature_shape) >= 2:
            expected_height, expected_width = feature_shape[:2]

    summary = {
        "episode_index": args.episode_index,
        "video_key": video_key,
        "data_dir": str(Path(args.data_dir).resolve()),
        "resolved_video_path": str(video_path.resolve()),
        "video_path_exists": video_path.is_file(),
        "extractor": extractor,
        "exact_frame_zero_guaranteed": True,
        "extracted_frame_path": str(frame_path.resolve()),
        "extracted_frame_exists": frame_path.is_file(),
        "extracted_frame_size": {
            "width": int(image_width),
            "height": int(image_height),
        },
        "expected_frame_size_from_metadata": {
            "width": None if expected_width is None else int(expected_width),
            "height": None if expected_height is None else int(expected_height),
        },
        "size_matches_metadata": (
            expected_width is None
            or expected_height is None
            or (int(expected_width) == image_width and int(expected_height) == image_height)
        ),
        "dataset_video_template": info.get("video_path"),
        "notes": "Exact frame-0 extraction succeeded.",
    }

    summary_path = output_dir / f"episode_{args.episode_index:06d}_first_frame_check.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_first_frame] Wrote summary to {summary_path.resolve()}")

    if not summary["size_matches_metadata"]:
        raise SystemExit(
            "Extracted first-frame size does not match dataset metadata. "
            f"Expected {(expected_width, expected_height)}, got {(image_width, image_height)}."
        )


if __name__ == "__main__":
    main()
