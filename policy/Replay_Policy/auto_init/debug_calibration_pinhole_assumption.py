#!/usr/bin/env python3
"""Debug step 2 alternative: treat K_new as already-undistorted pinhole intrinsics."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
DEFAULT_CALIBRATION_PATH = THIS_DIR / "fisheye_calib_result_resized.npz"
if str(REPLAY_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_POLICY_DIR))

from auto_init.path_utils import resolve_cli_path, resolve_repo_path
from replay_lerobot_loader import extract_first_frame, load_dataset_info, resolve_episode_video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPLAY_POLICY_DIR / "deploy_policy.yml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--frame-image", default=None, help="Optional existing raw first-frame image.")
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Optional K_new/D_raw npz. Defaults to config path, then bundled fisheye_calib_result_resized.npz.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--fisheye-debug-dir",
        default=None,
        help="Optional output dir from debug_calibration_semantics.py for side-by-side comparison.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = resolve_cli_path(args.config, fallback_base=REPLAY_POLICY_DIR)
    config = load_yaml(config_path)
    auto_init_cfg = config.get("auto_init", {})
    video_key = args.video_key or auto_init_cfg.get("wrist_video_key", "observation.images.wrist")

    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else cache_dir / "step2_pinhole_assumption_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_path = _resolve_or_extract_frame(args, output_dir, video_key)
    calibration_path = _resolve_calibration_path(args, auto_init_cfg)
    K, D, rms = _load_npz_calibration(calibration_path)
    metadata_size = _metadata_image_size(args.data_dir, video_key)

    raw_out = output_dir / f"episode_{args.episode_index:06d}_pinhole_assumption_input.png"
    _copy_image(frame_path, raw_out)
    width, height = _image_size(raw_out)

    intrinsics_path = output_dir / f"episode_{args.episode_index:06d}_pinhole_assumption_intrinsics.json"
    _write_intrinsics(
        path=intrinsics_path,
        K=K,
        source="pinhole_output_assumption_from_K_new",
        calibration_path=calibration_path,
        D_raw=D,
        rms=rms,
    )

    comparison_path = _write_optional_comparison(
        args=args,
        cache_dir=cache_dir,
        raw_out=raw_out,
        output_dir=output_dir,
    )

    snippet_path = output_dir / "deploy_policy_pinhole_assumption_snippet.yml"
    snippet_path.write_text(_deploy_policy_snippet(K, calibration_path), encoding="utf-8")

    summary = {
        "episode_index": args.episode_index,
        "config_path": str(config_path),
        "calibration_path": str(calibration_path),
        "output_dir": str(output_dir.resolve()),
        "assumption": {
            "name": "pinhole_output_assumption",
            "description": "Use raw wrist frame directly and use K_new as the pinhole camera matrix. No runtime fisheye undistortion.",
            "pipeline_change_if_selected": {
                "auto_init.undistort.enabled": False,
                "auto_init.undistort.mask": False,
                "auto_init.camera_calibration.type": "pinhole",
                "auto_init.camera_calibration.path": _snippet_calibration_path(calibration_path),
                "K_new_as_intrinsics": _matrix_to_intrinsics_dict(K),
            },
        },
        "inputs_for_downstream_debug": {
            "image_path": str(raw_out.resolve()),
            "intrinsics_path": str(intrinsics_path.resolve()),
            "mesh_and_mask": "Unchanged from deploy_policy.yml / object_config.",
        },
        "outputs": {
            "pinhole_assumption_input": str(raw_out.resolve()),
            "pinhole_assumption_intrinsics": str(intrinsics_path.resolve()),
            "deploy_policy_snippet": str(snippet_path.resolve()),
            "pinhole_vs_fisheye_assumption": None if comparison_path is None else str(comparison_path.resolve()),
        },
        "calibration": {
            "K_new": K.tolist(),
            "D_raw_recorded_but_not_used": D.reshape(-1).tolist(),
            "rms": rms,
            **_matrix_to_intrinsics_dict(K),
        },
        "image_size": {"width": width, "height": height},
        "metadata_image_size": metadata_size,
        "principal_point_offset_from_image_center_px": {
            "dx": float(K[0, 2] - width / 2.0),
            "dy": float(K[1, 2] - height / 2.0),
        },
        "compare_against_fisheye_reference": {
            "run_fisheye_reference_script": "python auto_init/debug_calibration_semantics.py --config deploy_policy.yml --data-dir ../../replay_data/block_stack --episode-index 0",
            "fisheye_reference_output_dir": str((cache_dir / "step2_calibration_debug").resolve()),
            "fisheye_reference_image_to_compare": f"episode_{args.episode_index:06d}_fisheye_assumption_undistorted.png",
            "this_logic_image_to_compare": raw_out.name,
        },
        "decision_guide": [
            "If this raw/pinhole-assumption image looks more geometrically plausible than the fisheye-assumption image, set undistort.enabled=false and use K_new directly.",
            "If the fisheye-assumption image straightens geometry without severe cropping or squeeze, runtime undistortion may be the better path.",
            "Use the intrinsics JSON from this folder to run Depth Anything 3 / FoundationPose manually under the pinhole-output assumption.",
        ],
    }
    summary_path = output_dir / f"episode_{args.episode_index:06d}_pinhole_assumption.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_calibration_pinhole_assumption] Wrote summary to {summary_path.resolve()}")


def _resolve_or_extract_frame(args: argparse.Namespace, output_dir: Path, video_key: str) -> Path:
    if args.frame_image:
        frame_path = resolve_cli_path(args.frame_image)
        if not frame_path.is_file():
            raise FileNotFoundError(f"Frame image not found: {frame_path}")
        return frame_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract the raw first frame for this debug step.")

    video_path = resolve_episode_video_path(args.data_dir, args.episode_index, video_key=video_key)
    frame_path = output_dir / f"episode_{args.episode_index:06d}_raw_first_frame.png"
    extract_first_frame(video_path, frame_path)
    return frame_path


def _resolve_calibration_path(args: argparse.Namespace, auto_init_cfg: dict) -> Path:
    if args.calibration_path:
        return resolve_cli_path(args.calibration_path)
    calib_cfg = auto_init_cfg.get("camera_calibration", {})
    if isinstance(calib_cfg, dict) and calib_cfg.get("path"):
        return resolve_repo_path(calib_cfg["path"])
    return DEFAULT_CALIBRATION_PATH.resolve()


def _load_npz_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with np.load(path) as data:
        if "K_new" not in data.files or "D_raw" not in data.files:
            raise KeyError(f"Expected calibration keys K_new/D_raw in {path}; found {data.files}")
        K = np.asarray(data["K_new"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["D_raw"], dtype=np.float64).reshape(4, 1)
        rms = float(np.asarray(data["rms"]).reshape(())) if "rms" in data.files else None
    return K, D, rms


def _metadata_image_size(data_dir: str, video_key: str) -> dict[str, int] | None:
    try:
        info = load_dataset_info(data_dir)
    except Exception:
        return None
    feature = info.get("features", {}).get(video_key, {})
    video_info = feature.get("info", {})
    width = video_info.get("video.width")
    height = video_info.get("video.height")
    if width is None or height is None:
        shape = feature.get("shape", [])
        if len(shape) >= 2:
            height, width = shape[:2]
    if width is None or height is None:
        return None
    return {"width": int(width), "height": int(height)}


def _copy_image(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        image.save(output_path)


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _write_intrinsics(path: Path, K: np.ndarray, source: str, calibration_path: Path, D_raw: np.ndarray, rms: float | None) -> None:
    payload = _matrix_to_intrinsics_dict(K)
    payload.update(
        {
            "K": np.asarray(K, dtype=float).tolist(),
            "source": source,
            "calibration_path": str(calibration_path),
            "D_raw_recorded_but_not_used": np.asarray(D_raw, dtype=float).reshape(-1).tolist(),
            "rms": rms,
        }
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _matrix_to_intrinsics_dict(K: np.ndarray) -> dict[str, float]:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def _write_optional_comparison(args: argparse.Namespace, cache_dir: Path, raw_out: Path, output_dir: Path) -> Path | None:
    fisheye_dir = resolve_cli_path(args.fisheye_debug_dir) if args.fisheye_debug_dir else cache_dir / "step2_calibration_debug"
    fisheye_image = fisheye_dir / f"episode_{args.episode_index:06d}_fisheye_assumption_undistorted.png"
    if not fisheye_image.is_file():
        return None

    output_path = output_dir / f"episode_{args.episode_index:06d}_pinhole_vs_fisheye_assumption.png"
    left = _open_rgb_with_label(raw_out, "pinhole assumption: raw frame + K_new")
    right = _open_rgb_with_label(fisheye_image, "fisheye reference: runtime undistort")
    height = max(left.height, right.height)
    width = left.width + right.width
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(output_path)
    return output_path


def _open_rgb_with_label(path: Path, label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, min(image.width - 10, 520), 44), fill=(0, 0, 0))
    draw.text((18, 18), label, fill=(255, 255, 0))
    return image


def _deploy_policy_snippet(K: np.ndarray, calibration_path: Path) -> str:
    intrinsics = _matrix_to_intrinsics_dict(K)
    path_text = _snippet_calibration_path(calibration_path)
    return (
        "auto_init:\n"
        "  camera_calibration:\n"
        "    type: pinhole\n"
        f"    path: {path_text}\n"
        "  undistort:\n"
        "    enabled: false\n"
        "    mask: false\n"
        "  # Manual fallback only if camera_calibration.path is removed.\n"
        "  intrinsics:\n"
        f"    fx: {intrinsics['fx']:.10f}\n"
        f"    fy: {intrinsics['fy']:.10f}\n"
        f"    cx: {intrinsics['cx']:.10f}\n"
        f"    cy: {intrinsics['cy']:.10f}\n"
    )


def _snippet_calibration_path(calibration_path: Path) -> str:
    if calibration_path.resolve() == DEFAULT_CALIBRATION_PATH.resolve():
        return "policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz"
    return str(calibration_path)


if __name__ == "__main__":
    main()
