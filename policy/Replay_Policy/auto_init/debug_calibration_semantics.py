#!/usr/bin/env python3
"""Debug step 2: inspect whether K_new/D_raw should be used for runtime undistortion."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--balance", default=None, type=float)
    parser.add_argument("--fov-scale", default=None, type=float)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = resolve_cli_path(args.config, fallback_base=REPLAY_POLICY_DIR)
    config = load_yaml(config_path)
    auto_init_cfg = config.get("auto_init", {})
    undistort_cfg = auto_init_cfg.get("undistort", {})
    video_key = args.video_key or auto_init_cfg.get("wrist_video_key", "observation.images.wrist")

    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else cache_dir / "step2_calibration_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_path = _resolve_or_extract_frame(
        args=args,
        auto_init_cfg=auto_init_cfg,
        output_dir=output_dir,
        video_key=video_key,
    )
    calibration_path = resolve_repo_path(auto_init_cfg["camera_calibration"]["path"])
    K, D, rms, keys = _load_npz_calibration(calibration_path)
    metadata_size = _metadata_image_size(args.data_dir, video_key)

    balance = float(undistort_cfg.get("balance", 0.0) if args.balance is None else args.balance)
    fov_scale = float(undistort_cfg.get("fov_scale", 1.0) if args.fov_scale is None else args.fov_scale)

    cv2 = _require_cv2()
    raw_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if raw_bgr is None:
        raise RuntimeError(f"Failed to read frame image with OpenCV: {frame_path}")
    height, width = raw_bgr.shape[:2]

    result = _undistort_and_measure(
        cv2=cv2,
        raw_bgr=raw_bgr,
        K=K,
        D=D,
        balance=balance,
        fov_scale=fov_scale,
    )

    raw_out = output_dir / f"episode_{args.episode_index:06d}_raw.png"
    undistorted_out = output_dir / f"episode_{args.episode_index:06d}_fisheye_assumption_undistorted.png"
    side_by_side_out = output_dir / f"episode_{args.episode_index:06d}_raw_vs_fisheye_assumption.png"
    displacement_out = output_dir / f"episode_{args.episode_index:06d}_undistort_displacement_heatmap.png"
    absdiff_out = output_dir / f"episode_{args.episode_index:06d}_raw_vs_undistorted_absdiff.png"

    cv2.imwrite(str(raw_out), raw_bgr)
    cv2.imwrite(str(undistorted_out), result["undistorted_bgr"])
    cv2.imwrite(str(side_by_side_out), _make_side_by_side(cv2, raw_bgr, result["undistorted_bgr"]))
    cv2.imwrite(str(displacement_out), _make_heatmap(cv2, result["displacement"]))
    cv2.imwrite(str(absdiff_out), _make_absdiff(cv2, raw_bgr, result["undistorted_bgr"]))

    summary = _build_summary(
        args=args,
        config_path=config_path,
        frame_path=frame_path,
        calibration_path=calibration_path,
        output_dir=output_dir,
        raw_out=raw_out,
        undistorted_out=undistorted_out,
        side_by_side_out=side_by_side_out,
        displacement_out=displacement_out,
        absdiff_out=absdiff_out,
        K=K,
        D=D,
        rms=rms,
        keys=keys,
        width=width,
        height=height,
        metadata_size=metadata_size,
        balance=balance,
        fov_scale=fov_scale,
        result=result,
    )

    summary_path = output_dir / f"episode_{args.episode_index:06d}_calibration_semantics.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_calibration_semantics] Wrote summary to {summary_path.resolve()}")


def _resolve_or_extract_frame(args: argparse.Namespace, auto_init_cfg: dict, output_dir: Path, video_key: str) -> Path:
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


def _load_npz_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, float | None, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    with np.load(path) as data:
        if "K_new" not in data.files or "D_raw" not in data.files:
            raise KeyError(f"Expected calibration keys K_new/D_raw in {path}; found {data.files}")
        K = np.asarray(data["K_new"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["D_raw"], dtype=np.float64).reshape(4, 1)
        rms = float(np.asarray(data["rms"]).reshape(())) if "rms" in data.files else None
    return K, D, rms, {"K": "K_new", "D": "D_raw"}


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


def _undistort_and_measure(cv2, raw_bgr: np.ndarray, K: np.ndarray, D: np.ndarray, balance: float, fov_scale: float) -> dict:
    height, width = raw_bgr.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D.reshape(4, 1),
        (width, height),
        np.eye(3),
        balance=balance,
        fov_scale=fov_scale,
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K,
        D.reshape(4, 1),
        np.eye(3),
        new_K,
        (width, height),
        cv2.CV_32FC1,
    )
    undistorted = cv2.remap(raw_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    displacement = np.sqrt((map_x - grid_x) ** 2 + (map_y - grid_y) ** 2)
    valid_map = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    diff_gray = np.mean(np.abs(raw_bgr.astype(np.float32) - undistorted.astype(np.float32)), axis=2)

    border = max(1, min(width, height) // 20)
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    blackish = np.all(undistorted <= 3, axis=2)

    return {
        "new_K": np.asarray(new_K, dtype=np.float64),
        "map_x": map_x,
        "map_y": map_y,
        "undistorted_bgr": undistorted,
        "displacement": displacement,
        "valid_map_ratio": float(np.mean(valid_map)),
        "blackish_ratio": float(np.mean(blackish)),
        "border_blackish_ratio": float(np.mean(blackish[border_mask])),
        "displacement_stats": _stats(displacement),
        "absdiff_stats": _stats(diff_gray),
        "sample_displacements": _sample_displacements(displacement),
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _sample_displacements(displacement: np.ndarray) -> dict[str, float]:
    height, width = displacement.shape
    points = {
        "center": (height // 2, width // 2),
        "top_left": (0, 0),
        "top_right": (0, width - 1),
        "bottom_left": (height - 1, 0),
        "bottom_right": (height - 1, width - 1),
        "mid_top": (0, width // 2),
        "mid_bottom": (height - 1, width // 2),
        "mid_left": (height // 2, 0),
        "mid_right": (height // 2, width - 1),
    }
    return {name: float(displacement[y, x]) for name, (y, x) in points.items()}


def _make_side_by_side(cv2, raw_bgr: np.ndarray, undistorted_bgr: np.ndarray) -> np.ndarray:
    raw = raw_bgr.copy()
    undistorted = undistorted_bgr.copy()
    cv2.putText(raw, "raw frame: no runtime undistort", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(undistorted, "fisheye assumption: K_new + D_raw", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return np.concatenate([raw, undistorted], axis=1)


def _make_heatmap(cv2, displacement: np.ndarray) -> np.ndarray:
    scale = np.percentile(displacement, 99)
    if scale <= 0:
        scale = 1.0
    normalized = np.clip(displacement / scale, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET))


def _make_absdiff(cv2, raw_bgr: np.ndarray, undistorted_bgr: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(raw_bgr, undistorted_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET))


def _build_summary(
    args: argparse.Namespace,
    config_path: Path,
    frame_path: Path,
    calibration_path: Path,
    output_dir: Path,
    raw_out: Path,
    undistorted_out: Path,
    side_by_side_out: Path,
    displacement_out: Path,
    absdiff_out: Path,
    K: np.ndarray,
    D: np.ndarray,
    rms: float | None,
    keys: dict[str, str],
    width: int,
    height: int,
    metadata_size: dict[str, int] | None,
    balance: float,
    fov_scale: float,
    result: dict,
) -> dict:
    cx, cy = float(K[0, 2]), float(K[1, 2])
    center_dx = cx - width / 2.0
    center_dy = cy - height / 2.0
    displacement_stats = result["displacement_stats"]
    absdiff_stats = result["absdiff_stats"]
    return {
        "episode_index": args.episode_index,
        "config_path": str(config_path),
        "frame_path": str(frame_path.resolve()),
        "calibration_path": str(calibration_path),
        "output_dir": str(output_dir.resolve()),
        "outputs": {
            "raw": str(raw_out.resolve()),
            "fisheye_assumption_undistorted": str(undistorted_out.resolve()),
            "raw_vs_fisheye_assumption": str(side_by_side_out.resolve()),
            "undistort_displacement_heatmap": str(displacement_out.resolve()),
            "raw_vs_undistorted_absdiff": str(absdiff_out.resolve()),
        },
        "calibration": {
            "keys": keys,
            "K_new": K.tolist(),
            "D_raw": D.reshape(-1).tolist(),
            "rms": rms,
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": cx,
            "cy": cy,
        },
        "image_size": {"width": width, "height": height},
        "metadata_image_size": metadata_size,
        "principal_point_offset_from_image_center_px": {
            "dx": center_dx,
            "dy": center_dy,
            "abs_dx": abs(center_dx),
            "abs_dy": abs(center_dy),
        },
        "fisheye_assumption": {
            "description": "Treat K_new/D_raw as raw fisheye intrinsics/distortion, then runtime-undistort the raw wrist frame.",
            "balance": balance,
            "fov_scale": fov_scale,
            "new_K_after_runtime_undistort": result["new_K"].tolist(),
            "new_K_delta_from_K_new": (result["new_K"] - K).tolist(),
            "valid_map_ratio": result["valid_map_ratio"],
            "blackish_ratio": result["blackish_ratio"],
            "border_blackish_ratio": result["border_blackish_ratio"],
            "displacement_px": displacement_stats,
            "sample_displacements_px": result["sample_displacements"],
            "raw_vs_undistorted_absdiff": absdiff_stats,
        },
        "pinhole_output_assumption": {
            "description": "Treat K_new as the pinhole intrinsics for already-preprocessed imagery; use raw frame directly and set undistort.enabled=false.",
            "foundationpose_intrinsics_if_selected": {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": cx,
                "cy": cy,
                "K": K.tolist(),
            },
        },
        "heuristics": _heuristics(
            width=width,
            height=height,
            center_dx=center_dx,
            center_dy=center_dy,
            valid_map_ratio=result["valid_map_ratio"],
            border_blackish_ratio=result["border_blackish_ratio"],
            displacement_stats=displacement_stats,
            absdiff_stats=absdiff_stats,
        ),
        "decision_guide": [
            "If raw_vs_fisheye_assumption shows straighter lines without severe cropping or content squeeze, K_new/D_raw is plausible as raw fisheye input and current undistort.enabled=true is reasonable.",
            "If the undistorted side looks over-warped, cropped, or less geometrically plausible than raw, K_new is likely already a pinhole output K; set undistort.enabled=false and use K_new directly.",
            "Without checkerboard images or known straight scene lines this script cannot prove the semantic label; it produces the evidence needed for visual confirmation.",
        ],
    }


def _heuristics(
    width: int,
    height: int,
    center_dx: float,
    center_dy: float,
    valid_map_ratio: float,
    border_blackish_ratio: float,
    displacement_stats: dict[str, float],
    absdiff_stats: dict[str, float],
) -> dict:
    centered = abs(center_dx) <= max(2.0, width * 0.01) and abs(center_dy) <= max(2.0, height * 0.01)
    nontrivial_warp = displacement_stats["p95"] > 3.0 or displacement_stats["mean"] > 1.0
    severe_border_loss = valid_map_ratio < 0.92 or border_blackish_ratio > 0.25
    changed_image = absdiff_stats["mean"] > 1.0 or absdiff_stats["p95"] > 8.0
    return {
        "principal_point_near_image_center": centered,
        "runtime_undistort_has_nontrivial_warp": nontrivial_warp,
        "runtime_undistort_has_severe_border_loss": severe_border_loss,
        "runtime_undistort_changes_image": changed_image,
        "initial_read": (
            "Numerically the current fisheye assumption is not obviously invalid; inspect the side-by-side image."
            if not severe_border_loss
            else "Runtime undistortion loses a large border or valid-map area; inspect closely for over-undistortion."
        ),
    }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for calibration semantic debugging.") from exc
    return cv2


if __name__ == "__main__":
    main()
