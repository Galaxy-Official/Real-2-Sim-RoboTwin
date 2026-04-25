#!/usr/bin/env python3
"""Debug step 3: verify camera_calibration.py image, mask, and intrinsics outputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
if str(REPLAY_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_POLICY_DIR))

from auto_init.camera_calibration import maybe_undistort_mask
from auto_init.mask_provider import resolve_mask_path
from auto_init.path_utils import resolve_cli_path, resolve_repo_path
from auto_init.real_data_reader import extract_first_frame_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPLAY_POLICY_DIR / "deploy_policy.yml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--frame-image", default=None, help="Optional existing raw first-frame image.")
    parser.add_argument("--mask-image", default=None, help="Optional explicit mask image for this episode.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-missing-mask",
        action="store_true",
        help="Write image/intrinsics diagnostics even if the configured mask is missing.",
    )
    parser.add_argument(
        "--include-fisheye-reference",
        action="store_true",
        help="Also force camera_calibration.type=fisheye and undistort.enabled=true for comparison.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = resolve_cli_path(args.config, fallback_base=REPLAY_POLICY_DIR)
    config = load_yaml(config_path)
    auto_init_cfg = config.get("auto_init", {})

    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else cache_dir / "step3_camera_calibration_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path, mask_error = _resolve_debug_mask(args, config)
    if mask_error and not args.allow_missing_mask:
        raise FileNotFoundError(
            f"{mask_error} Pass --mask-image /path/to/mask.png or --allow-missing-mask to skip mask checks."
        )

    current = _run_calibration_pass(
        name="current_config",
        config=config,
        data_dir=args.data_dir,
        episode_index=args.episode_index,
        frame_image=args.frame_image,
        mask_path=mask_path,
        pass_dir=output_dir / "current_config",
        output_dir=output_dir,
    )

    fisheye_reference = None
    if args.include_fisheye_reference:
        fisheye_reference = _run_fisheye_reference_pass(
            config=config,
            data_dir=args.data_dir,
            episode_index=args.episode_index,
            frame_image=args.frame_image,
            mask_path=mask_path,
            pass_dir=output_dir / "fisheye_reference",
            output_dir=output_dir,
        )

    summary = {
        "episode_index": args.episode_index,
        "config_path": str(config_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "mask_input": None if mask_path is None else str(mask_path.resolve()),
        "mask_resolution_error": mask_error,
        "current_config": current,
        "fisheye_reference": fisheye_reference,
        "expected_current_result": _expected_current_result(config),
        "decision_note": (
            "Current raw+K_new pinhole logic should keep frame_path equal to raw_frame_path, "
            "keep the mask unchanged, and write intrinsics with source=pinhole_calibration."
        ),
    }

    summary_path = output_dir / f"episode_{args.episode_index:06d}_camera_calibration_outputs.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_camera_calibration_outputs] Wrote summary to {summary_path.resolve()}")

    _raise_if_current_logic_is_unexpected(summary)


def _resolve_debug_mask(args: argparse.Namespace, config: dict) -> tuple[Path | None, str | None]:
    if args.mask_image:
        path = resolve_cli_path(args.mask_image)
        if not path.is_file():
            return None, f"Explicit mask image not found: {path}"
        return path, None
    try:
        return resolve_mask_path(config, args.data_dir, args.episode_index), None
    except FileNotFoundError as exc:
        return None, str(exc)


def _run_calibration_pass(
    name: str,
    config: dict,
    data_dir: str,
    episode_index: int,
    frame_image: str | None,
    mask_path: Path | None,
    pass_dir: Path,
    output_dir: Path,
) -> dict:
    pass_dir.mkdir(parents=True, exist_ok=True)
    run_config = deepcopy(config)
    run_config.setdefault("auto_init", {})["first_frame_cache_dir"] = str(pass_dir)

    first_frame = extract_first_frame_inputs(
        run_config,
        data_dir,
        episode_index,
        frame_image_override=frame_image,
    )
    processed_mask_path = None
    if mask_path is not None:
        processed_mask_path = maybe_undistort_mask(
            auto_init_cfg=run_config.get("auto_init", {}),
            mask_path=mask_path,
            cache_dir=pass_dir,
            episode_index=episode_index,
            calibration=first_frame.get("calibration"),
        )

    copied = _copy_pass_outputs(
        name=name,
        output_dir=output_dir,
        episode_index=episode_index,
        first_frame=first_frame,
        mask_path=mask_path,
        processed_mask_path=processed_mask_path,
    )
    intrinsics = _load_json(first_frame["intrinsics_path"])

    return {
        "pass_dir": str(pass_dir.resolve()),
        "frame_path": str(Path(first_frame["frame_path"]).resolve()),
        "raw_frame_path": str(Path(first_frame["raw_frame_path"]).resolve()),
        "frame_undistorted": bool(first_frame["undistorted"]),
        "mask_path": None if processed_mask_path is None else str(Path(processed_mask_path).resolve()),
        "mask_undistorted": processed_mask_path is not None
        and mask_path is not None
        and Path(processed_mask_path).resolve() != Path(mask_path).resolve(),
        "intrinsics_path": str(Path(first_frame["intrinsics_path"]).resolve()),
        "intrinsics_json": intrinsics,
        "intrinsics_source": intrinsics.get("source"),
        "calibration": first_frame["calibration"],
        "copied_outputs": copied,
        "checks": {
            "frame_matches_raw": _compare_arrays(first_frame["raw_frame_path"], first_frame["frame_path"]),
            "mask_matches_raw": None
            if mask_path is None or processed_mask_path is None
            else _compare_arrays(mask_path, processed_mask_path, grayscale=True),
            "intrinsics_match_runtime": _compare_K(intrinsics.get("K"), first_frame["intrinsics_matrix"]),
        },
    }


def _run_fisheye_reference_pass(
    config: dict,
    data_dir: str,
    episode_index: int,
    frame_image: str | None,
    mask_path: Path | None,
    pass_dir: Path,
    output_dir: Path,
) -> dict:
    reference_config = deepcopy(config)
    auto_init_cfg = reference_config.setdefault("auto_init", {})
    auto_init_cfg["first_frame_cache_dir"] = str(pass_dir)
    auto_init_cfg["camera_calibration"] = auto_init_cfg.get("camera_calibration") or {}
    auto_init_cfg["undistort"] = auto_init_cfg.get("undistort") or {}
    auto_init_cfg["camera_calibration"]["type"] = "fisheye"
    auto_init_cfg["undistort"]["enabled"] = True
    auto_init_cfg["undistort"]["mask"] = True
    try:
        return _run_calibration_pass(
            name="fisheye_reference",
            config=reference_config,
            data_dir=data_dir,
            episode_index=episode_index,
            frame_image=frame_image,
            mask_path=mask_path,
            pass_dir=pass_dir,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - debug script should report optional reference failures.
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "hint": "Install opencv-python if the error says cv2 is missing.",
        }


def _copy_pass_outputs(
    name: str,
    output_dir: Path,
    episode_index: int,
    first_frame: dict,
    mask_path: Path | None,
    processed_mask_path: Path | None,
) -> dict:
    prefix = f"episode_{episode_index:06d}_{name}"
    raw_frame_copy = output_dir / f"{prefix}_raw_frame.png"
    runtime_frame_copy = output_dir / f"{prefix}_runtime_frame.png"
    intrinsics_copy = output_dir / f"{prefix}_intrinsics.json"

    _copy_file(first_frame["raw_frame_path"], raw_frame_copy)
    _copy_file(first_frame["frame_path"], runtime_frame_copy)
    _copy_file(first_frame["intrinsics_path"], intrinsics_copy)

    frame_pair = output_dir / f"{prefix}_raw_vs_runtime_frame.png"
    _write_side_by_side(
        left=raw_frame_copy,
        right=runtime_frame_copy,
        output=frame_pair,
        left_label=f"{name}: raw frame",
        right_label=f"{name}: runtime frame",
    )

    outputs = {
        "raw_frame": str(raw_frame_copy.resolve()),
        "runtime_frame": str(runtime_frame_copy.resolve()),
        "raw_vs_runtime_frame": str(frame_pair.resolve()),
        "intrinsics": str(intrinsics_copy.resolve()),
        "raw_mask": None,
        "runtime_mask": None,
        "raw_vs_runtime_mask": None,
    }
    if mask_path is not None and processed_mask_path is not None:
        raw_mask_copy = output_dir / f"{prefix}_raw_mask.png"
        runtime_mask_copy = output_dir / f"{prefix}_runtime_mask.png"
        mask_pair = output_dir / f"{prefix}_raw_vs_runtime_mask.png"
        _copy_image_like(mask_path, raw_mask_copy, grayscale=True)
        _copy_image_like(processed_mask_path, runtime_mask_copy, grayscale=True)
        _write_side_by_side(
            left=raw_mask_copy,
            right=runtime_mask_copy,
            output=mask_pair,
            left_label=f"{name}: raw mask",
            right_label=f"{name}: runtime mask",
        )
        outputs["raw_mask"] = str(raw_mask_copy.resolve())
        outputs["runtime_mask"] = str(runtime_mask_copy.resolve())
        outputs["raw_vs_runtime_mask"] = str(mask_pair.resolve())
    return outputs


def _copy_file(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(src), dst)


def _copy_image_like(src: str | Path, dst: Path, grayscale: bool = False) -> None:
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".npy":
        array = np.load(src)
        image = Image.fromarray(_array_to_uint8(array, grayscale=grayscale))
        image.save(dst)
        return
    with Image.open(src) as image:
        image = image.convert("L" if grayscale else "RGB")
        image.save(dst)


def _write_side_by_side(left: Path, right: Path, output: Path, left_label: str, right_label: str) -> None:
    left_image = _open_labeled(left, left_label)
    right_image = _open_labeled(right, right_label)
    height = max(left_image.height, right_image.height)
    width = left_image.width + right_image.width
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(left_image, (0, 0))
    canvas.paste(right_image, (left_image.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _open_labeled(path: Path, label: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, min(image.width - 10, 560), 44), fill=(0, 0, 0))
    draw.text((18, 18), label, fill=(255, 255, 0))
    return image


def _compare_arrays(left: str | Path, right: str | Path, grayscale: bool = False) -> dict:
    left_path = Path(left)
    right_path = Path(right)
    left_array = _read_array(left_path, grayscale=grayscale)
    right_array = _read_array(right_path, grayscale=grayscale)
    same_path = left_path.resolve() == right_path.resolve()
    if left_array.shape != right_array.shape:
        return {
            "same_path": same_path,
            "shape_equal": False,
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
            "exact_equal": False,
        }
    diff = np.abs(left_array.astype(np.float32) - right_array.astype(np.float32))
    return {
        "same_path": same_path,
        "shape_equal": True,
        "shape": list(left_array.shape),
        "exact_equal": bool(np.array_equal(left_array, right_array)),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "nonzero_diff_ratio": float(np.count_nonzero(diff) / diff.size),
    }


def _read_array(path: Path, grayscale: bool = False) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
        return _array_to_uint8(array, grayscale=grayscale)
    with Image.open(path) as image:
        return np.asarray(image.convert("L" if grayscale else "RGB"))


def _array_to_uint8(array: np.ndarray, grayscale: bool = False) -> np.ndarray:
    array = np.asarray(array)
    if grayscale and array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.uint8:
        return array
    if array.dtype == bool:
        return array.astype(np.uint8) * 255
    clipped = np.clip(array, 0, 255)
    return clipped.astype(np.uint8)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _compare_K(left: Any, right: Any) -> dict:
    if left is None or right is None:
        return {"available": False, "allclose": False}
    left_array = np.asarray(left, dtype=np.float64).reshape(3, 3)
    right_array = np.asarray(right, dtype=np.float64).reshape(3, 3)
    return {
        "available": True,
        "allclose": bool(np.allclose(left_array, right_array, rtol=0.0, atol=1e-9)),
        "max_abs_diff": float(np.max(np.abs(left_array - right_array))),
    }


def _expected_current_result(config: dict) -> dict:
    auto_init_cfg = config.get("auto_init", {})
    calib_cfg = auto_init_cfg.get("camera_calibration") or {}
    undistort_cfg = auto_init_cfg.get("undistort") or {}
    return {
        "camera_calibration.type": calib_cfg.get("type"),
        "camera_calibration.path": calib_cfg.get("path"),
        "undistort.enabled": undistort_cfg.get("enabled"),
        "undistort.mask": undistort_cfg.get("mask"),
        "expected_frame_undistorted": False
        if str(calib_cfg.get("type", "")).lower() == "pinhole"
        else bool(undistort_cfg.get("enabled", True)),
        "expected_intrinsics_source_for_current_default": "pinhole_calibration",
    }


def _raise_if_current_logic_is_unexpected(summary: dict) -> None:
    current = summary["current_config"]
    expected = summary["expected_current_result"]
    if str(expected["camera_calibration.type"]).lower() != "pinhole":
        return
    if current["frame_undistorted"]:
        raise SystemExit("Current pinhole config unexpectedly produced an undistorted frame.")
    frame_check = current["checks"]["frame_matches_raw"]
    if not frame_check.get("exact_equal"):
        raise SystemExit("Current pinhole config changed the RGB frame; expected raw frame to pass through.")
    mask_check = current["checks"].get("mask_matches_raw")
    if mask_check is not None and not mask_check.get("exact_equal"):
        raise SystemExit("Current pinhole config changed the mask; expected raw mask to pass through.")
    if current["intrinsics_source"] != "pinhole_calibration":
        raise SystemExit(
            "Current pinhole config wrote an unexpected intrinsics source: "
            f"{current['intrinsics_source']!r}"
        )


if __name__ == "__main__":
    main()
