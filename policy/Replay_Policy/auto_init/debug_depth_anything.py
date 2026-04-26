#!/usr/bin/env python3
"""Debug step 4: run Depth Anything 3 on the first-frame RGB and inspect depth output."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
if str(REPLAY_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_POLICY_DIR))

from auto_init.depth_anything_v2_runner import run_depth_anything
from auto_init.mask_provider import resolve_mask_path
from auto_init.path_utils import resolve_cli_path, resolve_repo_path
from auto_init.real_data_reader import extract_first_frame_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPLAY_POLICY_DIR / "deploy_policy.yml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--frame-image", default=None)
    parser.add_argument("--depth-path", default=None, help="Optional existing .npy depth file to analyze without running DA3.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-missing-mask", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare RGB/K inputs; do not run Depth Anything.")
    parser.add_argument("--vis-percentiles", nargs=2, type=float, default=(1.0, 99.0), metavar=("LOW", "HIGH"))
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = resolve_cli_path(args.config, fallback_base=REPLAY_POLICY_DIR)
    config = load_yaml(config_path)
    auto_init_cfg = config.get("auto_init", {})
    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else cache_dir / "step4_depth_anything_debug"
    inputs_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    run_config = deepcopy(config)
    run_config.setdefault("auto_init", {})["first_frame_cache_dir"] = str(inputs_dir)
    first_frame = extract_first_frame_inputs(
        run_config,
        args.data_dir,
        args.episode_index,
        frame_image_override=args.frame_image,
    )
    mask_path, mask_error = _resolve_mask(config, args)
    if mask_error and not args.allow_missing_mask:
        raise FileNotFoundError(
            f"{mask_error} Pass --allow-missing-mask to skip mask depth statistics."
        )

    depth_path = None
    run_status: dict[str, Any]
    if args.prepare_only:
        run_status = {"status": "prepared_only"}
    elif args.depth_path:
        depth_path = resolve_cli_path(args.depth_path)
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        run_status = {"status": "used_existing_depth", "depth_path": str(depth_path.resolve())}
    else:
        depth_path = run_depth_anything(
            config=run_config,
            image_path=first_frame["frame_path"],
            cache_dir=str(output_dir),
            episode_index=args.episode_index,
            intrinsics_path=first_frame["intrinsics_path"],
        )
        run_status = {"status": "ran_depth_anything", "depth_path": str(Path(depth_path).resolve())}

    depth_summary = None
    if depth_path is not None:
        depth_summary = _analyze_depth(
            depth_path=Path(depth_path),
            image_path=Path(first_frame["frame_path"]),
            mask_path=mask_path,
            output_dir=output_dir,
            episode_index=args.episode_index,
            vis_percentiles=tuple(args.vis_percentiles),
        )

    summary = {
        "episode_index": args.episode_index,
        "config_path": str(config_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_status": run_status,
        "inputs": {
            "image_path": str(Path(first_frame["frame_path"]).resolve()),
            "raw_frame_path": str(Path(first_frame["raw_frame_path"]).resolve()),
            "first_frame_undistorted": first_frame["undistorted"],
            "intrinsics_path": str(Path(first_frame["intrinsics_path"]).resolve()),
            "intrinsics": first_frame["intrinsics"],
            "intrinsics_matrix": first_frame["intrinsics_matrix"],
            "calibration": first_frame["calibration"],
            "mask_path": None if mask_path is None else str(mask_path.resolve()),
            "mask_error": mask_error,
        },
        "depth": depth_summary,
        "expected": {
            "image_size": [640, 480],
            "runtime_frame_should_equal_raw": True,
            "intrinsics_source_should_be": "pinhole_calibration",
            "depth_shape_should_match_image_hw": True,
            "depth_units": "meters after DA3 metric scaling",
        },
    }
    summary["checks"] = _build_checks(summary)

    summary_path = output_dir / f"episode_{args.episode_index:06d}_depth_anything_debug.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_depth_anything] Wrote summary to {summary_path.resolve()}")

    _raise_on_failed_checks(summary["checks"], require_depth=not args.prepare_only)


def _resolve_mask(config: dict, args: argparse.Namespace) -> tuple[Path | None, str | None]:
    try:
        return resolve_mask_path(config, args.data_dir, args.episode_index), None
    except FileNotFoundError as exc:
        return None, str(exc)


def _analyze_depth(
    depth_path: Path,
    image_path: Path,
    mask_path: Path | None,
    output_dir: Path,
    episode_index: int,
    vis_percentiles: tuple[float, float],
) -> dict:
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    vis_path = output_dir / f"episode_{episode_index:06d}_depth_vis.png"
    _write_depth_vis(depth, vis_path, vis_percentiles)

    mask_stats = None
    if mask_path is not None:
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape == depth.shape:
            mask_stats = _depth_stats(depth[mask])
            mask_stats["mask_area_px"] = int(mask.sum())
            mask_stats["mask_area_ratio"] = float(mask.sum() / mask.size)
        else:
            mask_stats = {
                "error": "mask shape does not match depth shape",
                "mask_shape": list(mask.shape),
                "depth_shape": list(depth.shape),
            }

    return {
        "depth_path": str(depth_path.resolve()),
        "depth_vis_path": str(vis_path.resolve()),
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "image_size": {"width": width, "height": height},
        "global_stats": _depth_stats(depth),
        "mask_stats": mask_stats,
    }


def _depth_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    total = int(values.size)
    if total == 0:
        return {"count": 0, "finite_ratio": 0.0, "positive_finite_ratio": 0.0}
    finite = np.isfinite(values)
    finite_values = values[finite]
    positive_finite = finite_values > 0
    stats = {
        "count": total,
        "finite_count": int(finite.sum()),
        "finite_ratio": float(finite.sum() / total),
        "positive_finite_count": int(positive_finite.sum()),
        "positive_finite_ratio": float(positive_finite.sum() / total),
    }
    if finite_values.size == 0:
        return stats
    for name, percentile in (
        ("min", 0),
        ("p1", 1),
        ("p5", 5),
        ("p50", 50),
        ("p95", 95),
        ("p99", 99),
        ("max", 100),
    ):
        stats[name] = float(np.percentile(finite_values, percentile))
    stats["mean"] = float(np.mean(finite_values))
    return stats


def _write_depth_vis(depth: np.ndarray, output_path: Path, percentiles: tuple[float, float]) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(depth[finite], percentiles)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(depth[finite])), float(np.nanmax(depth[finite]))
        denom = max(high - low, 1e-6)
        normalized = np.clip((depth - low) / denom, 0.0, 1.0)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, mode="L").save(output_path)


def _build_checks(summary: dict) -> dict:
    inputs = summary["inputs"]
    depth = summary["depth"]
    checks = {
        "first_frame_not_undistorted": inputs["first_frame_undistorted"] is False,
        "intrinsics_source_is_pinhole_calibration": _intrinsics_source(inputs["intrinsics_path"]) == "pinhole_calibration",
    }
    if depth is None:
        checks["depth_present"] = False
        return checks

    checks["depth_present"] = True
    checks["depth_is_2d"] = len(depth["shape"]) == 2
    image_w = int(depth["image_size"]["width"])
    image_h = int(depth["image_size"]["height"])
    checks["depth_shape_matches_image"] = depth["shape"] == [image_h, image_w]
    checks["depth_global_finite_ratio_ok"] = depth["global_stats"].get("finite_ratio", 0.0) >= 0.99
    checks["depth_global_positive_ratio_ok"] = depth["global_stats"].get("positive_finite_ratio", 0.0) >= 0.95
    if depth.get("mask_stats") and "error" not in depth["mask_stats"]:
        checks["depth_mask_finite_ratio_ok"] = depth["mask_stats"].get("finite_ratio", 0.0) >= 0.99
        checks["depth_mask_positive_ratio_ok"] = depth["mask_stats"].get("positive_finite_ratio", 0.0) >= 0.95
    return checks


def _intrinsics_source(path: str | Path) -> str | None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("source")


def _raise_on_failed_checks(checks: dict, require_depth: bool = True) -> None:
    required_checks = {
        "first_frame_not_undistorted",
        "intrinsics_source_is_pinhole_calibration",
    }
    if require_depth:
        required_checks.update(
            {
                "depth_present",
                "depth_is_2d",
                "depth_shape_matches_image",
                "depth_global_finite_ratio_ok",
                "depth_global_positive_ratio_ok",
            }
        )
    hard_failures = [
        name
        for name, ok in checks.items()
        if name in required_checks and ok is False
    ]
    if hard_failures:
        raise SystemExit(f"Depth Anything debug checks failed: {hard_failures}")


if __name__ == "__main__":
    main()
