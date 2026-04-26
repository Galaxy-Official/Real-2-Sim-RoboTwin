#!/usr/bin/env python3
"""Debug step 5: run FoundationPose on prepared RGB/depth/mask/K/mesh inputs."""

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

from auto_init.foundationpose_runner import run_foundationpose
from auto_init.mask_provider import resolve_mask_path
from auto_init.path_utils import REPLAY_POLICY_DIR, REPO_ROOT, resolve_cli_path, resolve_existing_path, resolve_repo_path
from auto_init.real_data_reader import extract_first_frame_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPLAY_POLICY_DIR / "deploy_policy.yml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--depth-path", default=None, help="Existing depth .npy. Defaults to step4 output.")
    parser.add_argument("--frame-image", default=None)
    parser.add_argument("--mask-image", default=None)
    parser.add_argument("--mesh", default=None, help="Override object mesh path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs but do not run FoundationPose.")
    parser.add_argument("--allow-placeholder-mesh", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config_path = resolve_cli_path(args.config, fallback_base=REPLAY_POLICY_DIR)
    config = load_yaml(config_path)
    auto_init_cfg = config.get("auto_init", {})
    cache_dir = resolve_repo_path(auto_init_cfg.get("first_frame_cache_dir", "policy/Replay_Policy/init_meta/cache"))
    output_dir = resolve_cli_path(args.output_dir) if args.output_dir else cache_dir / "step5_foundationpose_debug"
    inputs_dir = output_dir / "inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    object_config_path = resolve_repo_path(config["object_config_path"])
    object_config = load_yaml(object_config_path)
    mesh_path = _resolve_mesh(args, object_config, object_config_path)
    mesh_status = _mesh_status(mesh_path)
    if mesh_status["looks_placeholder"] and not args.allow_placeholder_mesh:
        raise FileNotFoundError(
            "Object mesh is still a placeholder. Update object_configs/block_stack_default.yml "
            "or pass --mesh /path/to/object_mesh.obj before running FoundationPose. "
            f"Current mesh path: {mesh_path}"
        )
    if not mesh_path.is_file() and not args.allow_placeholder_mesh:
        raise FileNotFoundError(f"Object mesh file not found: {mesh_path}")

    run_config = deepcopy(config)
    run_config.setdefault("auto_init", {})["first_frame_cache_dir"] = str(inputs_dir)
    first_frame = extract_first_frame_inputs(
        run_config,
        args.data_dir,
        args.episode_index,
        frame_image_override=args.frame_image,
    )
    depth_path = _resolve_depth_path(args, cache_dir)
    mask_path = _resolve_mask_path(config, args)

    input_summary = _summarize_inputs(
        image_path=Path(first_frame["frame_path"]),
        raw_frame_path=Path(first_frame["raw_frame_path"]),
        depth_path=depth_path,
        mask_path=mask_path,
        intrinsics_path=Path(first_frame["intrinsics_path"]),
        mesh_path=mesh_path,
        first_frame_undistorted=first_frame["undistorted"],
    )
    _write_input_overlay(
        image_path=Path(first_frame["frame_path"]),
        mask_path=mask_path,
        output_path=output_dir / f"episode_{args.episode_index:06d}_foundationpose_input_overlay.png",
    )

    fp_output_path = None
    pose_summary = None
    run_status: dict[str, Any]
    if args.prepare_only:
        run_status = {"status": "prepared_only"}
    else:
        cam_T_obj, fp_output_path = run_foundationpose(
            config=run_config,
            image_path=first_frame["frame_path"],
            depth_path=str(depth_path),
            mask_path=str(mask_path),
            intrinsics_path=first_frame["intrinsics_path"],
            mesh_path=str(mesh_path),
            cache_dir=str(output_dir),
            episode_index=args.episode_index,
        )
        pose_summary = _summarize_pose(cam_T_obj)
        run_status = {
            "status": "ran_foundationpose",
            "foundationpose_output_path": str(Path(fp_output_path).resolve()),
        }

    summary = {
        "episode_index": args.episode_index,
        "config_path": str(config_path),
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "object_config_path": str(object_config_path),
        "object_config": object_config,
        "mesh_status": mesh_status,
        "run_status": run_status,
        "inputs": input_summary,
        "foundationpose_output_path": None if fp_output_path is None else str(Path(fp_output_path).resolve()),
        "pose": pose_summary,
    }
    summary["checks"] = _build_checks(summary, require_pose=not args.prepare_only)

    summary_path = output_dir / f"episode_{args.episode_index:06d}_foundationpose_debug.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[debug_foundationpose] Wrote summary to {summary_path.resolve()}")

    _raise_on_failed_checks(summary["checks"], allow_placeholder_mesh=args.allow_placeholder_mesh)


def _resolve_mesh(args: argparse.Namespace, object_config: dict, object_config_path: Path) -> Path:
    if args.mesh:
        return resolve_cli_path(args.mesh)
    return resolve_existing_path(
        object_config["mesh_path"],
        REPO_ROOT,
        REPLAY_POLICY_DIR,
        object_config_path.parent,
    )


def _mesh_status(mesh_path: Path) -> dict:
    path_text = str(mesh_path)
    looks_placeholder = "replace_with" in path_text or "placeholder" in path_text
    return {
        "mesh_path": str(mesh_path.resolve()),
        "exists": mesh_path.is_file(),
        "suffix": mesh_path.suffix.lower(),
        "looks_placeholder": looks_placeholder,
    }


def _resolve_depth_path(args: argparse.Namespace, cache_dir: Path) -> Path:
    if args.depth_path:
        depth_path = resolve_cli_path(args.depth_path)
    else:
        depth_path = cache_dir / "step4_depth_anything_debug" / f"episode_{args.episode_index:06d}_depth.npy"
    if not depth_path.is_file():
        raise FileNotFoundError(
            f"Depth file not found: {depth_path}. Run step 4 first or pass --depth-path."
        )
    return depth_path.resolve()


def _resolve_mask_path(config: dict, args: argparse.Namespace) -> Path:
    if args.mask_image:
        mask_path = resolve_cli_path(args.mask_image)
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        return mask_path.resolve()
    return resolve_mask_path(config, args.data_dir, args.episode_index).resolve()


def _summarize_inputs(
    image_path: Path,
    raw_frame_path: Path,
    depth_path: Path,
    mask_path: Path,
    intrinsics_path: Path,
    mesh_path: Path,
    first_frame_undistorted: bool,
) -> dict:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    depth = np.asarray(np.load(depth_path), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    return {
        "image_path": str(image_path.resolve()),
        "raw_frame_path": str(raw_frame_path.resolve()),
        "first_frame_undistorted": bool(first_frame_undistorted),
        "depth_path": str(depth_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "intrinsics_path": str(intrinsics_path.resolve()),
        "mesh_path": str(mesh_path.resolve()),
        "rgb_shape": list(rgb.shape),
        "depth_shape": list(depth.shape),
        "mask_shape": list(mask.shape),
        "mask_area_px": int(mask.sum()),
        "mask_area_ratio": float(mask.sum() / mask.size),
        "depth_stats_global": _depth_stats(depth),
        "depth_stats_mask": _depth_stats(depth[mask]) if mask.shape == depth.shape else None,
        "intrinsics": intrinsics,
    }


def _write_input_overlay(image_path: Path, mask_path: Path, output_path: Path) -> None:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    if mask.shape != rgb.shape[:2]:
        return
    overlay = rgb.copy()
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([0, 255, 80], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output_path)


def _depth_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"count": 0, "finite_ratio": 0.0, "positive_finite_ratio": 0.0}
    finite = np.isfinite(values)
    finite_values = values[finite]
    positive_finite = finite_values > 0
    stats = {
        "count": int(values.size),
        "finite_ratio": float(finite.sum() / values.size),
        "positive_finite_ratio": float(positive_finite.sum() / values.size),
    }
    if finite_values.size:
        for key, percentile in (("min", 0), ("p5", 5), ("p50", 50), ("p95", 95), ("max", 100)):
            stats[key] = float(np.percentile(finite_values, percentile))
        stats["mean"] = float(np.mean(finite_values))
    return stats


def _summarize_pose(cam_T_obj: np.ndarray) -> dict:
    matrix = np.asarray(cam_T_obj, dtype=np.float64).reshape(4, 4)
    R = matrix[:3, :3]
    t = matrix[:3, 3]
    return {
        "matrix": matrix.tolist(),
        "translation_xyz_m": t.tolist(),
        "translation_norm_m": float(np.linalg.norm(t)),
        "rotation_det": float(np.linalg.det(R)),
        "rotation_orthogonality_error_fro": float(np.linalg.norm(R.T @ R - np.eye(3))),
        "bottom_row": matrix[3].tolist(),
    }


def _build_checks(summary: dict, require_pose: bool) -> dict:
    inputs = summary["inputs"]
    checks = {
        "mesh_exists": bool(summary["mesh_status"]["exists"]),
        "mesh_not_placeholder": not bool(summary["mesh_status"]["looks_placeholder"]),
        "first_frame_not_undistorted": inputs["first_frame_undistorted"] is False,
        "rgb_depth_shape_match": inputs["rgb_shape"][:2] == inputs["depth_shape"],
        "depth_mask_shape_match": inputs["depth_shape"] == inputs["mask_shape"],
        "mask_nonempty": inputs["mask_area_px"] > 0,
        "intrinsics_source_is_pinhole_calibration": inputs["intrinsics"].get("source") == "pinhole_calibration",
        "depth_mask_positive_ratio_ok": inputs["depth_stats_mask"] is not None
        and inputs["depth_stats_mask"].get("positive_finite_ratio", 0.0) >= 0.95,
    }
    if require_pose:
        pose = summary.get("pose")
        checks["pose_present"] = pose is not None
        if pose is not None:
            checks["pose_bottom_row_ok"] = np.allclose(pose["bottom_row"], [0, 0, 0, 1], atol=1e-6)
            checks["pose_rotation_det_ok"] = abs(pose["rotation_det"] - 1.0) < 1e-2
            checks["pose_rotation_orthogonality_ok"] = pose["rotation_orthogonality_error_fro"] < 1e-2
            checks["pose_translation_positive_z"] = pose["translation_xyz_m"][2] > 0
            checks["pose_translation_reasonable_norm"] = 0.05 < pose["translation_norm_m"] < 5.0
    return checks


def _raise_on_failed_checks(checks: dict, allow_placeholder_mesh: bool = False) -> None:
    ignored = {"mesh_exists", "mesh_not_placeholder"} if allow_placeholder_mesh else set()
    failures = [name for name, ok in checks.items() if ok is False and name not in ignored]
    if failures:
        raise SystemExit(f"FoundationPose debug checks failed: {failures}")


if __name__ == "__main__":
    main()
