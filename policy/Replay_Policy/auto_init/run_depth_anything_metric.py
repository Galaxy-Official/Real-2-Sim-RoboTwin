#!/usr/bin/env python3
"""Run Depth Anything 3 on a single RGB image and save metric depth as .npy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional path to the cloned Depth-Anything-3 repo. Used when it is not pip-installed.",
    )
    parser.add_argument("--image", required=True, help="Single RGB image path.")
    parser.add_argument("--output", required=True, help="Output .npy depth path.")
    parser.add_argument(
        "--intrinsics",
        required=True,
        help="FoundationPose intrinsics JSON. Used to convert DA3METRIC-LARGE output to meters.",
    )
    parser.add_argument(
        "--model-dir",
        default="depth-anything/da3metric-large",
        help="Hugging Face model id or local model directory for Depth Anything 3.",
    )
    parser.add_argument(
        "--metric-scale",
        default="da3metric",
        choices=["da3metric", "already_metric"],
        help="DA3METRIC-LARGE needs metric_depth=focal*net_output/300; DA3NESTED output is already meters.",
    )
    parser.add_argument(
        "--process-res",
        default=504,
        type=int,
        help="Depth Anything 3 processing resolution.",
    )
    parser.add_argument(
        "--process-res-method",
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
    )
    parser.add_argument("--device", default=None, help="cuda / cpu / mps. Defaults to best available.")
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional JSON file recording which repo/checkpoint/mode were used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()
    intrinsics_path = Path(args.intrinsics).resolve()

    if repo_root is not None and not repo_root.is_dir():
        raise FileNotFoundError(f"Depth Anything repo not found: {repo_root}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Intrinsics JSON not found: {intrinsics_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if repo_root is not None:
        _add_depth_anything_3_to_sys_path(repo_root)

    torch, DepthAnything3 = _import_runtime()
    device = _resolve_device(torch, args.device)
    K = _load_intrinsics_matrix(intrinsics_path)
    focal_px = float((K[0, 0] + K[1, 1]) * 0.5)

    model = DepthAnything3.from_pretrained(args.model_dir).to(device)
    if hasattr(model, "eval"):
        model = model.eval()

    with torch.inference_mode():
        prediction = model.inference(
            image=[str(image_path)],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )

    depth = _extract_single_depth(prediction)
    depth = _scale_depth(depth, focal_px=focal_px, metric_scale=args.metric_scale)
    depth = _resize_depth_to_image(depth, image_path)
    np.save(output_path, depth)

    if args.meta_output:
        meta_path = Path(args.meta_output).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_root": None if repo_root is None else str(repo_root),
            "image_path": str(image_path),
            "output_path": str(output_path),
            "intrinsics_path": str(intrinsics_path),
            "model_dir": args.model_dir,
            "metric_scale": args.metric_scale,
            "focal_px": focal_px,
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "device": device,
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
        }
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "[run_depth_anything_metric] "
        f"Saved Depth Anything 3 metric depth to {output_path} using {args.model_dir}"
    )


def _import_runtime():
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "run_depth_anything_metric.py requires Depth Anything 3 and torch in the active environment. "
            "Install the official Depth-Anything-3 repo, for example with `pip install -e third_party/Depth-Anything-3`."
        ) from exc
    return torch, DepthAnything3


def _add_depth_anything_3_to_sys_path(repo_root: Path) -> None:
    for candidate in (repo_root / "src", repo_root):
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)


def _resolve_device(torch, requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_intrinsics_matrix(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "K" in payload:
        return np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(payload["fx"])
    K[1, 1] = float(payload["fy"])
    K[0, 2] = float(payload["cx"])
    K[1, 2] = float(payload["cy"])
    return K


def _extract_single_depth(prediction) -> np.ndarray:
    if not hasattr(prediction, "depth"):
        raise ValueError("Depth Anything 3 prediction does not contain a depth attribute.")
    depth = prediction.depth
    if isinstance(depth, (list, tuple)):
        if len(depth) != 1:
            raise ValueError(f"Expected one depth map, got {len(depth)}")
        depth = depth[0]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"Expected one depth map, got shape {depth.shape}")
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")
    return depth


def _scale_depth(depth: np.ndarray, focal_px: float, metric_scale: str) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if metric_scale == "da3metric":
        if focal_px <= 0:
            raise ValueError(f"Invalid focal length for DA3METRIC scaling: {focal_px}")
        depth = depth * np.float32(focal_px / 300.0)
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _resize_depth_to_image(depth: np.ndarray, image_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to align Depth Anything 3 output to the input image size.") from exc

    with Image.open(image_path) as image:
        width, height = image.size
    if depth.shape == (height, width):
        return depth.astype(np.float32)

    resized = Image.fromarray(depth.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


if __name__ == "__main__":
    main()
