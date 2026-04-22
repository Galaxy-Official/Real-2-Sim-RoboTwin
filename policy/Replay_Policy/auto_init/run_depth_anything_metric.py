#!/usr/bin/env python3
"""Run Depth Anything V2 on a single RGB image and save depth as .npy."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, help="Path to the cloned Depth-Anything-V2 repo.")
    parser.add_argument("--image", required=True, help="Single RGB image path.")
    parser.add_argument("--output", required=True, help="Output .npy depth path.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--encoder", default="vitl", choices=sorted(MODEL_CONFIGS))
    parser.add_argument(
        "--mode",
        default="metric",
        choices=["metric", "relative", "auto"],
        help="Prefer metric-depth weights for FoundationPose. 'auto' tries metric first.",
    )
    parser.add_argument(
        "--metric-dataset",
        default="hypersim",
        help="Metric checkpoint family name, e.g. hypersim or vkitti.",
    )
    parser.add_argument("--input-size", default=518, type=int)
    parser.add_argument("--max-depth", default=20.0, type=float)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps. Defaults to best available.")
    parser.add_argument(
        "--meta-output",
        default=None,
        help="Optional JSON file recording which repo/checkpoint/mode were used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Depth Anything repo not found: {repo_root}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch, cv2 = _import_runtime()
    device = _resolve_device(torch, args.device)
    model_cls, loaded_mode = _load_model_class(repo_root, args.mode)
    checkpoint_path = _resolve_checkpoint_path(repo_root, args, loaded_mode)
    model = _build_model(model_cls, loaded_mode, args.encoder, args.max_depth)

    state_dict = _load_checkpoint(torch, checkpoint_path)
    _load_model_state(model, state_dict, checkpoint_path)

    model = model.to(device).eval()
    raw_img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_img is None:
        raise RuntimeError(f"Failed to read RGB image with OpenCV: {image_path}")

    try:
        depth = model.infer_image(raw_img, input_size=args.input_size)
    except TypeError:
        depth = model.infer_image(raw_img)

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")
    np.save(output_path, depth)

    if args.meta_output:
        meta_path = Path(args.meta_output).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "repo_root": str(repo_root),
            "image_path": str(image_path),
            "output_path": str(output_path),
            "checkpoint_path": str(checkpoint_path),
            "encoder": args.encoder,
            "mode": loaded_mode,
            "input_size": args.input_size,
            "max_depth": args.max_depth,
            "device": device,
            "depth_shape": list(depth.shape),
            "depth_dtype": str(depth.dtype),
            "depth_min": float(np.nanmin(depth)),
            "depth_max": float(np.nanmax(depth)),
        }
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        "[run_depth_anything_metric] "
        f"Saved {loaded_mode} depth to {output_path} using {checkpoint_path.name}"
    )


def _import_runtime():
    try:
        import torch
        import cv2
    except ImportError as exc:
        raise ImportError(
            "run_depth_anything_metric.py requires torch and opencv-python in the active environment."
        ) from exc
    return torch, cv2


def _resolve_device(torch, requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _purge_depth_anything_modules() -> None:
    for name in list(sys.modules):
        if name == "depth_anything_v2" or name.startswith("depth_anything_v2."):
            del sys.modules[name]


def _load_model_class(repo_root: Path, mode: str):
    attempts: list[tuple[str, Path]] = []
    if mode in {"metric", "auto"}:
        attempts.append(("metric", repo_root / "metric_depth"))
    if mode in {"relative", "auto"}:
        attempts.append(("relative", repo_root))

    errors: list[str] = []
    for import_mode, import_root in attempts:
        if not import_root.is_dir():
            errors.append(f"{import_mode}: missing import root {import_root}")
            continue
        _purge_depth_anything_modules()
        sys.path.insert(0, str(import_root))
        try:
            module = importlib.import_module("depth_anything_v2.dpt")
            model_cls = getattr(module, "DepthAnythingV2")
            return model_cls, import_mode
        except Exception as exc:
            errors.append(f"{import_mode}: {exc}")
        finally:
            if sys.path and sys.path[0] == str(import_root):
                sys.path.pop(0)

    raise ImportError(
        "Failed to import DepthAnythingV2. "
        f"Attempted modes: {', '.join(m for m, _ in attempts)}. Errors: {' | '.join(errors)}"
    )


def _build_model(model_cls, loaded_mode: str, encoder: str, max_depth: float):
    model_cfg = MODEL_CONFIGS[encoder].copy()
    if loaded_mode == "metric":
        try:
            return model_cls(**model_cfg, max_depth=max_depth)
        except TypeError:
            pass
    return model_cls(**model_cfg)


def _resolve_checkpoint_path(repo_root: Path, args: argparse.Namespace, loaded_mode: str) -> Path:
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Explicit checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    candidates: list[Path] = []
    if loaded_mode == "metric":
        candidates.extend(
            [
                repo_root / "metric_depth" / "checkpoints" / f"depth_anything_v2_metric_{args.metric_dataset}_{args.encoder}.pth",
                repo_root / "checkpoints" / f"depth_anything_v2_metric_{args.metric_dataset}_{args.encoder}.pth",
            ]
        )
    else:
        candidates.extend(
            [
                repo_root / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth",
                repo_root / f"depth_anything_v2_{args.encoder}.pth",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find a Depth Anything checkpoint automatically. "
        f"Tried: {[str(path) for path in candidates]}. "
        "Pass --checkpoint explicitly if your file lives elsewhere."
    )


def _load_checkpoint(torch, checkpoint_path: Path):
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
    return payload


def _load_model_state(model, state_dict, checkpoint_path: Path) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Depth Anything checkpoint does not match the selected model config. "
            f"Checkpoint: {checkpoint_path}"
        ) from exc


if __name__ == "__main__":
    main()
