#!/usr/bin/env python3
"""Generate a high-quality FoundationPose object mask from the wrist first frame."""

from __future__ import annotations

import argparse
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

from auto_init.path_utils import resolve_cli_path, resolve_repo_path
from replay_lerobot_loader import extract_first_frame, resolve_episode_video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a binary object mask for FoundationPose. Provide a tight bbox "
            "and optionally positive/negative click points for best quality."
        )
    )
    parser.add_argument("--config", default=str(REPLAY_POLICY_DIR / "deploy_policy.yml"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--video-key", default=None)
    parser.add_argument("--image", default=None, help="Optional existing first-frame RGB image.")
    parser.add_argument("--output", default=None, help="Mask output path. Defaults to auto_init.mask.template.")
    parser.add_argument("--prompt-json", default=None, help="Prompt JSON containing bbox/points.")
    parser.add_argument("--write-prompt-template", default=None, help="Write a prompt template JSON and exit.")
    parser.add_argument("--grid-step", type=int, default=40, help="Pixel spacing for the coordinate guide image.")
    parser.add_argument("--box", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument(
        "--positive-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help="Point inside the target object. Can be repeated.",
    )
    parser.add_argument(
        "--negative-point",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help="Point outside the target object. Can be repeated.",
    )
    parser.add_argument("--backend", choices=("auto", "sam2", "sam"), default="auto")
    parser.add_argument("--checkpoint", default=None, help="SAM/SAM2 checkpoint path.")
    parser.add_argument("--model-type", default="vit_h", help="SAM v1 model type: vit_h, vit_l, or vit_b.")
    parser.add_argument("--sam2-config", default=None, help="SAM2 model config, e.g. sam2_hiera_l.yaml.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multimask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--keep-components", type=int, default=1)
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=0)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
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
    debug_dir = cache_dir / "mask_generation_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_path = _resolve_or_extract_frame(args, auto_init_cfg, video_key, debug_dir)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    output_path = _resolve_output_path(args, config)

    if args.write_prompt_template:
        template_path = resolve_cli_path(args.write_prompt_template)
        coordinate_guide_path = debug_dir / f"episode_{args.episode_index:06d}_coordinate_guide.png"
        _write_coordinate_guide(image, coordinate_guide_path, grid_step=args.grid_step)
        _write_prompt_template(
            path=template_path,
            image_path=image_path,
            width=image.shape[1],
            height=image.shape[0],
            coordinate_guide_path=coordinate_guide_path,
        )
        print(f"[generate_sam_mask] Wrote prompt template to {template_path.resolve()}")
        print(f"[generate_sam_mask] Wrote coordinate guide to {coordinate_guide_path.resolve()}")
        return

    prompts = _load_prompts(args)
    _validate_prompts(prompts, image.shape[1], image.shape[0])
    raw_masks, raw_scores, backend_name = _predict_masks(args, image, prompts)
    candidate_metrics = [
        _score_candidate(mask=mask, score=float(raw_scores[index]), prompts=prompts)
        for index, mask in enumerate(raw_masks)
    ]
    best_index = int(np.argmax([item["selection_score"] for item in candidate_metrics]))
    raw_mask = raw_masks[best_index].astype(bool)
    if args.invert:
        raw_mask = ~raw_mask

    final_mask = raw_mask if args.no_postprocess else _postprocess_mask(raw_mask, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_mask.astype(np.uint8) * 255, mode="L").save(output_path)

    prefix = f"episode_{args.episode_index:06d}"
    raw_candidate_path = debug_dir / f"{prefix}_sam_raw_candidate.png"
    overlay_path = debug_dir / f"{prefix}_sam_mask_overlay.png"
    prompt_preview_path = debug_dir / f"{prefix}_sam_prompt_preview.png"
    summary_path = debug_dir / f"{prefix}_sam_mask_summary.json"
    Image.fromarray(raw_mask.astype(np.uint8) * 255, mode="L").save(raw_candidate_path)
    _write_overlay(image, final_mask, prompts, overlay_path, alpha=args.overlay_alpha)
    _write_prompt_preview(image, prompts, prompt_preview_path)

    summary = {
        "episode_index": args.episode_index,
        "backend": backend_name,
        "image_path": str(image_path.resolve()),
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "mask_output_path": str(output_path.resolve()),
        "debug_dir": str(debug_dir.resolve()),
        "prompt": prompts,
        "selected_candidate_index": best_index,
        "selected_candidate": candidate_metrics[best_index],
        "all_candidates": candidate_metrics,
        "postprocess": {
            "enabled": not args.no_postprocess,
            "min_area": args.min_area,
            "keep_components": args.keep_components,
            "close_kernel": args.close_kernel,
            "open_kernel": args.open_kernel,
            "dilate": args.dilate,
            "erode": args.erode,
            "invert": args.invert,
        },
        "final_mask_stats": _mask_stats(final_mask),
        "outputs": {
            "raw_candidate": str(raw_candidate_path.resolve()),
            "overlay": str(overlay_path.resolve()),
            "prompt_preview": str(prompt_preview_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
        "next_check": (
            "Inspect the overlay. The mask should include the full target object and exclude "
            "the robot gripper, table, and neighboring blocks before running FoundationPose."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[generate_sam_mask] Wrote mask to {output_path.resolve()}")
    print(f"[generate_sam_mask] Wrote overlay to {overlay_path.resolve()}")


def _resolve_or_extract_frame(args: argparse.Namespace, auto_init_cfg: dict, video_key: str, debug_dir: Path) -> Path:
    if args.image:
        image_path = resolve_cli_path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"First-frame image not found: {image_path}")
        return image_path

    video_path = resolve_episode_video_path(args.data_dir, args.episode_index, video_key=video_key)
    image_path = debug_dir / f"episode_{args.episode_index:06d}_wrist_first_frame.png"
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to extract the wrist first frame.")
    extract_first_frame(video_path, image_path)
    return image_path


def _resolve_output_path(args: argparse.Namespace, config: dict) -> Path:
    if args.output:
        return resolve_cli_path(args.output)
    mask_cfg = config.get("auto_init", {}).get("mask", {})
    mode = mask_cfg.get("mode", "file_template")
    if mode == "file_template":
        template = mask_cfg.get("template")
        if not template:
            raise ValueError("auto_init.mask.template must be set when mask mode is file_template")
        return Path(
            template.format(
                data_dir=args.data_dir,
                episode_index=args.episode_index,
            )
        )
    if mode == "explicit":
        return resolve_repo_path(mask_cfg["path"])
    raise ValueError(f"Unsupported mask mode: {mode}")


def _write_prompt_template(
    path: Path,
    image_path: Path,
    width: int,
    height: int,
    coordinate_guide_path: Path,
) -> None:
    payload = {
        "image_path": str(image_path.resolve()),
        "coordinate_guide_path": str(coordinate_guide_path.resolve()),
        "image_size": {"width": width, "height": height},
        "box": [100, 100, 300, 300],
        "positive_points": [[200, 200]],
        "negative_points": [[50, 50]],
        "notes": [
            "Replace box with a tight rectangle around the target object: [x1, y1, x2, y2].",
            "Put at least one positive point well inside the object.",
            "Add negative points on nearby distractors such as gripper fingers or other blocks.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_coordinate_guide(image: np.ndarray, output_path: Path, grid_step: int) -> None:
    grid_step = max(10, int(grid_step))
    guide = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(guide, "RGBA")
    width, height = guide.size

    for x in range(0, width, grid_step):
        draw.line((x, 0, x, height), fill=(255, 255, 0, 120), width=1)
        draw.text((x + 3, 3), str(x), fill=(255, 255, 0, 255))
    for y in range(0, height, grid_step):
        draw.line((0, y, width, y), fill=(0, 255, 255, 120), width=1)
        draw.text((3, y + 3), str(y), fill=(0, 255, 255, 255))

    draw.rectangle((0, 0, min(width - 1, 250), 24), fill=(0, 0, 0, 170))
    draw.text((6, 5), "x: yellow vertical, y: cyan horizontal", fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    guide.save(output_path)


def _load_prompts(args: argparse.Namespace) -> dict:
    payload: dict[str, Any] = {}
    if args.prompt_json:
        payload = json.loads(resolve_cli_path(args.prompt_json).read_text(encoding="utf-8"))
    if args.box is not None:
        payload["box"] = [float(value) for value in args.box]
    if args.positive_point:
        payload["positive_points"] = [[float(x), float(y)] for x, y in args.positive_point]
    if args.negative_point:
        payload["negative_points"] = [[float(x), float(y)] for x, y in args.negative_point]

    prompts = {
        "box": payload.get("box"),
        "positive_points": payload.get("positive_points", []),
        "negative_points": payload.get("negative_points", []),
    }
    if prompts["box"] is None and not prompts["positive_points"]:
        raise ValueError(
            "Provide at least --box or --positive-point. For a template, run with "
            "--write-prompt-template init_meta/cache/mask_generation_debug/prompt.json"
        )
    return prompts


def _validate_prompts(prompts: dict, width: int, height: int) -> None:
    if prompts["box"] is not None:
        x1, y1, x2, y2 = prompts["box"]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid box {prompts['box']} for image size {(width, height)}")
    for label, points in (("positive_points", prompts["positive_points"]), ("negative_points", prompts["negative_points"])):
        for point in points:
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Invalid {label} point {point} for image size {(width, height)}")


def _predict_masks(args: argparse.Namespace, image: np.ndarray, prompts: dict) -> tuple[list[np.ndarray], list[float], str]:
    errors = []
    if args.backend in ("auto", "sam2"):
        try:
            return _predict_masks_sam2(args, image, prompts)
        except Exception as exc:  # noqa: BLE001 - auto backend should report both import paths.
            errors.append(f"sam2: {type(exc).__name__}: {exc}")
            if args.backend == "sam2":
                raise
    if args.backend in ("auto", "sam"):
        try:
            return _predict_masks_sam(args, image, prompts)
        except Exception as exc:  # noqa: BLE001 - auto backend should report both import paths.
            errors.append(f"segment_anything: {type(exc).__name__}: {exc}")
            if args.backend == "sam":
                raise
    raise ImportError(
        "No SAM backend is available. Install SAM2 or segment-anything, provide a checkpoint, "
        "and rerun. Backend errors: " + " | ".join(errors)
    )


def _predict_masks_sam2(args: argparse.Namespace, image: np.ndarray, prompts: dict) -> tuple[list[np.ndarray], list[float], str]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for SAM2.")
    if not args.sam2_config:
        raise ValueError("--sam2-config is required for SAM2, e.g. sam2_hiera_l.yaml.")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError("SAM2 Python package is not importable.") from exc

    model = build_sam2(args.sam2_config, args.checkpoint, device=args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=_point_coords(prompts),
        point_labels=_point_labels(prompts),
        box=None if prompts["box"] is None else np.asarray(prompts["box"], dtype=np.float32),
        multimask_output=args.multimask,
    )
    return _normalize_predictor_output(masks, scores), [float(score) for score in np.asarray(scores).reshape(-1)], "sam2"


def _predict_masks_sam(args: argparse.Namespace, image: np.ndarray, prompts: dict) -> tuple[list[np.ndarray], list[float], str]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for SAM v1.")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError("segment-anything Python package is not importable.") from exc

    if args.model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type {args.model_type!r}. Available: {sorted(sam_model_registry)}")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        point_coords=_point_coords(prompts),
        point_labels=_point_labels(prompts),
        box=None if prompts["box"] is None else np.asarray(prompts["box"], dtype=np.float32),
        multimask_output=args.multimask,
    )
    return _normalize_predictor_output(masks, scores), [float(score) for score in np.asarray(scores).reshape(-1)], "segment_anything"


def _point_coords(prompts: dict) -> np.ndarray | None:
    points = prompts["positive_points"] + prompts["negative_points"]
    if not points:
        return None
    return np.asarray(points, dtype=np.float32)


def _point_labels(prompts: dict) -> np.ndarray | None:
    if not prompts["positive_points"] and not prompts["negative_points"]:
        return None
    labels = [1] * len(prompts["positive_points"]) + [0] * len(prompts["negative_points"])
    return np.asarray(labels, dtype=np.int32)


def _normalize_predictor_output(masks: Any, scores: Any) -> list[np.ndarray]:
    masks_array = np.asarray(masks)
    if masks_array.ndim == 2:
        masks_array = masks_array[None, ...]
    if masks_array.ndim != 3:
        raise ValueError(f"Expected predictor masks with shape (N,H,W), got {masks_array.shape}")
    scores_array = np.asarray(scores).reshape(-1)
    if masks_array.shape[0] != scores_array.shape[0]:
        raise ValueError(f"Mask count {masks_array.shape[0]} does not match score count {scores_array.shape[0]}")
    return [masks_array[index].astype(bool) for index in range(masks_array.shape[0])]


def _score_candidate(mask: np.ndarray, score: float, prompts: dict) -> dict:
    stats = _mask_stats(mask)
    selection_score = score
    box_metrics = None
    if prompts["box"] is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in prompts["box"]]
        box_mask = np.zeros(mask.shape, dtype=bool)
        box_mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = True
        mask_area = max(int(mask.sum()), 1)
        inside = int(np.logical_and(mask, box_mask).sum())
        inside_ratio = inside / mask_area
        box_fill_ratio = inside / max(int(box_mask.sum()), 1)
        outside_ratio = 1.0 - inside_ratio
        selection_score += 0.20 * inside_ratio - 0.35 * outside_ratio
        if box_fill_ratio < 0.05:
            selection_score -= 0.50
        box_metrics = {
            "inside_box_ratio": float(inside_ratio),
            "outside_box_ratio": float(outside_ratio),
            "box_fill_ratio": float(box_fill_ratio),
        }
    point_metrics = _point_metrics(mask, prompts)
    selection_score += 0.10 * point_metrics["positive_points_inside_ratio"]
    selection_score -= 0.25 * point_metrics["negative_points_inside_ratio"]
    return {
        "model_score": float(score),
        "selection_score": float(selection_score),
        "mask_stats": stats,
        "box_metrics": box_metrics,
        "point_metrics": point_metrics,
    }


def _point_metrics(mask: np.ndarray, prompts: dict) -> dict:
    positive_inside = _count_points_inside(mask, prompts["positive_points"])
    negative_inside = _count_points_inside(mask, prompts["negative_points"])
    pos_total = len(prompts["positive_points"])
    neg_total = len(prompts["negative_points"])
    return {
        "positive_points_inside": positive_inside,
        "positive_points_total": pos_total,
        "positive_points_inside_ratio": 1.0 if pos_total == 0 else positive_inside / pos_total,
        "negative_points_inside": negative_inside,
        "negative_points_total": neg_total,
        "negative_points_inside_ratio": 0.0 if neg_total == 0 else negative_inside / neg_total,
    }


def _count_points_inside(mask: np.ndarray, points: list[list[float]]) -> int:
    count = 0
    h, w = mask.shape
    for x, y in points:
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < w and 0 <= iy < h and mask[iy, ix]:
            count += 1
    return count


def _postprocess_mask(mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cv2 = _require_cv2_for_postprocess()
    result = mask.astype(np.uint8)
    if args.open_kernel > 0:
        result = _morph(cv2, result, cv2.MORPH_OPEN, args.open_kernel)
    if args.close_kernel > 0:
        result = _morph(cv2, result, cv2.MORPH_CLOSE, args.close_kernel)
    result = _fill_holes(cv2, result)
    result = _keep_components(cv2, result, keep=args.keep_components, min_area=args.min_area)
    if args.erode > 0:
        result = _morph(cv2, result, cv2.MORPH_ERODE, 2 * args.erode + 1)
    if args.dilate > 0:
        result = _morph(cv2, result, cv2.MORPH_DILATE, 2 * args.dilate + 1)
    return result.astype(bool)


def _require_cv2_for_postprocess():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for mask post-processing. Install it or pass --no-postprocess."
        ) from exc
    return cv2


def _morph(cv2: Any, mask: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel)


def _fill_holes(cv2: Any, mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    inv = (mask_u8 == 0).astype(np.uint8)
    padded = np.pad(inv, pad_width=1, mode="constant", constant_values=1)
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 0)
    holes = padded[1:-1, 1:-1] == 1
    return np.logical_or(mask_u8 > 0, holes).astype(np.uint8)


def _keep_components(cv2: Any, mask: np.ndarray, keep: int, min_area: int) -> np.ndarray:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return (mask > 0).astype(np.uint8)
    components = []
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))
    components.sort(reverse=True)
    selected = {label for _, label in components[: max(1, keep)]}
    return np.isin(labels, list(selected)).astype(np.uint8)


def _mask_stats(mask: np.ndarray) -> dict:
    mask_bool = mask.astype(bool)
    area = int(mask_bool.sum())
    h, w = mask_bool.shape
    if area == 0:
        return {
            "area_px": 0,
            "area_ratio": 0.0,
            "bbox_xyxy": None,
        }
    ys, xs = np.where(mask_bool)
    return {
        "area_px": area,
        "area_ratio": float(area / (h * w)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
    }


def _write_overlay(image: np.ndarray, mask: np.ndarray, prompts: dict, output_path: Path, alpha: float) -> None:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = image.copy().astype(np.float32)
    color = np.array([0, 255, 80], dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    _draw_prompts(draw, prompts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def _write_prompt_preview(image: np.ndarray, prompts: dict, output_path: Path) -> None:
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    _draw_prompts(draw, prompts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def _draw_prompts(draw: ImageDraw.ImageDraw, prompts: dict) -> None:
    if prompts["box"] is not None:
        draw.rectangle(prompts["box"], outline=(255, 255, 0), width=3)
    for point in prompts["positive_points"]:
        _draw_cross(draw, point, color=(0, 255, 0))
    for point in prompts["negative_points"]:
        _draw_cross(draw, point, color=(255, 0, 0))


def _draw_cross(draw: ImageDraw.ImageDraw, point: list[float], color: tuple[int, int, int]) -> None:
    x, y = point
    r = 6
    draw.line((x - r, y, x + r, y), fill=color, width=3)
    draw.line((x, y - r, x, y + r), fill=color, width=3)


if __name__ == "__main__":
    main()
