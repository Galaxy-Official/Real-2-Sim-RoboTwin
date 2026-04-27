#!/usr/bin/env python3
"""Run FoundationPose registration on a single RGB/depth/mask frame."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_FOUNDATIONPOSE_WEIGHTS = (
    Path("weights/2023-10-28-18-33-37/config.yml"),
    Path("weights/2023-10-28-18-33-37/model_best.pth"),
    Path("weights/2024-01-11-20-02-45/config.yml"),
    Path("weights/2024-01-11-20-02-45/model_best.pth"),
)

try:
    from ..坐标系转换 import matrix_to_pose6d, matrix_to_pose7d_wxyz
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    REPLAY_POLICY_DIR = THIS_DIR.parent
    if str(REPLAY_POLICY_DIR) not in sys.path:
        sys.path.insert(0, str(REPLAY_POLICY_DIR))
    from 坐标系转换 import matrix_to_pose6d, matrix_to_pose7d_wxyz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, help="Path to the cloned FoundationPose repo.")
    parser.add_argument("--rgb", required=True, help="Single RGB image path.")
    parser.add_argument("--depth", required=True, help="Depth file path (.npy / .png / .tiff).")
    parser.add_argument("--mask", required=True, help="Binary mask file path (.png / .npy).")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON or txt path.")
    parser.add_argument("--mesh", required=True, help="Object mesh path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--debug-dir", default=None, help="Optional FoundationPose debug dir.")
    parser.add_argument("--est-refine-iter", default=5, type=int, help="Registration refinement iterations.")
    parser.add_argument("--depth-scale", default=1.0, type=float, help="Scale multiplier applied after loading depth.")
    parser.add_argument("--debug", default=0, type=int, help="Forwarded to FoundationPose when supported.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not repo_root.is_dir():
        raise FileNotFoundError(f"FoundationPose repo not found: {repo_root}")
    _check_foundationpose_weights(repo_root)

    rgb = _load_rgb(Path(args.rgb).resolve())
    depth = _load_depth(Path(args.depth).resolve(), depth_scale=args.depth_scale)
    mask = _load_mask(Path(args.mask).resolve())
    K = _load_intrinsics(Path(args.intrinsics).resolve())

    if depth.shape != mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: depth {depth.shape}, mask {mask.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth shape mismatch: rgb {rgb.shape[:2]}, depth {depth.shape}")
    if mask.sum() == 0:
        raise ValueError("Input mask is empty; FoundationPose registration cannot proceed.")

    pose = _run_registration(
        repo_root=repo_root,
        mesh_path=Path(args.mesh).resolve(),
        K=K,
        rgb=rgb,
        depth=depth,
        mask=mask,
        est_refine_iter=args.est_refine_iter,
        debug_dir=Path(args.debug_dir).resolve() if args.debug_dir else None,
        debug=args.debug,
    )

    payload = {
        "matrix": np.asarray(pose, dtype=float).tolist(),
        "T_cam_obj": np.asarray(pose, dtype=float).tolist(),
        "pose6d_rotvec": matrix_to_pose6d(pose).tolist(),
        "pose7d_wxyz": matrix_to_pose7d_wxyz(pose).tolist(),
        "rgb_path": str(Path(args.rgb).resolve()),
        "depth_path": str(Path(args.depth).resolve()),
        "mask_path": str(Path(args.mask).resolve()),
        "intrinsics_path": str(Path(args.intrinsics).resolve()),
        "mesh_path": str(Path(args.mesh).resolve()),
        "intrinsics_matrix": np.asarray(K, dtype=float).tolist(),
        "est_refine_iter": args.est_refine_iter,
        "depth_scale": args.depth_scale,
        "mask_pixel_count": int(mask.sum()),
    }
    if args.debug_dir:
        payload["debug_dir"] = str(Path(args.debug_dir).resolve())

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run_foundationpose_once] Saved pose json to {output_path}")


def _check_foundationpose_weights(repo_root: Path) -> None:
    missing: list[str] = []
    for rel_path in REQUIRED_FOUNDATIONPOSE_WEIGHTS:
        path = repo_root / rel_path
        if not path.is_file():
            missing.append(str(rel_path))
        elif path.suffix == ".pth" and path.stat().st_size < 1024 * 1024:
            missing.append(f"{rel_path} (too small, likely not a real checkpoint)")
    if missing:
        missing_text = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "FoundationPose pretrained weights are missing or incomplete:\n"
            f"{missing_text}\n"
            "Download them into third_party/FoundationPose/weights. "
            "From the RoboTwin root, run:\n"
            "  bash policy/Replay_Policy/auto_init/setup_foundationpose_weights.sh "
            "--foundationpose-root third_party/FoundationPose"
        )


def _load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"RGB image not found: {path}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("run_foundationpose_once.py requires Pillow in the active environment.") from exc

    rgb = np.asarray(Image.open(path).convert("RGB"))
    return rgb


def _load_depth(path: Path, depth_scale: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Depth file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    else:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "run_foundationpose_once.py requires opencv-python to read non-npy depth files."
            ) from exc
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {path}")
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth array, got shape {depth.shape}")
    return depth * float(depth_scale)


def _load_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("run_foundationpose_once.py requires Pillow to read mask images.") from exc
        mask = np.asarray(Image.open(path))
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=-1)
    else:
        mask = mask > 0
    return mask.astype(np.uint8)


def _load_intrinsics(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if {"fx", "fy", "cx", "cy"}.issubset(payload):
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = float(payload["fx"])
            K[1, 1] = float(payload["fy"])
            K[0, 2] = float(payload["cx"])
            K[1, 2] = float(payload["cy"])
            return K
        for key in ("K", "camera_matrix", "intrinsics_matrix", "matrix"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float64).reshape(3, 3)
        raise ValueError(f"Unsupported intrinsics JSON format: {path}")

    values = np.loadtxt(path, dtype=np.float64)
    if values.size == 4:
        fx, fy, cx, cy = values.reshape(4)
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _run_registration(
    repo_root: Path,
    mesh_path: Path,
    K: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    est_refine_iter: int,
    debug_dir: Path | None,
    debug: int,
) -> np.ndarray:
    sys.path.insert(0, str(repo_root))
    try:
        import trimesh
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    except Exception as exc:
        hint = ""
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            hint = f" Missing Python module: {exc.name}."
            if exc.name == "pytorch3d":
                hint += (
                    " Install PyTorch3D in the FoundationPose environment, "
                    "for example from the official pytorch3d stable source build."
                )
        raise ImportError(
            "Failed to import FoundationPose runtime. "
            "Check that the FoundationPose environment is activated and build_all.sh/build_all_conda.sh has completed."
            + hint
        ) from exc

    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh = _make_mesh_visual_foundationpose_compatible(mesh, trimesh)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    init_kwargs = {
        "model_pts": np.asarray(mesh.vertices),
        "model_normals": np.asarray(mesh.vertex_normals),
        "mesh": mesh,
        "scorer": scorer,
        "refiner": refiner,
    }
    init_sig = inspect.signature(FoundationPose)
    if "debug_dir" in init_sig.parameters:
        init_kwargs["debug_dir"] = str(debug_dir) if debug_dir is not None else None
    if "debug" in init_sig.parameters:
        init_kwargs["debug"] = debug
    if "glctx" in init_sig.parameters:
        init_kwargs["glctx"] = glctx
    estimator = FoundationPose(**init_kwargs)

    register_sig = inspect.signature(estimator.register)
    register_kwargs = {}
    if "K" in register_sig.parameters:
        register_kwargs["K"] = np.asarray(K, dtype=np.float64)
    if "rgb" in register_sig.parameters:
        register_kwargs["rgb"] = np.asarray(rgb, dtype=np.uint8)
    if "depth" in register_sig.parameters:
        register_kwargs["depth"] = np.asarray(depth, dtype=np.float32)
    if "ob_mask" in register_sig.parameters:
        register_kwargs["ob_mask"] = np.asarray(mask, dtype=np.uint8)
    if "iteration" in register_sig.parameters:
        register_kwargs["iteration"] = est_refine_iter

    pose = estimator.register(**register_kwargs)
    if isinstance(pose, tuple):
        pose = pose[0]
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
    return pose


def _make_mesh_visual_foundationpose_compatible(mesh, trimesh_module):
    """FoundationPose expects TextureVisuals materials to expose material.image."""
    texture_visuals_cls = getattr(trimesh_module.visual.texture, "TextureVisuals", None)
    if texture_visuals_cls is None or not isinstance(mesh.visual, texture_visuals_cls):
        return mesh

    material = getattr(mesh.visual, "material", None)
    if getattr(material, "image", None) is not None:
        return mesh

    color = _extract_material_rgba(material)
    vertex_colors = np.tile(color.reshape(1, 4), (len(mesh.vertices), 1))
    try:
        color_visuals_cls = trimesh_module.visual.ColorVisuals
    except AttributeError:
        color_visuals_cls = trimesh_module.visual.color.ColorVisuals
    mesh.visual = color_visuals_cls(mesh=mesh, vertex_colors=vertex_colors)
    print(
        "[run_foundationpose_once] Converted texture material without image "
        f"({type(material).__name__}) to vertex colors {color.tolist()}"
    )
    return mesh


def _extract_material_rgba(material) -> np.ndarray:
    for attr_name in ("baseColorFactor", "main_color", "diffuse", "ambient"):
        try:
            value = getattr(material, attr_name, None)
        except Exception:
            value = None
        if value is None:
            continue
        color = _coerce_rgba(value)
        if color is not None:
            return color
    return np.array([128, 128, 128, 255], dtype=np.uint8)


def _coerce_rgba(value) -> np.ndarray | None:
    try:
        color = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if color.size < 3 or not np.all(np.isfinite(color[:3])):
        return None
    if color.size == 3:
        color = np.concatenate([color, np.array([1.0 if np.nanmax(color) <= 1.0 else 255.0])])
    else:
        color = color[:4]
    if np.nanmax(color) <= 1.0:
        color = color * 255.0
    return np.clip(np.round(color), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    main()
