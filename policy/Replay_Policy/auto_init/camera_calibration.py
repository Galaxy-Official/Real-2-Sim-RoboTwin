"""Camera intrinsics and optional fisheye undistortion helpers for auto-init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def prepare_frame_and_intrinsics(
    auto_init_cfg: dict,
    frame_path: str | Path,
    cache_dir: str | Path,
    episode_index: int,
) -> dict:
    """Return the image path and pinhole intrinsics used by FoundationPose.

    If camera_calibration.type is pinhole, the calibration K is used directly
    with the raw frame. If fisheye undistortion is enabled, the returned image
    path points to the undistorted frame and intrinsics are the new pinhole K.
    Otherwise the original frame path and configured/raw K are returned.
    """

    frame_path = Path(frame_path)
    cache_dir = Path(cache_dir)
    intrinsics_cfg = auto_init_cfg.get("intrinsics", {})
    calib_cfg = auto_init_cfg.get("camera_calibration") or {}
    undistort_cfg = auto_init_cfg.get("undistort") or {}
    calib_type = _calibration_type(calib_cfg)
    calibration = _load_calibration(calib_cfg)

    if calibration is None:
        K = _manual_intrinsics_to_matrix(intrinsics_cfg)
        intrinsics_path = _write_intrinsics(
            cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
            K,
            source="manual",
        )
        return {
            "frame_path": str(frame_path),
            "raw_frame_path": str(frame_path),
            "intrinsics": _matrix_to_intrinsics_dict(K),
            "intrinsics_matrix": K.tolist(),
            "intrinsics_path": str(intrinsics_path),
            "calibration": None,
            "undistorted": False,
        }

    K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calibration["D"], dtype=np.float64).reshape(-1, 1)
    image_size = _read_image_size(frame_path)
    K = _maybe_scale_K_to_image(K, calib_cfg, image_size)

    if calib_type == "pinhole":
        if undistort_cfg.get("enabled", False):
            raise ValueError(
                "auto_init.camera_calibration.type=pinhole is incompatible with "
                "auto_init.undistort.enabled=true. Use type=fisheye for runtime "
                "undistortion, or set undistort.enabled=false."
            )
        intrinsics_path = _write_intrinsics(
            cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
            K,
            source="pinhole_calibration",
            raw_K=K,
            D=D,
            rms=calibration.get("rms"),
        )
        return {
            "frame_path": str(frame_path),
            "raw_frame_path": str(frame_path),
            "intrinsics": _matrix_to_intrinsics_dict(K),
            "intrinsics_matrix": K.tolist(),
            "intrinsics_path": str(intrinsics_path),
            "calibration": _calibration_metadata(K, D, calibration, calib_type),
            "undistorted": False,
        }

    if undistort_cfg.get("enabled", True):
        undistorted_path, new_K = undistort_image_file(
            input_path=frame_path,
            output_path=_format_output_path(
                undistort_cfg.get(
                    "frame_output_template",
                    "{cache_dir}/episode_{episode_index:06d}_wrist_undistorted.png",
                ),
                cache_dir=cache_dir,
                episode_index=episode_index,
            ),
            K=K,
            D=D,
            balance=float(undistort_cfg.get("balance", 0.0)),
            fov_scale=float(undistort_cfg.get("fov_scale", 1.0)),
            interpolation="linear",
        )
        intrinsics_path = _write_intrinsics(
            cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
            new_K,
            source="fisheye_undistorted",
            raw_K=K,
            D=D,
            rms=calibration.get("rms"),
        )
        return {
            "frame_path": str(undistorted_path),
            "raw_frame_path": str(frame_path),
            "intrinsics": _matrix_to_intrinsics_dict(new_K),
            "intrinsics_matrix": new_K.tolist(),
            "intrinsics_path": str(intrinsics_path),
            "calibration": _calibration_metadata(K, D, calibration, calib_type),
            "undistorted": True,
        }

    intrinsics_path = _write_intrinsics(
        cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
        K,
        source="fisheye_raw_not_undistorted",
        raw_K=K,
        D=D,
        rms=calibration.get("rms"),
    )
    return {
        "frame_path": str(frame_path),
        "raw_frame_path": str(frame_path),
        "intrinsics": _matrix_to_intrinsics_dict(K),
        "intrinsics_matrix": K.tolist(),
        "intrinsics_path": str(intrinsics_path),
        "calibration": _calibration_metadata(K, D, calibration, calib_type),
        "undistorted": False,
    }


def maybe_undistort_mask(
    auto_init_cfg: dict,
    mask_path: str | Path,
    cache_dir: str | Path,
    episode_index: int,
    calibration: dict | None,
) -> Path:
    undistort_cfg = auto_init_cfg.get("undistort", {})
    if not undistort_cfg.get("enabled", True):
        return Path(mask_path)
    if not undistort_cfg.get("mask", True):
        return Path(mask_path)
    if calibration is None:
        return Path(mask_path)

    K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calibration["D"], dtype=np.float64).reshape(-1, 1)
    output_path = _format_output_path(
        undistort_cfg.get(
            "mask_output_template",
            "{cache_dir}/episode_{episode_index:06d}_mask_undistorted.png",
        ),
        cache_dir=Path(cache_dir),
        episode_index=episode_index,
    )
    mask_output, _ = undistort_image_file(
        input_path=mask_path,
        output_path=output_path,
        K=K,
        D=D,
        balance=float(undistort_cfg.get("balance", 0.0)),
        fov_scale=float(undistort_cfg.get("fov_scale", 1.0)),
        interpolation="nearest",
        binary_output=True,
    )
    return mask_output


def undistort_image_file(
    input_path: str | Path,
    output_path: str | Path,
    K: np.ndarray,
    D: np.ndarray,
    balance: float = 0.0,
    fov_scale: float = 1.0,
    interpolation: str = "linear",
    binary_output: bool = False,
) -> tuple[Path, np.ndarray]:
    cv2 = _require_cv2()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.suffix.lower() == ".npy":
        image = np.load(input_path)
    else:
        read_flag = cv2.IMREAD_GRAYSCALE if binary_output else cv2.IMREAD_COLOR
        image = cv2.imread(str(input_path), read_flag)
    if image is None:
        raise RuntimeError(f"Failed to read image for undistortion: {input_path}")

    h, w = image.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64).reshape(4, 1),
        (w, h),
        np.eye(3),
        balance=balance,
        fov_scale=fov_scale,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64).reshape(4, 1),
        np.eye(3),
        new_K,
        (w, h),
        cv2.CV_16SC2,
    )
    interp = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    undistorted = cv2.remap(image, map1, map2, interpolation=interp)
    if binary_output:
        undistorted = (undistorted > 0).astype(np.uint8) * 255

    if output_path.suffix.lower() == ".npy":
        np.save(output_path, undistorted)
    else:
        cv2.imwrite(str(output_path), undistorted)
    return output_path, np.asarray(new_K, dtype=np.float64)


def _load_calibration(calib_cfg: dict) -> dict[str, Any] | None:
    path = calib_cfg.get("path")
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "K_new" not in data.files or "D_raw" not in data.files:
                raise KeyError(
                    f"Unsupported calibration npz layout in {path}. "
                    "Expected keys K_new / D_raw / rms."
                )
            return {
                "K": np.asarray(data["K_new"], dtype=np.float64),
                "D": np.asarray(data["D_raw"], dtype=np.float64),
                "rms": _as_optional_float(data["rms"]) if "rms" in data.files else None,
                "K_key": "K_new",
                "D_key": "D_raw",
                "path": str(path),
            }
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "K": np.asarray(payload["K"], dtype=np.float64),
            "D": np.asarray(payload["D"], dtype=np.float64),
            "rms": payload.get("rms"),
            "K_key": "K",
            "D_key": "D",
            "path": str(path),
        }
    raise ValueError(f"Unsupported camera calibration file type: {path}")


def _calibration_type(calib_cfg: dict) -> str:
    raw_type = str(calib_cfg.get("type", "fisheye")).strip().lower()
    aliases = {
        "fisheye": "fisheye",
        "opencv_fisheye": "fisheye",
        "pinhole": "pinhole",
        "pinhole_output": "pinhole",
        "opencv_pinhole": "pinhole",
    }
    if raw_type not in aliases:
        raise ValueError(
            f"Unsupported auto_init.camera_calibration.type={raw_type!r}. "
            "Expected 'pinhole' or 'fisheye'."
        )
    return aliases[raw_type]


def _calibration_metadata(K: np.ndarray, D: np.ndarray, calibration: dict[str, Any], calib_type: str) -> dict:
    return {
        "type": calib_type,
        "path": calibration.get("path"),
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "rms": _as_optional_float(calibration.get("rms")),
        "K_key": calibration.get("K_key"),
        "D_key": calibration.get("D_key"),
    }


def _manual_intrinsics_to_matrix(intrinsics_cfg: dict) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = float(intrinsics_cfg.get("fx", 0.0))
    K[1, 1] = float(intrinsics_cfg.get("fy", 0.0))
    K[0, 2] = float(intrinsics_cfg.get("cx", 0.0))
    K[1, 2] = float(intrinsics_cfg.get("cy", 0.0))
    return K


def _maybe_scale_K_to_image(K: np.ndarray, calib_cfg: dict, image_size: tuple[int, int]) -> np.ndarray:
    source_size = calib_cfg.get("calibration_image_size")
    if not source_size:
        _warn_if_principal_point_looks_outside_image(K, image_size)
        return K
    src_w, src_h = float(source_size[0]), float(source_size[1])
    dst_w, dst_h = float(image_size[0]), float(image_size[1])
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid calibration_image_size: {source_size}")
    scaled = K.copy()
    scaled[0, :] *= dst_w / src_w
    scaled[1, :] *= dst_h / src_h
    scaled[2, :] = np.array([0.0, 0.0, 1.0])
    return scaled


def _warn_if_principal_point_looks_outside_image(K: np.ndarray, image_size: tuple[int, int]) -> None:
    w, h = image_size
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if cx < -0.05 * w or cx > 1.05 * w or cy < -0.05 * h or cy > 1.05 * h:
        print(
            "[camera_calibration] WARNING: calibration principal point "
            f"({cx:.2f}, {cy:.2f}) is outside image size ({w}, {h}). "
            "Set auto_init.camera_calibration.calibration_image_size to the "
            "original calibration resolution so K can be scaled correctly."
        )


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to inspect the first-frame image size.") from exc
    with Image.open(path) as image:
        return image.size


def _write_intrinsics(
    path: str | Path,
    K: np.ndarray,
    source: str,
    raw_K: np.ndarray | None = None,
    D: np.ndarray | None = None,
    rms: Any = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _matrix_to_intrinsics_dict(K)
    payload["K"] = np.asarray(K, dtype=float).tolist()
    payload["source"] = source
    if raw_K is not None:
        payload["raw_K"] = np.asarray(raw_K, dtype=float).tolist()
    if D is not None:
        payload["D"] = np.asarray(D, dtype=float).reshape(-1).tolist()
    if rms is not None:
        payload["rms"] = _as_optional_float(rms)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _matrix_to_intrinsics_dict(K: np.ndarray) -> dict[str, float]:
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
    }


def _format_output_path(template: str, cache_dir: Path, episode_index: int) -> Path:
    return Path(
        template.format(
            cache_dir=str(cache_dir),
            episode_index=episode_index,
        )
    )


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(np.asarray(value).reshape(()))


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for fisheye undistortion. Install opencv-python in the "
            "environment that runs build_init_meta.py."
        ) from exc
    return cv2
