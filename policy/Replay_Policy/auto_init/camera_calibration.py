"""Camera intrinsics and optional fisheye undistortion helpers for auto-init."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RAW_FISHEYE_SIZE = (1536, 1536)  # width, height
DEFAULT_DISTORTED_CANVAS_SIZE = (640, 480)  # width, height
DEFAULT_RESIZED_CONTENT_SIZE = (480, 480)  # width, height


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

    The ALOHA wrist stream used here is not a direct resize from the original
    fisheye calibration image. It is:

    1536x1536 raw fisheye -> 480x480 isotropic resize -> centered on a
    640x480 canvas with 80 px black borders on left/right.

    Therefore runtime fisheye undistortion must first express K_raw_1536 in
    the distorted 640x480 canvas coordinate system, then compute the final
    undistorted 640x480 pinhole K.
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
            "preprocessing_debug": None,
        }

    K = np.asarray(calibration["K"], dtype=np.float64).reshape(3, 3)
    D = np.asarray(calibration["D"], dtype=np.float64).reshape(-1, 1)
    image_size = _read_image_size(frame_path)

    if calib_type == "pinhole":
        K = _maybe_scale_K_to_image(K, calib_cfg, image_size)
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
            "calibration": _calibration_metadata(calibration=calibration, calib_type=calib_type, D=D, K=K),
            "undistorted": False,
            "preprocessing_debug": None,
        }

    _require_raw_fisheye_calibration(calibration)
    fisheye_preprocess = _build_fisheye_preprocess_metadata(calib_cfg, image_size)
    K_raw_1536 = K
    K_distorted_640 = build_distorted_640_intrinsics_from_raw_1536(
        K_raw_1536,
        raw_resolution=fisheye_preprocess["raw_resolution"],
        resized_resolution=fisheye_preprocess["resized_resolution"],
        canvas_resolution=fisheye_preprocess["canvas_resolution"],
        content_offset=fisheye_preprocess["content_offset"],
    )

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
            K=K_distorted_640,
            D=D,
            balance=float(undistort_cfg.get("balance", 0.0)),
            fov_scale=float(undistort_cfg.get("fov_scale", 1.0)),
            interpolation="linear",
            expected_input_size=fisheye_preprocess["canvas_resolution"],
        )
        intrinsics_path = _write_intrinsics(
            cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
            new_K,
            source="fisheye_undistorted",
            raw_K=K_raw_1536,
            D=D,
            rms=calibration.get("rms"),
            extra={
                "pipeline": "raw1536_resize480_center_pad640_then_fisheye_undistort",
                "K_raw_1536": K_raw_1536.tolist(),
                "K_distorted_640": K_distorted_640.tolist(),
                "K_undistorted_640": new_K.tolist(),
                "D_raw": D.reshape(-1).tolist(),
                "raw_resolution": list(fisheye_preprocess["raw_resolution"]),
                "resized_resolution": list(fisheye_preprocess["resized_resolution"]),
                "canvas_resolution": list(fisheye_preprocess["canvas_resolution"]),
                "content_offset": list(fisheye_preprocess["content_offset"]),
            },
        )
        preprocessing_debug = _write_fisheye_preprocess_debug(
            frame_path=frame_path,
            cache_dir=cache_dir,
            episode_index=episode_index,
            K_raw_1536=K_raw_1536,
            D_raw=D,
            K_distorted_640=K_distorted_640,
            K_undistorted_640=new_K,
            fisheye_preprocess=fisheye_preprocess,
            intrinsics_path=intrinsics_path,
            undistorted_path=undistorted_path,
        )
        return {
            "frame_path": str(undistorted_path),
            "raw_frame_path": str(frame_path),
            "intrinsics": _matrix_to_intrinsics_dict(new_K),
            "intrinsics_matrix": new_K.tolist(),
            "intrinsics_path": str(intrinsics_path),
            "calibration": _calibration_metadata(
                calibration=calibration,
                calib_type=calib_type,
                D=D,
                K=K_distorted_640,
                K_raw_1536=K_raw_1536,
                K_distorted_640=K_distorted_640,
                K_undistorted_640=new_K,
                fisheye_preprocess=fisheye_preprocess,
            ),
            "undistorted": True,
            "preprocessing_debug": preprocessing_debug,
        }

    intrinsics_path = _write_intrinsics(
        cache_dir / f"episode_{episode_index:06d}_intrinsics.json",
        K_distorted_640,
        source="fisheye_raw_not_undistorted",
        raw_K=K_raw_1536,
        D=D,
        rms=calibration.get("rms"),
        extra={
            "pipeline": "raw1536_resize480_center_pad640_without_undistort",
            "K_raw_1536": K_raw_1536.tolist(),
            "K_distorted_640": K_distorted_640.tolist(),
            "D_raw": D.reshape(-1).tolist(),
            "raw_resolution": list(fisheye_preprocess["raw_resolution"]),
            "resized_resolution": list(fisheye_preprocess["resized_resolution"]),
            "canvas_resolution": list(fisheye_preprocess["canvas_resolution"]),
            "content_offset": list(fisheye_preprocess["content_offset"]),
        },
    )
    return {
        "frame_path": str(frame_path),
        "raw_frame_path": str(frame_path),
        "intrinsics": _matrix_to_intrinsics_dict(K_distorted_640),
        "intrinsics_matrix": K_distorted_640.tolist(),
        "intrinsics_path": str(intrinsics_path),
        "calibration": _calibration_metadata(
            calibration=calibration,
            calib_type=calib_type,
            D=D,
            K=K_distorted_640,
            K_raw_1536=K_raw_1536,
            K_distorted_640=K_distorted_640,
            fisheye_preprocess=fisheye_preprocess,
        ),
        "undistorted": False,
        "preprocessing_debug": None,
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

    K = np.asarray(
        calibration.get("K_distorted_640", calibration["K"]),
        dtype=np.float64,
    ).reshape(3, 3)
    D = np.asarray(calibration.get("D_raw", calibration["D"]), dtype=np.float64).reshape(-1, 1)
    new_K = calibration.get("K_undistorted_640")
    new_K = None if new_K is None else np.asarray(new_K, dtype=np.float64).reshape(3, 3)
    expected_input_size = calibration.get("fisheye_preprocess", {}).get("canvas_resolution")
    expected_input_size = None if expected_input_size is None else tuple(expected_input_size)
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
        new_K=new_K,
        expected_input_size=expected_input_size,
    )
    return mask_output


def build_distorted_640_intrinsics_from_raw_1536(
    K_raw_1536: np.ndarray,
    raw_resolution: tuple[int, int] = DEFAULT_RAW_FISHEYE_SIZE,
    resized_resolution: tuple[int, int] = DEFAULT_RESIZED_CONTENT_SIZE,
    canvas_resolution: tuple[int, int] = DEFAULT_DISTORTED_CANVAS_SIZE,
    content_offset: tuple[float, float] | None = None,
) -> np.ndarray:
    """Express raw 1536x1536 fisheye K in the 640x480 padded image coordinates."""

    raw_w, raw_h = _as_size(raw_resolution, "raw_resolution")
    resized_w, resized_h = _as_size(resized_resolution, "resized_resolution")
    canvas_w, canvas_h = _as_size(canvas_resolution, "canvas_resolution")
    if content_offset is None:
        tx = (canvas_w - resized_w) * 0.5
        ty = (canvas_h - resized_h) * 0.5
    else:
        tx, ty = _as_offset(content_offset, "content_offset")

    if resized_w > canvas_w or resized_h > canvas_h:
        raise ValueError(
            f"resized_resolution {resized_resolution} cannot fit in canvas_resolution {canvas_resolution}."
        )

    sx = float(resized_w) / float(raw_w)
    sy = float(resized_h) / float(raw_h)
    if not np.isclose(sx, sy, rtol=0.0, atol=1e-12):
        raise ValueError(
            "The recorded wrist preprocessing is expected to be an isotropic resize. "
            f"Got scale_x={sx}, scale_y={sy}."
        )

    K_distorted = np.asarray(K_raw_1536, dtype=np.float64).reshape(3, 3).copy()
    K_distorted[0, :] *= sx
    K_distorted[1, :] *= sy
    K_distorted[0, 2] += tx
    K_distorted[1, 2] += ty
    K_distorted[2, :] = np.array([0.0, 0.0, 1.0])
    return K_distorted


def undistort_distorted_640_and_compute_output_K(
    image: np.ndarray,
    K_distorted_640: np.ndarray,
    D_raw: np.ndarray,
    original_resolution: tuple[int, int] = DEFAULT_DISTORTED_CANVAS_SIZE,
    balance: float = 0.0,
    fov_scale: float = 1.0,
    interpolation: str = "linear",
    binary_output: bool = False,
    K_undistorted_640: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort a 640x480 padded fisheye frame and return its matching pinhole K."""

    cv2 = _require_cv2()
    expected_w, expected_h = _as_size(original_resolution, "original_resolution")
    h, w = image.shape[:2]
    if (w, h) != (expected_w, expected_h):
        raise ValueError(
            "Fisheye undistortion input size does not match its intrinsics. "
            f"Expected {(expected_w, expected_h)}, got {(w, h)}."
        )

    K_distorted_640 = np.asarray(K_distorted_640, dtype=np.float64).reshape(3, 3)
    D_raw = np.asarray(D_raw, dtype=np.float64).reshape(4, 1)
    if K_undistorted_640 is None:
        K_undistorted_640 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K_distorted_640,
            D_raw,
            (w, h),
            np.eye(3),
            balance=balance,
            new_size=(w, h),
            fov_scale=fov_scale,
        )
    else:
        K_undistorted_640 = np.asarray(K_undistorted_640, dtype=np.float64).reshape(3, 3)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K_distorted_640,
        D_raw,
        np.eye(3),
        K_undistorted_640,
        (w, h),
        cv2.CV_16SC2,
    )
    interp = cv2.INTER_NEAREST if interpolation == "nearest" else cv2.INTER_LINEAR
    undistorted = cv2.remap(image, map1, map2, interpolation=interp)
    if binary_output:
        undistorted = (undistorted > 0).astype(np.uint8) * 255
    return undistorted, np.asarray(K_undistorted_640, dtype=np.float64)


def undistort_image_file(
    input_path: str | Path,
    output_path: str | Path,
    K: np.ndarray,
    D: np.ndarray,
    balance: float = 0.0,
    fov_scale: float = 1.0,
    interpolation: str = "linear",
    binary_output: bool = False,
    new_K: np.ndarray | None = None,
    expected_input_size: tuple[int, int] | None = None,
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
    undistorted, new_K = undistort_distorted_640_and_compute_output_K(
        image=image,
        K_distorted_640=K,
        D_raw=D,
        original_resolution=(w, h) if expected_input_size is None else expected_input_size,
        balance=balance,
        fov_scale=fov_scale,
        interpolation=interpolation,
        binary_output=binary_output,
        K_undistorted_640=new_K,
    )

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
            if "K" in data.files and "D" in data.files:
                K_key, D_key = "K", "D"
            elif "K_raw" in data.files and "D_raw" in data.files:
                K_key, D_key = "K_raw", "D_raw"
            elif "K_new" in data.files and "D_raw" in data.files:
                K_key, D_key = "K_new", "D_raw"
            else:
                raise KeyError(
                    f"Unsupported calibration npz layout in {path}. "
                    "Expected raw fisheye keys K / D, K_raw / D_raw, or legacy K_new / D_raw."
                )
            return {
                "K": np.asarray(data[K_key], dtype=np.float64),
                "D": np.asarray(data[D_key], dtype=np.float64),
                "rms": _as_optional_float(data["rms"]) if "rms" in data.files else None,
                "K_key": K_key,
                "D_key": D_key,
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


def _calibration_metadata(
    calibration: dict[str, Any],
    calib_type: str,
    D: np.ndarray,
    K: np.ndarray | None = None,
    K_raw_1536: np.ndarray | None = None,
    K_distorted_640: np.ndarray | None = None,
    K_undistorted_640: np.ndarray | None = None,
    fisheye_preprocess: dict | None = None,
) -> dict:
    payload = {
        "type": calib_type,
        "path": calibration.get("path"),
        "K": None if K is None else np.asarray(K, dtype=float).tolist(),
        "D": D.reshape(-1).tolist(),
        "rms": _as_optional_float(calibration.get("rms")),
        "K_key": calibration.get("K_key"),
        "D_key": calibration.get("D_key"),
    }
    if K_raw_1536 is not None:
        payload["K_raw_1536"] = np.asarray(K_raw_1536, dtype=float).tolist()
    if K_distorted_640 is not None:
        payload["K_distorted_640"] = np.asarray(K_distorted_640, dtype=float).tolist()
    if K_undistorted_640 is not None:
        payload["K_undistorted_640"] = np.asarray(K_undistorted_640, dtype=float).tolist()
    if fisheye_preprocess is not None:
        payload["fisheye_preprocess"] = _jsonable_preprocess(fisheye_preprocess)
    return payload


def _require_raw_fisheye_calibration(calibration: dict[str, Any]) -> None:
    if calibration.get("K_key") in {"K", "K_raw"} and calibration.get("D_key") in {"D", "D_raw"}:
        return
    raise ValueError(
        "Runtime fisheye undistortion now requires raw fisheye calibration. "
        f"Got keys {calibration.get('K_key')}/{calibration.get('D_key')} from {calibration.get('path')}. "
        "Use policy/Replay_Policy/auto_init/fisheye_calib_result.npz with keys K/D. "
        "Do not use fisheye_calib_result_resized.npz K_new as fisheye input intrinsics."
    )


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


def _build_fisheye_preprocess_metadata(calib_cfg: dict, image_size: tuple[int, int]) -> dict:
    raw_resolution = _as_size(
        calib_cfg.get("raw_image_size", DEFAULT_RAW_FISHEYE_SIZE),
        "camera_calibration.raw_image_size",
    )
    resized_resolution = _as_size(
        calib_cfg.get("resized_content_size", DEFAULT_RESIZED_CONTENT_SIZE),
        "camera_calibration.resized_content_size",
    )
    canvas_resolution = _as_size(
        calib_cfg.get("distorted_image_size", DEFAULT_DISTORTED_CANVAS_SIZE),
        "camera_calibration.distorted_image_size",
    )
    content_offset_cfg = calib_cfg.get("content_offset")
    if content_offset_cfg is None:
        content_offset = (
            (canvas_resolution[0] - resized_resolution[0]) * 0.5,
            (canvas_resolution[1] - resized_resolution[1]) * 0.5,
        )
    else:
        content_offset = _as_offset(content_offset_cfg, "camera_calibration.content_offset")

    if tuple(image_size) != tuple(canvas_resolution):
        raise ValueError(
            "Current wrist frame size does not match the configured distorted fisheye canvas. "
            f"Frame size={image_size}, configured distorted_image_size={canvas_resolution}."
        )
    return {
        "raw_resolution": raw_resolution,
        "resized_resolution": resized_resolution,
        "canvas_resolution": canvas_resolution,
        "content_offset": content_offset,
    }


def _write_fisheye_preprocess_debug(
    frame_path: Path,
    cache_dir: Path,
    episode_index: int,
    K_raw_1536: np.ndarray,
    D_raw: np.ndarray,
    K_distorted_640: np.ndarray,
    K_undistorted_640: np.ndarray,
    fisheye_preprocess: dict,
    intrinsics_path: Path,
    undistorted_path: Path,
) -> dict:
    debug_images = _write_fisheye_intermediate_images(
        frame_path=frame_path,
        cache_dir=cache_dir,
        episode_index=episode_index,
        fisheye_preprocess=fisheye_preprocess,
    )
    payload = {
        "pipeline": "raw1536_resize480_center_pad640_then_fisheye_undistort",
        "K_raw_1536": np.asarray(K_raw_1536, dtype=float).tolist(),
        "K_distorted_640": np.asarray(K_distorted_640, dtype=float).tolist(),
        "K_undistorted_640": np.asarray(K_undistorted_640, dtype=float).tolist(),
        "D_raw": np.asarray(D_raw, dtype=float).reshape(-1).tolist(),
        "raw_resolution": list(fisheye_preprocess["raw_resolution"]),
        "resized_resolution": list(fisheye_preprocess["resized_resolution"]),
        "canvas_resolution": list(fisheye_preprocess["canvas_resolution"]),
        "content_offset": list(fisheye_preprocess["content_offset"]),
        "final_da3_image": str(Path(undistorted_path).resolve()),
        "final_da3_intrinsics_path": str(Path(intrinsics_path).resolve()),
        "final_da3_intrinsics_json": json.loads(Path(intrinsics_path).read_text(encoding="utf-8")),
        "debug_images": debug_images,
    }
    debug_path = cache_dir / f"episode_{episode_index:06d}_fisheye_preprocess_debug.json"
    payload["debug_json_path"] = str(debug_path.resolve())
    debug_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "[camera_calibration] fisheye preprocessing: "
        f"K_raw_1536 -> K_distorted_640 -> K_undistorted_640. Debug: {debug_path.resolve()}"
    )
    return payload


def _write_fisheye_intermediate_images(
    frame_path: Path,
    cache_dir: Path,
    episode_index: int,
    fisheye_preprocess: dict,
) -> dict:
    cv2 = _require_cv2()
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image for fisheye preprocessing debug: {frame_path}")

    resized_w, resized_h = fisheye_preprocess["resized_resolution"]
    canvas_w, canvas_h = fisheye_preprocess["canvas_resolution"]
    tx, ty = fisheye_preprocess["content_offset"]
    tx_i, ty_i = int(round(tx)), int(round(ty))
    if (tx_i, ty_i) != (tx, ty):
        raise ValueError(f"Non-integer content_offset is not supported for debug image extraction: {(tx, ty)}")

    content = image[ty_i:ty_i + resized_h, tx_i:tx_i + resized_w]
    if content.shape[:2] != (resized_h, resized_w):
        raise ValueError(
            f"Cannot crop resized content {resized_w}x{resized_h} at offset {(tx_i, ty_i)} "
            f"from image shape {image.shape[:2]}."
        )
    reconstructed = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=image.dtype)
    reconstructed[ty_i:ty_i + resized_h, tx_i:tx_i + resized_w] = content

    content_path = cache_dir / f"episode_{episode_index:06d}_distorted_480_content.png"
    reconstructed_path = cache_dir / f"episode_{episode_index:06d}_distorted_640_center_pad.png"
    cv2.imwrite(str(content_path), content)
    cv2.imwrite(str(reconstructed_path), reconstructed)
    return {
        "distorted_480_content": str(content_path.resolve()),
        "distorted_640_center_pad": str(reconstructed_path.resolve()),
    }


def _jsonable_preprocess(fisheye_preprocess: dict) -> dict:
    return {
        "raw_resolution": list(fisheye_preprocess["raw_resolution"]),
        "resized_resolution": list(fisheye_preprocess["resized_resolution"]),
        "canvas_resolution": list(fisheye_preprocess["canvas_resolution"]),
        "content_offset": list(fisheye_preprocess["content_offset"]),
    }


def _as_size(value: Any, name: str) -> tuple[int, int]:
    if value is None or len(value) != 2:
        raise ValueError(f"{name} must be [width, height], got {value!r}.")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return width, height


def _as_offset(value: Any, name: str) -> tuple[float, float]:
    if value is None or len(value) != 2:
        raise ValueError(f"{name} must be [x, y], got {value!r}.")
    return float(value[0]), float(value[1])


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
    extra: dict | None = None,
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
    if extra:
        payload.update(extra)
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
