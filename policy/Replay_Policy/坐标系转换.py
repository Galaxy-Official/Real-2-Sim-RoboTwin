"""Coordinate transform helpers shared by auto-init and replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REAL_BASE_TO_ROBOTWIN_ROT = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
REAL_BASE_TO_ROBOTWIN_TRANS = np.zeros(3, dtype=np.float64)


def _as_pose6d_rotvec(pose: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.size != 6:
        raise ValueError(f"Expected 6 values [x, y, z, rx, ry, rz], got shape {arr.shape}")
    return arr


def pose6d_to_matrix(pose: list[float] | np.ndarray) -> np.ndarray:
    pose = _as_pose6d_rotvec(pose)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def matrix_to_pose6d(mat: list[list[float]] | np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    pose = np.zeros(6, dtype=np.float64)
    pose[:3] = mat[:3, 3]
    pose[3:] = R.from_matrix(mat[:3, :3]).as_rotvec()
    return pose


def matrix_to_pose7d_wxyz(mat: list[list[float]] | np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    quat_xyzw = R.from_matrix(mat[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    return np.concatenate([mat[:3, 3], quat_wxyz])


def pose7d_wxyz_to_matrix(pose: list[float] | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected 7 values [x, y, z, qw, qx, qy, qz], got shape {pose.shape}")
    quat_xyzw = np.array([pose[4], pose[5], pose[6], pose[3]], dtype=np.float64)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = pose[:3]
    return mat


def invert_transform(mat: list[list[float]] | np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(mat, dtype=np.float64).reshape(4, 4))


def compose_transform(lhs: list[list[float]] | np.ndarray, rhs: list[list[float]] | np.ndarray) -> np.ndarray:
    return np.asarray(lhs, dtype=np.float64).reshape(4, 4) @ np.asarray(rhs, dtype=np.float64).reshape(4, 4)


def real_base_matrix_to_robotwin(mat: list[list[float]] | np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = REAL_BASE_TO_ROBOTWIN_ROT @ mat[:3, :3]
    out[:3, 3] = REAL_BASE_TO_ROBOTWIN_ROT @ mat[:3, 3] + REAL_BASE_TO_ROBOTWIN_TRANS
    return out


def real_base_pose_to_robotwin(pose: list[float] | np.ndarray) -> np.ndarray:
    return matrix_to_pose6d(real_base_matrix_to_robotwin(pose6d_to_matrix(pose)))


def camera_matrix_to_real_base(cam_target: list[list[float]] | np.ndarray, real_T_cam: list[list[float]] | np.ndarray) -> np.ndarray:
    return compose_transform(real_T_cam, cam_target)


def camera_matrix_to_robotwin(cam_target: list[list[float]] | np.ndarray, real_T_cam: list[list[float]] | np.ndarray) -> np.ndarray:
    return real_base_matrix_to_robotwin(camera_matrix_to_real_base(cam_target, real_T_cam))


def build_real_T_cam_from_eef(real_eef_pose6d: list[float] | np.ndarray, cam_T_eef: list[list[float]] | np.ndarray) -> np.ndarray:
    real_T_eef = pose6d_to_matrix(real_eef_pose6d)
    return compose_transform(real_T_eef, invert_transform(cam_T_eef))


def save_matrix_json(path: str | Path, key: str, mat: np.ndarray) -> None:
    path = Path(path)
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload[key] = np.asarray(mat, dtype=float).tolist()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
