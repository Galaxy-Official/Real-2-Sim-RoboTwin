"""Fixed ALOHA wrist-camera-to-eef transform helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_ALOHA_CAMERA_CONVENTION = "opencv"
DEFAULT_ALOHA_WRIST_TO_EEF_MATRIX = np.array(
    [
        [0.000001, -0.000796, -1.000000, 0.000000],
        [0.905971, 0.423340, -0.000337, 0.136103],
        [0.423341, -0.905970, 0.000722, 0.074117],
        [0.000000, 0.000000, 0.000000, 1.000000],
    ],
    dtype=np.float64,
)
DEFAULT_ALOHA_WRIST_TO_EEF_TRANSLATION = DEFAULT_ALOHA_WRIST_TO_EEF_MATRIX[:3, 3].copy()
DEFAULT_ALOHA_WRIST_TO_EEF_ROTVEC = np.array(
    [-0.8808659531652236, -1.384415791655276, 0.8819689407892274],
    dtype=np.float64,
)


def load_cam_T_eef(config: dict) -> np.ndarray:
    auto_init_cfg = config.get("auto_init", {})
    pose_cfg = auto_init_cfg.get("aloha_wrist_to_eef_pose", {})
    matrix = pose_cfg.get("matrix")
    if matrix is not None:
        mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        return mat

    translation = np.asarray(
        pose_cfg.get("translation", DEFAULT_ALOHA_WRIST_TO_EEF_TRANSLATION),
        dtype=np.float64,
    ).reshape(3)
    rotvec = np.asarray(
        pose_cfg.get("rotvec", DEFAULT_ALOHA_WRIST_TO_EEF_ROTVEC),
        dtype=np.float64,
    ).reshape(3)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    mat[:3, 3] = translation
    return mat


def load_camera_convention(config: dict) -> str:
    auto_init_cfg = config.get("auto_init", {})
    pose_cfg = auto_init_cfg.get("aloha_wrist_to_eef_pose", {})
    return str(pose_cfg.get("camera_convention", DEFAULT_ALOHA_CAMERA_CONVENTION)).lower()
