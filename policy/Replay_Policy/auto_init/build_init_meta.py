#!/usr/bin/env python3
"""Generate task-side auto-init metadata from the first frame of a LeRobot episode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPLAY_POLICY_DIR = THIS_DIR.parent
if str(REPLAY_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(REPLAY_POLICY_DIR))

from auto_init.aloha_extrinsics import load_cam_T_eef, load_camera_convention
from auto_init.camera_calibration import maybe_undistort_mask
from auto_init.depth_anything_v2_runner import run_depth_anything
from auto_init.foundationpose_runner import run_foundationpose
from auto_init.mask_provider import resolve_mask_path
from auto_init.real_data_reader import extract_first_frame_inputs
from 坐标系转换 import (
    build_real_T_cam_from_eef,
    camera_matrix_to_real_base,
    camera_matrix_to_robotwin,
    matrix_to_pose6d,
    matrix_to_pose7d_wxyz,
    real_base_matrix_to_robotwin,
)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--replay-arm", default="right")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-image", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    object_config = load_yaml(config["object_config_path"])

    first_frame = extract_first_frame_inputs(
        config,
        args.data_dir,
        args.episode_index,
        frame_image_override=args.frame_image,
    )
    cache_dir = str(Path(config["auto_init"]["first_frame_cache_dir"]).resolve())
    mask_path = resolve_mask_path(config, args.data_dir, args.episode_index)
    mask_path = maybe_undistort_mask(
        auto_init_cfg=config.get("auto_init", {}),
        mask_path=mask_path,
        cache_dir=cache_dir,
        episode_index=args.episode_index,
        calibration=first_frame.get("calibration"),
    )
    depth_path = run_depth_anything(
        config=config,
        image_path=first_frame["frame_path"],
        cache_dir=cache_dir,
        episode_index=args.episode_index,
    )
    cam_T_obj, fp_output_path = run_foundationpose(
        config=config,
        image_path=first_frame["frame_path"],
        depth_path=str(depth_path),
        mask_path=str(mask_path),
        intrinsics_path=first_frame["intrinsics_path"],
        mesh_path=object_config["mesh_path"],
        cache_dir=cache_dir,
        episode_index=args.episode_index,
    )

    cam_T_eef = load_cam_T_eef(config)
    camera_convention = load_camera_convention(config)
    real_eef_pose6d = np.asarray(first_frame["real_eef_pose6d"], dtype=np.float64)
    real_T_cam = build_real_T_cam_from_eef(real_eef_pose6d=real_eef_pose6d, cam_T_eef=cam_T_eef)
    real_T_obj = camera_matrix_to_real_base(cam_T_obj, real_T_cam)
    robotwin_T_obj = real_base_matrix_to_robotwin(real_T_obj)
    robotwin_T_eef = real_base_matrix_to_robotwin(
        camera_matrix_to_real_base(cam_T_eef, real_T_cam)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_index": args.episode_index,
        "replay_arm": args.replay_arm,
        "object_name": object_config.get("name", object_config.get("modelname", "target_object")),
        "target_object_modelname": object_config.get("modelname", "target_object"),
        "first_frame_image": first_frame["frame_path"],
        "raw_first_frame_image": first_frame["raw_frame_path"],
        "first_frame_undistorted": first_frame["undistorted"],
        "wrist_video_path": first_frame["video_path"],
        "mask_path": str(mask_path),
        "depth_path": str(depth_path),
        "foundationpose_output_path": str(fp_output_path),
        "mesh_path": object_config["mesh_path"],
        "intrinsics": first_frame["intrinsics"],
        "intrinsics_matrix": first_frame["intrinsics_matrix"],
        "calibration": first_frame["calibration"],
        "cam_T_obj": np.asarray(cam_T_obj, dtype=float).tolist(),
        "cam_T_eef": np.asarray(cam_T_eef, dtype=float).tolist(),
        "camera_convention": camera_convention,
        "real_T_cam": np.asarray(real_T_cam, dtype=float).tolist(),
        "real_T_obj": np.asarray(real_T_obj, dtype=float).tolist(),
        "real_T_eef": np.asarray(camera_matrix_to_real_base(cam_T_eef, real_T_cam), dtype=float).tolist(),
        "robotwin_T_obj": np.asarray(robotwin_T_obj, dtype=float).tolist(),
        "robotwin_T_eef": np.asarray(robotwin_T_eef, dtype=float).tolist(),
        "robotwin_obj_pose6d": matrix_to_pose6d(robotwin_T_obj).tolist(),
        "robotwin_eef_pose7d_wxyz": matrix_to_pose7d_wxyz(robotwin_T_eef).tolist(),
        "robotwin_obj_pose7d_wxyz": matrix_to_pose7d_wxyz(robotwin_T_obj).tolist(),
        "real_eef_pose6d": real_eef_pose6d.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build_init_meta] Wrote init metadata to {output_path}")


if __name__ == "__main__":
    main()
