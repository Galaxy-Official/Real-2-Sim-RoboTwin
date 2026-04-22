"""Replay policy for LeRobot parquet data with task-side auto initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .replay_lerobot_loader import load_episode_state_sequence
    from .坐标系转换 import matrix_to_pose7d_wxyz, pose6d_to_matrix, real_base_matrix_to_robotwin
except ImportError:
    from replay_lerobot_loader import load_episode_state_sequence
    from 坐标系转换 import matrix_to_pose7d_wxyz, pose6d_to_matrix, real_base_matrix_to_robotwin


REPLAY_DATA = None
STEP_IDX = 0
ANCHOR_SIM_MAT = None
ANCHOR_RAW_MAT = None
ACTIVE_ARM = "right"
CHUNK_SIZE = 1
HOLD_GRIPPER_OVERRIDE = None
MAX_CONSECUTIVE_FAILS = 5
CONSECUTIVE_ACTIVE_FAILS = 0


@dataclass
class ReplayModel:
    replay_data: dict
    hold_gripper: float | None

    def update_obs(self, obs: Any) -> None:
        return None

    def reset(self) -> None:
        return None


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def _pose7d_wxyz_to_matrix(pose7d: np.ndarray) -> np.ndarray:
    pose7d = np.asarray(pose7d, dtype=np.float64).reshape(7)
    mat = np.eye(4, dtype=np.float64)
    quat_xyzw = _wxyz_to_xyzw(pose7d[3:])
    from scipy.spatial.transform import Rotation as R

    mat[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    mat[:3, 3] = pose7d[:3]
    return mat


def _get_active_arm_pose(sim_obs: dict, arm: str) -> tuple[np.ndarray, float]:
    key = f"{arm}_endpose"
    grip_key = f"{arm}_gripper"
    pose = np.asarray(sim_obs["endpose"][key][:7], dtype=np.float64)
    pose[3:7] /= np.linalg.norm(pose[3:7])
    gripper = float(sim_obs["endpose"][grip_key])
    return pose, gripper


def _get_idle_arm_action(sim_obs: dict, active_arm: str) -> np.ndarray:
    idle_arm = "left" if active_arm == "right" else "right"
    pose, gripper = _get_active_arm_pose(sim_obs, idle_arm)
    return np.concatenate([pose, [gripper]])


def _load_replay_data(data_dir: str, episode_index: int, state_column: str) -> dict:
    episode = load_episode_state_sequence(
        data_dir=data_dir,
        episode_index=episode_index,
        state_column=state_column,
    )
    robotwin_pose_mats = [real_base_matrix_to_robotwin(pose6d_to_matrix(pose)) for pose in episode["poses_real"]]
    robotwin_poses = np.stack([matrix_to_pose7d_wxyz(mat) for mat in robotwin_pose_mats], axis=0)
    return {
        "poses_real": episode["poses_real"],
        "poses_robotwin": robotwin_poses,
        "pose_mats_robotwin": robotwin_pose_mats,
        "gripper": episode["gripper"],
        "times": episode["times"],
        "length": episode["length"],
        "parquet_path": episode["parquet_path"],
    }


def get_model(usr_args: dict) -> ReplayModel:
    global REPLAY_DATA, STEP_IDX, ACTIVE_ARM, ANCHOR_SIM_MAT, ANCHOR_RAW_MAT
    global HOLD_GRIPPER_OVERRIDE, CONSECUTIVE_ACTIVE_FAILS, CHUNK_SIZE, MAX_CONSECUTIVE_FAILS

    ACTIVE_ARM = usr_args.get("replay_arm", "right")
    state_column = usr_args.get("state_column", "observation.state")
    HOLD_GRIPPER_OVERRIDE = usr_args.get("hold_gripper")
    HOLD_GRIPPER_OVERRIDE = None if HOLD_GRIPPER_OVERRIDE is None else float(HOLD_GRIPPER_OVERRIDE)
    CHUNK_SIZE = int(usr_args.get("chunk_size", 1))
    MAX_CONSECUTIVE_FAILS = int(usr_args.get("max_consecutive_fails", 5))

    REPLAY_DATA = _load_replay_data(
        data_dir=usr_args["data_dir"],
        episode_index=int(usr_args.get("episode_index", 0)),
        state_column=state_column,
    )
    STEP_IDX = 0
    ANCHOR_SIM_MAT = None
    ANCHOR_RAW_MAT = None
    CONSECUTIVE_ACTIVE_FAILS = 0

    print(f"[Replay_Policy] Loaded parquet replay from {REPLAY_DATA['parquet_path']}")
    print(f"[Replay_Policy] arm={ACTIVE_ARM} steps={REPLAY_DATA['length']}")

    return ReplayModel(replay_data=REPLAY_DATA, hold_gripper=HOLD_GRIPPER_OVERRIDE)


def _build_target_action(target_mat: np.ndarray, target_gripper: float, sim_obs: dict, arm: str) -> np.ndarray:
    active_pose = matrix_to_pose7d_wxyz(target_mat)
    active_action = np.concatenate([active_pose, [target_gripper]])
    idle_action = _get_idle_arm_action(sim_obs, arm)
    if arm == "left":
        return np.concatenate([active_action, idle_action])
    return np.concatenate([idle_action, active_action])


def eval(TASK_ENV, model: ReplayModel, observation: dict) -> None:
    global REPLAY_DATA, STEP_IDX, ANCHOR_SIM_MAT, ANCHOR_RAW_MAT
    global CONSECUTIVE_ACTIVE_FAILS, HOLD_GRIPPER_OVERRIDE

    if REPLAY_DATA is None:
        REPLAY_DATA = model.replay_data
    if HOLD_GRIPPER_OVERRIDE is None:
        HOLD_GRIPPER_OVERRIDE = model.hold_gripper

    if REPLAY_DATA["length"] == 0:
        TASK_ENV.replay_finished = True
        TASK_ENV.eval_success = True
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    sim_obs = TASK_ENV.get_obs()
    if ANCHOR_SIM_MAT is None or ANCHOR_RAW_MAT is None:
        sim_pose7d, _ = _get_active_arm_pose(sim_obs, ACTIVE_ARM)
        ANCHOR_SIM_MAT = _pose7d_wxyz_to_matrix(sim_pose7d)
        ANCHOR_RAW_MAT = np.asarray(REPLAY_DATA["pose_mats_robotwin"][0], dtype=np.float64)
        print("[Replay_Policy] Anchored replay to current task-side initialized ee pose.")

    if STEP_IDX >= REPLAY_DATA["length"]:
        TASK_ENV.replay_finished = True
        TASK_ENV.eval_success = True
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    end_idx = min(STEP_IDX + CHUNK_SIZE, REPLAY_DATA["length"])
    for seq_idx in range(STEP_IDX, end_idx):
        target_raw_mat = np.asarray(REPLAY_DATA["pose_mats_robotwin"][seq_idx], dtype=np.float64)
        delta_mat = np.linalg.inv(ANCHOR_RAW_MAT) @ target_raw_mat
        target_sim_mat = ANCHOR_SIM_MAT @ delta_mat
        target_gripper = (
            HOLD_GRIPPER_OVERRIDE
            if HOLD_GRIPPER_OVERRIDE is not None
            else float(REPLAY_DATA["gripper"][seq_idx])
        )
        full_action = _build_target_action(target_sim_mat, target_gripper, sim_obs, ACTIVE_ARM)
        action_debug = TASK_ENV.take_action(full_action, action_type="ee")
        active_status = None
        if isinstance(action_debug, dict):
            active_status = action_debug.get(f"{ACTIVE_ARM}_status")

        if active_status in {"Success", "Skip", None}:
            CONSECUTIVE_ACTIVE_FAILS = 0
        else:
            CONSECUTIVE_ACTIVE_FAILS += 1
            print(f"[Replay_Policy] frame={seq_idx} active_status={active_status}")
            if CONSECUTIVE_ACTIVE_FAILS >= MAX_CONSECUTIVE_FAILS:
                print("[Replay_Policy] abort replay: too many consecutive active-arm planning failures")
                TASK_ENV.take_action_cnt = TASK_ENV.step_lim
                TASK_ENV.replay_failed = True
                return

    STEP_IDX = end_idx
    if STEP_IDX >= REPLAY_DATA["length"]:
        print(f"[Replay_Policy] Finished all {REPLAY_DATA['length']} frames")
        TASK_ENV.replay_finished = True
        TASK_ENV.eval_success = True
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim


def reset_model(model: ReplayModel) -> None:
    global STEP_IDX, ANCHOR_SIM_MAT, ANCHOR_RAW_MAT, CONSECUTIVE_ACTIVE_FAILS
    STEP_IDX = 0
    ANCHOR_SIM_MAT = None
    ANCHOR_RAW_MAT = None
    CONSECUTIVE_ACTIVE_FAILS = 0
    model.reset()
