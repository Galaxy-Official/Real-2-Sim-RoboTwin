"""
RoboTwin Replay Policy for unified raw pose data
===============================================

Flexiv Rizon4 + GN01 版本：
- 使用师兄的 TCP 偏置逻辑：对原始位姿右乘 [R_y(180) * R_z(90)]，修正夹爪朝向。
- 平移使用已验证的 raw->RoboTwin 统一坐标变换。
- 不再执行 Aloha 风格的 6 轴 warmup，而是直接基于当前 nominal state 做单帧增量回放。
- 每一步都实时锁定 idle arm，避免非回放臂被旧缓存位姿拉扯。
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPLAY_ARM = "right"
RAW_QUAT_ORDER = "xyzw"
TRANSLATION_SCALE = 1.0

RAW_TO_WORLD_ROT = np.array(
    [
        [-0.997494312585, 0.016101908673, 0.068889947724],
        [0.067998412082, -0.050562147486, 0.996403374741],
        [0.019527219839, 0.998591106399, 0.049340550328],
    ],
    dtype=np.float64,
)

CHUNK_SIZE = 1
LOCK_ORIENTATION = True


REPLAY_DATA = None
STEP_IDX = 0
WARMUP_DONE = False
LAST_FULL_ACTION = None
ACTIVE_GRIPPER_HOLD = None
HOLD_GRIPPER_OVERRIDE = None
ANCHOR_SIM_XYZ = None
ANCHOR_SIM_QUAT_WXYZ = None
ANCHOR_RAW_XYZ = None
ANCHOR_RAW_QUAT_XYZW = None
CONSECUTIVE_ACTIVE_FAILS = 0


def wxyz_to_xyzw(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q_xyzw):
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def raw_quat_to_scipy_xyzw(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64)
    if RAW_QUAT_ORDER == "xyzw":
        return raw_quat
    if RAW_QUAT_ORDER == "wxyz":
        return wxyz_to_xyzw(raw_quat)
    raise ValueError(f"Unsupported raw quaternion order: {RAW_QUAT_ORDER}")


def ensure_quat_continuity(quat_wxyz, ref_quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    ref_quat_wxyz = np.asarray(ref_quat_wxyz, dtype=np.float64)
    if np.dot(quat_wxyz, ref_quat_wxyz) < 0:
        return -quat_wxyz
    return quat_wxyz


def compute_pose_transform(xyz, quat_xyzw):
    r_raw = R.from_quat(quat_xyzw)
    r_y = R.from_euler("y", 180, degrees=True)
    r_z = R.from_euler("z", 90, degrees=True)
    r_additional = r_y * r_z
    r_out = r_raw * r_additional
    return xyz, r_out.as_quat()


def transform_translation_delta(delta_xyz):
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
    return TRANSLATION_SCALE * (RAW_TO_WORLD_ROT @ delta_xyz)


def estimate_quat_drift_deg(raw_quat):
    quat = np.asarray(raw_quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    dots = np.abs(np.sum(quat * quat[0], axis=1))
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(angles)


def chunk_sort_key(path):
    stem_parts = Path(path).stem.split("_")
    if len(stem_parts) >= 3:
        try:
            return (float(stem_parts[1]), float(stem_parts[2]))
        except ValueError:
            pass
    return (float("inf"), Path(path).name)


def resolve_pose_files(data_dir, episode_index):
    path = Path(data_dir)

    if path.is_file() and path.suffix == ".npz":
        return [path]

    if path.is_dir():
        npz_files = sorted(path.glob("*.npz"), key=chunk_sort_key)
        if npz_files:
            return npz_files

        pose_dir = path / f"pose_{episode_index}"
        if pose_dir.is_dir():
            return sorted(pose_dir.glob("*.npz"), key=chunk_sort_key)

    raise FileNotFoundError(
        f"Cannot find replay pose chunks from `{data_dir}` with episode_index={episode_index}"
    )


def load_raw_pose_replay(data_dir, episode_index=0):
    pose_files = resolve_pose_files(data_dir, episode_index)
    all_pose = []
    all_time = []

    for pose_file in pose_files:
        data = np.load(pose_file)
        if "pose" not in data:
            raise KeyError(f"`pose` key not found in {pose_file}")
        all_pose.append(np.asarray(data["pose"], dtype=np.float32))
        if "time" in data:
            all_time.append(np.asarray(data["time"], dtype=np.float64))

    poses = np.concatenate(all_pose, axis=0)
    times = np.concatenate(all_time, axis=0) if all_time else np.arange(len(poses), dtype=np.float64)

    quat = poses[:, 3:].astype(np.float64)
    _ = estimate_quat_drift_deg(quat)

    if len(times) > 1:
        dt = np.diff(times)
        dt = dt[dt > 1e-6]
        fps = float(1.0 / np.median(dt)) if len(dt) > 0 else 24.0
    else:
        fps = 24.0

    print(f"[Pose Unified] Loaded {len(poses)} frames from {len(pose_files)} file(s)")
    print(f"[Pose Unified] Estimated fps: {fps:.2f}")

    return {
        "poses": poses,
        "times": times,
        "length": len(poses),
        "fps": fps,
    }


def get_active_arm_pose(sim_obs, arm):
    if arm == "right":
        endpose = sim_obs["endpose"]["right_endpose"]
        gripper = float(sim_obs["endpose"]["right_gripper"])
    else:
        endpose = sim_obs["endpose"]["left_endpose"]
        gripper = float(sim_obs["endpose"]["left_gripper"])
    xyz = np.array(endpose[:3], dtype=np.float64)
    quat_wxyz = np.array(endpose[3:7], dtype=np.float64)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    return xyz, quat_wxyz, gripper


def get_idle_arm_action(sim_obs, active_arm):
    if active_arm == "right":
        idle_pose = sim_obs["endpose"]["left_endpose"][:7]
        idle_grip = float(sim_obs["endpose"]["left_gripper"])
    else:
        idle_pose = sim_obs["endpose"]["right_endpose"][:7]
        idle_grip = float(sim_obs["endpose"]["right_gripper"])
    idle_pose = np.array(idle_pose, dtype=np.float64)
    idle_pose[3:7] = idle_pose[3:7] / np.linalg.norm(idle_pose[3:7])
    return np.concatenate([idle_pose, [idle_grip]])


def extract_source_pose(raw_pose_7d):
    xyz = np.asarray(raw_pose_7d[:3], dtype=np.float64)
    quat_xyzw = raw_quat_to_scipy_xyzw(raw_pose_7d[3:7])
    return xyz, quat_xyzw


def build_absolute_target_action(target_pose_7d, sim_obs, arm):
    global ACTIVE_GRIPPER_HOLD
    global ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, ANCHOR_RAW_XYZ, ANCHOR_RAW_QUAT_XYZW

    if ANCHOR_SIM_XYZ is None or ANCHOR_SIM_QUAT_WXYZ is None:
        raise RuntimeError("Replay anchor is not initialized.")
    if ANCHOR_RAW_XYZ is None or ANCHOR_RAW_QUAT_XYZW is None:
        raise RuntimeError("Raw pose anchor is not initialized.")

    target_xyz_raw, target_quat_xyzw = extract_source_pose(target_pose_7d)
    target_xyz_raw, target_quat_xyzw = compute_pose_transform(target_xyz_raw, target_quat_xyzw)

    translation_delta = target_xyz_raw - ANCHOR_RAW_XYZ
    target_xyz = ANCHOR_SIM_XYZ + transform_translation_delta(translation_delta)

    if LOCK_ORIENTATION:
        target_quat_wxyz = np.asarray(ANCHOR_SIM_QUAT_WXYZ, dtype=np.float64)
    else:
        r_anchor_raw = R.from_quat(ANCHOR_RAW_QUAT_XYZW)
        r_target_raw = R.from_quat(target_quat_xyzw)
        r_rel_local = r_anchor_raw.inv() * r_target_raw

        r_anchor_sim = R.from_quat(wxyz_to_xyzw(ANCHOR_SIM_QUAT_WXYZ))
        r_target = r_anchor_sim * r_rel_local

        target_quat_wxyz = xyzw_to_wxyz(r_target.as_quat())
        target_quat_wxyz = ensure_quat_continuity(target_quat_wxyz, ANCHOR_SIM_QUAT_WXYZ)

    active_action = np.concatenate([target_xyz, target_quat_wxyz, [ACTIVE_GRIPPER_HOLD]])
    idle_action = get_idle_arm_action(sim_obs, arm)
    if arm == "left":
        return np.concatenate([active_action, idle_action])
    return np.concatenate([idle_action, active_action])


def encode_obs(observation):
    return observation


def get_model(usr_args):
    global REPLAY_DATA, STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global ACTIVE_GRIPPER_HOLD, REPLAY_ARM, RAW_QUAT_ORDER
    global TRANSLATION_SCALE, HOLD_GRIPPER_OVERRIDE
    global ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, ANCHOR_RAW_XYZ, ANCHOR_RAW_QUAT_XYZW
    global CONSECUTIVE_ACTIVE_FAILS

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    ANCHOR_SIM_XYZ = None
    ANCHOR_SIM_QUAT_WXYZ = None
    ANCHOR_RAW_XYZ = None
    ANCHOR_RAW_QUAT_XYZW = None
    CONSECUTIVE_ACTIVE_FAILS = 0

    data_dir = usr_args["data_dir"]
    episode_index = int(usr_args.get("episode_index", 0))
    REPLAY_ARM = usr_args.get("replay_arm", "right")
    RAW_QUAT_ORDER = usr_args.get("raw_quat_order", "xyzw")
    TRANSLATION_SCALE = float(usr_args.get("translation_scale", 1.0))
    hold_gripper = usr_args.get("hold_gripper")
    HOLD_GRIPPER_OVERRIDE = float(hold_gripper) if hold_gripper is not None else None

    REPLAY_DATA = load_raw_pose_replay(data_dir, episode_index)

    print(f"[Pose Unified - Senior TCP Fix] Config: arm={REPLAY_ARM}")
    print(f"[Pose Unified - Senior TCP Fix] translation_scale={TRANSLATION_SCALE:.4f}")

    class ReplayModel:
        def __init__(self):
            self.obs_cache = []
            self.replay_data = REPLAY_DATA
            self.hold_gripper = HOLD_GRIPPER_OVERRIDE

        def update_obs(self, obs):
            self.obs_cache.append(obs)

        def reset(self):
            self.obs_cache = []

    return ReplayModel()


def eval(TASK_ENV, model, observation):
    global REPLAY_DATA, STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE
    global ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, ANCHOR_RAW_XYZ, ANCHOR_RAW_QUAT_XYZW
    global CONSECUTIVE_ACTIVE_FAILS

    if REPLAY_DATA is None and getattr(model, "replay_data", None) is not None:
        REPLAY_DATA = model.replay_data
    if HOLD_GRIPPER_OVERRIDE is None and getattr(model, "hold_gripper", None) is not None:
        HOLD_GRIPPER_OVERRIDE = model.hold_gripper

    if REPLAY_DATA is None:
        raise RuntimeError("REPLAY_DATA is not initialized.")

    poses_raw = REPLAY_DATA["poses"]
    total_steps = REPLAY_DATA["length"]

    if total_steps == 0:
        return

    if total_steps == 1:
        sim_obs = TASK_ENV.get_obs()
        ACTIVE_GRIPPER_HOLD = (
            HOLD_GRIPPER_OVERRIDE
            if HOLD_GRIPPER_OVERRIDE is not None
            else get_active_arm_pose(sim_obs, REPLAY_ARM)[2]
        )
        anchor_raw_xyz, anchor_raw_quat = extract_source_pose(poses_raw[0])
        anchor_raw_xyz, anchor_raw_quat = compute_pose_transform(anchor_raw_xyz, anchor_raw_quat)
        ANCHOR_RAW_XYZ = anchor_raw_xyz
        ANCHOR_RAW_QUAT_XYZW = anchor_raw_quat
        ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, _ = get_active_arm_pose(sim_obs, REPLAY_ARM)
        full_action = build_absolute_target_action(poses_raw[0], sim_obs, REPLAY_ARM)
        TASK_ENV.take_action(full_action, action_type="ee")
        LAST_FULL_ACTION = full_action.copy()
        return

    if not WARMUP_DONE:
        print("[Pose Unified] First call: align replay state with current nominal robot pose...")
        sim_obs = TASK_ENV.get_obs()
        ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, _ = get_active_arm_pose(sim_obs, REPLAY_ARM)
        anchor_raw_xyz, anchor_raw_quat = extract_source_pose(poses_raw[0])
        anchor_raw_xyz, anchor_raw_quat = compute_pose_transform(anchor_raw_xyz, anchor_raw_quat)
        ANCHOR_RAW_XYZ = anchor_raw_xyz
        ANCHOR_RAW_QUAT_XYZW = anchor_raw_quat
        _, _, active_gripper = get_active_arm_pose(sim_obs, REPLAY_ARM)
        ACTIVE_GRIPPER_HOLD = (
            HOLD_GRIPPER_OVERRIDE if HOLD_GRIPPER_OVERRIDE is not None else active_gripper
        )
        if LOCK_ORIENTATION:
            print("[Pose Unified] Mode: anchored absolute replay (translation-only, orientation locked)")
        else:
            print("[Pose Unified] Mode: Senior's Local TCP offset with anchored absolute replay")
        WARMUP_DONE = True

    if STEP_IDX >= total_steps:
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim
        return

    end_idx = min(STEP_IDX + CHUNK_SIZE, total_steps - 1)

    for i in range(STEP_IDX, end_idx):
        sim_obs = TASK_ENV.get_obs()
        full_action = build_absolute_target_action(poses_raw[i + 1], sim_obs, REPLAY_ARM)
        action_debug = TASK_ENV.take_action(full_action, action_type="ee")
        active_status = None
        if isinstance(action_debug, dict):
            active_status = action_debug.get(f"{REPLAY_ARM}_status")

        if active_status in {"Success", "Skip"}:
            CONSECUTIVE_ACTIVE_FAILS = 0
        else:
            CONSECUTIVE_ACTIVE_FAILS += 1
            print(
                f"[Pose Unified] frame={i+1} active_arm={REPLAY_ARM} "
                f"status={active_status} target_xyz={np.round(full_action[-8 if REPLAY_ARM == 'right' else 0:-5 if REPLAY_ARM == 'right' else 3], 4)}"
            )
            if CONSECUTIVE_ACTIVE_FAILS >= 5:
                print("[Pose Unified] abort replay: active arm planning failed 5 times consecutively")
                TASK_ENV.take_action_cnt = TASK_ENV.step_lim
                return
        LAST_FULL_ACTION = full_action.copy()

    STEP_IDX = end_idx
    if STEP_IDX >= total_steps - 1:
        STEP_IDX = total_steps
        print(f"[Pose Unified] Finished all {total_steps} frames")
        TASK_ENV.take_action_cnt = TASK_ENV.step_lim


def reset_model(model):
    global STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE
    global ANCHOR_SIM_XYZ, ANCHOR_SIM_QUAT_WXYZ, ANCHOR_RAW_XYZ, ANCHOR_RAW_QUAT_XYZW

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    HOLD_GRIPPER_OVERRIDE = getattr(model, "hold_gripper", None)
    ANCHOR_SIM_XYZ = None
    ANCHOR_SIM_QUAT_WXYZ = None
    ANCHOR_RAW_XYZ = None
    ANCHOR_RAW_QUAT_XYZW = None
    model.reset()
