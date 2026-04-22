"""
RoboTwin Replay Policy for unified raw pose data
===============================================

适用于同一套采集设置下的原始 `pose_*.npz` 数据。

基于 `base_data/pose_0`(向前), `base_data/pose_1`(向左), `base_data/pose_2`(向上)
三组平移标定数据，先拟合出一套固定的 raw->RoboTwin 刚体旋转矩阵。
`base_data/pose_3` 到 `pose_8` 的旋转数据再用来验证这套统一矩阵在姿态增量上也是自洽的。

统一语义约定为:
  - raw left   -> RoboTwin left
  - raw up     -> RoboTwin up
  - raw forward-> RoboTwin forward

实现上不再依赖理想化的轴交换，而是使用从标定数据拟合得到的旋转矩阵。
这样后续同一采集设置下的数据, 无论是平移、俯仰、翻转还是绕工具轴旋转,
都共用同一套坐标变换。

RoboTwin 世界坐标系按当前任务环境中的语义使用:
  - world x: right
  - world y: forward
  - world z: up

回放策略:
  - 继续使用相邻两帧的增量 replay，保证平滑
  - 平移增量用统一标定矩阵映射到世界系
  - 姿态完整保留，不强行固定末端姿态
  - 平移继续使用相邻帧增量；旋转则以 raw 首帧和 sim warmup 姿态为锚点做绝对对齐
  - 这样可以补上 raw 末端坐标系与 RoboTwin 工具坐标系之间的固定姿态偏置
"""

from pathlib import Path

import numpy as np


# ============================================================
# 配置
# ============================================================

REPLAY_ARM = "right"
# 由 `pose_3` 到 `pose_8` 的旋转标定数据验证，原始四元数以 `xyzw` 解释最自洽。
RAW_QUAT_ORDER = "xyzw"
TRANSLATION_SCALE = 1.0

# 由 `base_data/pose_0/pose_1/pose_2` 的主运动方向拟合得到的统一标定矩阵。
# 语义上对应:
#   raw left    -> world left
#   raw up      -> world up
#   raw forward -> world forward
RAW_TO_WORLD_ROT = np.array(
    [
        [-0.997494312585, 0.016101908673, 0.068889947724],
        [0.067998412082, -0.050562147486, 0.996403374741],
        [0.019527219839, 0.998591106399, 0.049340550328],
    ],
    dtype=np.float64,
)
WORLD_TO_RAW_ROT = RAW_TO_WORLD_ROT.T

HOME_JOINTS_RIGHT = np.array([0.2125, 1.8699, 1.6716, -1.3578, 0.0003, 0.2175], dtype=np.float64)
HOME_JOINTS_LEFT = np.array([0.2125, 1.8699, 1.6716, -1.3578, 0.0003, 0.2175], dtype=np.float64)
WARMUP_STEPS = 60
CHUNK_SIZE = 10


# ============================================================
# 全局状态
# ============================================================

REPLAY_DATA = None
STEP_IDX = 0
WARMUP_DONE = False
LAST_FULL_ACTION = None
IDLE_ACTION = None
ACTIVE_GRIPPER_HOLD = None
HOLD_GRIPPER_OVERRIDE = None
SIM_REF_ROT = None
RAW_REF_ROT = None


# ============================================================
# 数学工具
# ============================================================

def normalize_quat(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def raw_quat_to_wxyz(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64)
    if RAW_QUAT_ORDER == "xyzw":
        return normalize_quat(quat_xyzw_to_wxyz(raw_quat))
    if RAW_QUAT_ORDER == "wxyz":
        return normalize_quat(raw_quat)
    raise ValueError(f"Unsupported raw quaternion order: {RAW_QUAT_ORDER}")


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = normalize_quat(q1)
    w2, x2, y2, z2 = normalize_quat(q2)
    return normalize_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
    )


def quat_to_rotmat(q):
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quat(np.array([w, x, y, z], dtype=np.float64))


def transform_translation_delta(delta_xyz):
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
    return TRANSLATION_SCALE * (RAW_TO_WORLD_ROT @ delta_xyz)


def map_raw_world_rotation_to_sim_world(raw_world_rot):
    raw_world_rot = np.asarray(raw_world_rot, dtype=np.float64)
    return RAW_TO_WORLD_ROT @ raw_world_rot @ WORLD_TO_RAW_ROT


def build_absolute_target_quat(raw_quat_wxyz):
    if SIM_REF_ROT is None or RAW_REF_ROT is None:
        raise RuntimeError("SIM_REF_ROT / RAW_REF_ROT is not initialized before orientation replay")

    raw_rot = quat_to_rotmat(raw_quat_wxyz)
    raw_world_delta = raw_rot @ RAW_REF_ROT.T
    sim_world_delta = map_raw_world_rotation_to_sim_world(raw_world_delta)
    target_rot = sim_world_delta @ SIM_REF_ROT
    return rotmat_to_quat(target_rot)


def estimate_quat_drift_deg(raw_quat):
    quat = np.asarray(raw_quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    dots = np.abs(np.sum(quat * quat[0], axis=1))
    angles = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    return np.degrees(angles)


# ============================================================
# 数据读取
# ============================================================

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

    xyz = poses[:, :3].astype(np.float64)
    quat = poses[:, 3:].astype(np.float64)
    quat_drift_deg = estimate_quat_drift_deg(quat)

    if len(times) > 1:
        dt = np.diff(times)
        dt = dt[dt > 1e-6]
        fps = float(1.0 / np.median(dt)) if len(dt) > 0 else 24.0
    else:
        fps = 24.0

    print(f"[Pose Unified] Loaded {len(poses)} frames from {len(pose_files)} file(s)")
    print(f"[Pose Unified] Files: {[str(p) for p in pose_files]}")
    print(f"[Pose Unified] Estimated fps: {fps:.2f}")
    print(f"[Pose Unified] xyz first: {np.round(xyz[0], 6)}")
    print(f"[Pose Unified] xyz last : {np.round(xyz[-1], 6)}")
    print(f"[Pose Unified] xyz delta: {np.round(xyz[-1] - xyz[0], 6)}")
    print(f"[Pose Unified] xyz std  : {np.round(xyz.std(axis=0), 6)}")
    print(
        "[Pose Unified] quat drift deg "
        f"(mean/max): {quat_drift_deg.mean():.4f}/{quat_drift_deg.max():.4f}"
    )

    return {
        "poses": poses,
        "times": times,
        "length": len(poses),
        "fps": fps,
    }


# ============================================================
# RoboTwin 接口
# ============================================================

def get_active_arm_pose(sim_obs, arm):
    if arm == "right":
        endpose = sim_obs["endpose"]["right_endpose"]
        gripper = float(sim_obs["endpose"]["right_gripper"])
    else:
        endpose = sim_obs["endpose"]["left_endpose"]
        gripper = float(sim_obs["endpose"]["left_gripper"])
    xyz = np.array(endpose[:3], dtype=np.float64)
    quat = normalize_quat(np.array(endpose[3:7], dtype=np.float64))
    return xyz, quat, gripper


def get_idle_arm_action(sim_obs, active_arm):
    if active_arm == "right":
        idle_pose = sim_obs["endpose"]["left_endpose"][:7]
        idle_grip = float(sim_obs["endpose"]["left_gripper"])
    else:
        idle_pose = sim_obs["endpose"]["right_endpose"][:7]
        idle_grip = float(sim_obs["endpose"]["right_gripper"])
    idle_pose = np.array(idle_pose, dtype=np.float64)
    idle_pose[3:7] = normalize_quat(idle_pose[3:7])
    return np.concatenate([idle_pose, [idle_grip]])


def extract_source_pose(raw_pose_7d):
    xyz = np.asarray(raw_pose_7d[:3], dtype=np.float64)
    quat = raw_quat_to_wxyz(raw_pose_7d[3:7])
    return xyz, quat


def build_relative_target_action(curr_pose_7d, next_pose_7d, sim_obs, arm):
    global ACTIVE_GRIPPER_HOLD, IDLE_ACTION

    curr_xyz, _ = extract_source_pose(curr_pose_7d)
    next_xyz, next_quat = extract_source_pose(next_pose_7d)

    sim_xyz, _, _ = get_active_arm_pose(sim_obs, arm)

    translation_delta = next_xyz - curr_xyz
    target_xyz = sim_xyz + transform_translation_delta(translation_delta)

    target_quat = build_absolute_target_quat(next_quat)

    active_action = np.concatenate([target_xyz, target_quat, [ACTIVE_GRIPPER_HOLD]])
    if arm == "left":
        return np.concatenate([active_action, IDLE_ACTION])
    return np.concatenate([IDLE_ACTION, active_action])


def encode_obs(observation):
    return observation


def get_model(usr_args):
    global REPLAY_DATA, STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, REPLAY_ARM, RAW_QUAT_ORDER
    global TRANSLATION_SCALE, HOLD_GRIPPER_OVERRIDE, SIM_REF_ROT, RAW_REF_ROT

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    IDLE_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    SIM_REF_ROT = None
    RAW_REF_ROT = None

    data_dir = usr_args["data_dir"]
    episode_index = int(usr_args.get("episode_index", 0))
    REPLAY_ARM = usr_args.get("replay_arm", "right")
    RAW_QUAT_ORDER = usr_args.get("raw_quat_order", "xyzw")
    TRANSLATION_SCALE = float(usr_args.get("translation_scale", 1.0))
    hold_gripper = usr_args.get("hold_gripper")
    HOLD_GRIPPER_OVERRIDE = float(hold_gripper) if hold_gripper is not None else None

    REPLAY_DATA = load_raw_pose_replay(data_dir, episode_index)

    print(f"[Pose Unified] Config: arm={REPLAY_ARM}")
    print(f"[Pose Unified] raw_quat_order={RAW_QUAT_ORDER}")
    print(f"[Pose Unified] translation_scale={TRANSLATION_SCALE:.4f}")
    print(f"[Pose Unified] raw_to_world_rot=\n{RAW_TO_WORLD_ROT}")
    print("[Pose Unified] orientation_mode=absolute raw-first-frame anchoring")
    if hold_gripper is not None:
        print(f"[Pose Unified] hold_gripper={float(hold_gripper):.4f}")
    else:
        print("[Pose Unified] hold_gripper=<use sim init gripper>")

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
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE
    global SIM_REF_ROT, RAW_REF_ROT

    if REPLAY_DATA is None and getattr(model, "replay_data", None) is not None:
        REPLAY_DATA = model.replay_data
    if HOLD_GRIPPER_OVERRIDE is None and getattr(model, "hold_gripper", None) is not None:
        HOLD_GRIPPER_OVERRIDE = model.hold_gripper

    if REPLAY_DATA is None:
        raise RuntimeError(
            "REPLAY_DATA is not initialized. Please make sure get_model() runs before eval(), "
            "and reset_model() does not clear replay data."
        )

    poses_raw = REPLAY_DATA["poses"]
    total_steps = REPLAY_DATA["length"]

    if total_steps == 0:
        print("[Pose Unified] Empty sequence, skip replay")
        return

    if not WARMUP_DONE:
        print("[Pose Unified] First call: doing warmup...")

        sim_obs = TASK_ENV.get_obs()
        current_joints = np.array(sim_obs["joint_action"]["vector"], dtype=np.float64)
        target_joints = current_joints.copy()

        if REPLAY_ARM == "right":
            target_joints[7:13] = HOME_JOINTS_RIGHT
            target_joints[13] = 1.0
        else:
            target_joints[0:6] = HOME_JOINTS_LEFT
            target_joints[6] = 1.0

        for i in range(WARMUP_STEPS):
            alpha = (i + 1) / WARMUP_STEPS
            interp_joints = current_joints * (1.0 - alpha) + target_joints * alpha
            TASK_ENV.take_action(interp_joints, action_type="qpos")

        sim_obs = TASK_ENV.get_obs()
        _, sim_ref_quat, active_gripper = get_active_arm_pose(sim_obs, REPLAY_ARM)
        ACTIVE_GRIPPER_HOLD = (
            HOLD_GRIPPER_OVERRIDE if HOLD_GRIPPER_OVERRIDE is not None else active_gripper
        )
        IDLE_ACTION = get_idle_arm_action(sim_obs, REPLAY_ARM)

        first_xyz, first_quat = extract_source_pose(poses_raw[0])
        SIM_REF_ROT = quat_to_rotmat(sim_ref_quat)
        RAW_REF_ROT = quat_to_rotmat(first_quat)
        print(f"[Pose Unified] Hold gripper : {ACTIVE_GRIPPER_HOLD:.4f}")
        print(f"[Pose Unified] User first xyz(raw): {np.round(first_xyz, 6)}")
        print(f"[Pose Unified] User first quat(wxyz): {np.round(first_quat, 6)}")
        print("[Pose Unified] Mode: incremental translation + absolute orientation anchoring")

        WARMUP_DONE = True

    if total_steps == 1:
        sim_obs = TASK_ENV.get_obs()
        full_action = build_relative_target_action(poses_raw[0], poses_raw[0], sim_obs, REPLAY_ARM)
        TASK_ENV.take_action(full_action, action_type="ee")
        LAST_FULL_ACTION = full_action.copy()
        STEP_IDX = total_steps
        return

    if STEP_IDX >= total_steps:
        if LAST_FULL_ACTION is not None:
            TASK_ENV.take_action(LAST_FULL_ACTION, action_type="ee")
        return

    end_idx = min(STEP_IDX + CHUNK_SIZE, total_steps - 1)

    for i in range(STEP_IDX, end_idx):
        sim_obs = TASK_ENV.get_obs()
        full_action = build_relative_target_action(poses_raw[i], poses_raw[i + 1], sim_obs, REPLAY_ARM)
        TASK_ENV.take_action(full_action, action_type="ee")
        LAST_FULL_ACTION = full_action.copy()

        if i % 50 == 0:
            active_action = full_action[:8] if REPLAY_ARM == "left" else full_action[8:]
            print(
                f"[Pose Unified] Step {i}/{total_steps} | "
                f"target=({active_action[0]:.3f},{active_action[1]:.3f},{active_action[2]:.3f})"
            )

    STEP_IDX = end_idx

    if STEP_IDX >= total_steps - 1:
        STEP_IDX = total_steps
        print(f"[Pose Unified] Finished all {total_steps} frames")


def reset_model(model):
    global STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE
    global SIM_REF_ROT, RAW_REF_ROT

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    IDLE_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    SIM_REF_ROT = None
    RAW_REF_ROT = None
    HOLD_GRIPPER_OVERRIDE = getattr(model, "hold_gripper", None)
    model.reset()
