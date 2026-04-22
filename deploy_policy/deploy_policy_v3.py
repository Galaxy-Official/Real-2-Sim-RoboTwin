"""
RoboTwin Replay Policy for unified raw pose data
===============================================

基于师兄建议的精简重构版：
- 使用师兄的 TCP 偏置逻辑：对原始位姿右乘 [R_y(180) * R_z(90)]，修正夹爪朝向。
- 平移仍然使用之前已经验证过的 raw->RoboTwin 统一坐标变换，保证 `pose_3` 到 `pose_5` 在线上继续向前。
- 采用相对增量回放（Relative Replay），保障机器人的初始启动平滑无跳变。
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================
# 配置
# ============================================================

REPLAY_ARM = "right"
RAW_QUAT_ORDER = "xyzw"
TRANSLATION_SCALE = 1.0

# 由 `base_data/pose_0/pose_1/pose_2` 标定得到的统一平移坐标变换。
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


# ============================================================
# 数学与变换工具 (基于 SciPy)
# ============================================================

def wxyz_to_xyzw(q_wxyz):
    """仿真器 wxyz 转 SciPy xyzw"""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q_xyzw):
    """SciPy xyzw 转仿真器 wxyz"""
    q = np.asarray(q_xyzw, dtype=np.float64)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def raw_quat_to_scipy_xyzw(raw_quat):
    """统一原始四元数格式为 SciPy 需要的 xyzw"""
    raw_quat = np.asarray(raw_quat, dtype=np.float64)
    if RAW_QUAT_ORDER == "xyzw":
        return raw_quat
    if RAW_QUAT_ORDER == "wxyz":
        return wxyz_to_xyzw(raw_quat)
    raise ValueError(f"Unsupported raw quaternion order: {RAW_QUAT_ORDER}")


def ensure_quat_continuity(quat_wxyz, ref_quat_wxyz):
    """保证四元数符号连续性"""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    ref_quat_wxyz = np.asarray(ref_quat_wxyz, dtype=np.float64)
    if np.dot(quat_wxyz, ref_quat_wxyz) < 0:
        return -quat_wxyz
    return quat_wxyz


def compute_pose_transform(xyz, quat_xyzw):
    """
    修正末端坐标系
    """
    # 提取原始旋转
    r_raw = R.from_quat(quat_xyzw)
    
    # 绕Y转180，再绕Z转90
    r_y = R.from_euler('y', 180, degrees=True)
    #r_z = R.from_euler('z', -90, degrees=True)
    r_additional = r_y * r_z
    
    # 右乘，平移向量 xyz 保持不变
    r_out = r_raw * r_additional
    
    return xyz, r_out.as_quat()


def transform_translation_delta(delta_xyz):
    """把原始采集坐标系下的平移增量映射到 RoboTwin 世界系。"""
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
    return TRANSLATION_SCALE * (RAW_TO_WORLD_ROT @ delta_xyz)


def estimate_quat_drift_deg(raw_quat):
    """评估漂移量"""
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
    print(f"[Pose Unified] Estimated fps: {fps:.2f}")

    return {
        "poses": poses,
        "times": times,
        "length": len(poses),
        "fps": fps,
    }


# ============================================================
# RoboTwin 接口与核心逻辑
# ============================================================

def get_active_arm_pose(sim_obs, arm):
    """获取仿真器中的位姿"""
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
    """锁定非活动手臂"""
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
    """提取录制数据"""
    xyz = np.asarray(raw_pose_7d[:3], dtype=np.float64)
    quat_xyzw = raw_quat_to_scipy_xyzw(raw_pose_7d[3:7])
    return xyz, quat_xyzw


def build_relative_target_action(curr_pose_7d, next_pose_7d, sim_obs, arm):
    """
    使用增量方法，计算robotwin中目标位姿
    """
    global ACTIVE_GRIPPER_HOLD, IDLE_ACTION

    # 1. 提取原始数据，修正偏置(y180, z90)
    curr_xyz, curr_quat_xyzw = extract_source_pose(curr_pose_7d)
    curr_xyz, curr_quat_xyzw = compute_pose_transform(curr_xyz, curr_quat_xyzw)

    next_xyz, next_quat_xyzw = extract_source_pose(next_pose_7d)
    next_xyz, next_quat_xyzw = compute_pose_transform(next_xyz, next_quat_xyzw)

    # 2. 获取仿真器当前绝对状态
    sim_xyz, sim_quat_wxyz, _ = get_active_arm_pose(sim_obs, arm)

    # 3. 处理平移：计算增量，加入 raw->world 映射，确定目标位置
    translation_delta = next_xyz - curr_xyz
    target_xyz = sim_xyz + transform_translation_delta(translation_delta)

    # 4. 处理旋转：计算相对增量，确定目标姿态

    # 4.1 创建旋转对象
    r_curr = R.from_quat(curr_quat_xyzw)
    r_next = R.from_quat(next_quat_xyzw)
    
    # 4.2 提取局部的旋转增量
    r_rel_local = r_curr.inv() * r_next
    
    # 4.3 计算目标姿态
    r_sim = R.from_quat(wxyz_to_xyzw(sim_quat_wxyz))
    r_target = r_sim * r_rel_local
    
    # 5. 转换回 robotwin 需要的 wxyz 格式，处理符号连续性
    target_quat_wxyz = xyzw_to_wxyz(r_target.as_quat())
    target_quat_wxyz = ensure_quat_continuity(target_quat_wxyz, sim_quat_wxyz)

    # 6. 组装动作指令
    active_action = np.concatenate([target_xyz, target_quat_wxyz, [ACTIVE_GRIPPER_HOLD]])
    if arm == "left":
        return np.concatenate([active_action, IDLE_ACTION])
    return np.concatenate([IDLE_ACTION, active_action])


def encode_obs(observation):
    return observation


def get_model(usr_args):
    global REPLAY_DATA, STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, REPLAY_ARM, RAW_QUAT_ORDER
    global TRANSLATION_SCALE, HOLD_GRIPPER_OVERRIDE

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    IDLE_ACTION = None
    ACTIVE_GRIPPER_HOLD = None

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
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE

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
        IDLE_ACTION = get_idle_arm_action(sim_obs, REPLAY_ARM)
        full_action = build_relative_target_action(poses_raw[0], poses_raw[0], sim_obs, REPLAY_ARM)
        TASK_ENV.take_action(full_action, action_type="ee")
        LAST_FULL_ACTION = full_action.copy()
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
        _, _, active_gripper = get_active_arm_pose(sim_obs, REPLAY_ARM)
        ACTIVE_GRIPPER_HOLD = (
            HOLD_GRIPPER_OVERRIDE if HOLD_GRIPPER_OVERRIDE is not None else active_gripper
        )
        IDLE_ACTION = get_idle_arm_action(sim_obs, REPLAY_ARM)

        print("[Pose Unified] Mode: Senior's Local TCP offset with incremental replay")
        WARMUP_DONE = True

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

    STEP_IDX = end_idx
    if STEP_IDX >= total_steps - 1:
        STEP_IDX = total_steps
        print(f"[Pose Unified] Finished all {total_steps} frames")


def reset_model(model):
    global STEP_IDX, WARMUP_DONE, LAST_FULL_ACTION
    global IDLE_ACTION, ACTIVE_GRIPPER_HOLD, HOLD_GRIPPER_OVERRIDE

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    IDLE_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    HOLD_GRIPPER_OVERRIDE = getattr(model, "hold_gripper", None)
    model.reset()
