"""
RoboTwin Replay Policy for unified raw pose data
===============================================

适用于同一套采集设置下的原始 `pose_*.npz` 数据。
【重构版】：全面使用 scipy.spatial.transform.Rotation 替代手写数学库，保证数值稳定与优雅。

统一语义约定为:
  - raw left   -> RoboTwin left
  - raw up     -> RoboTwin up
  - raw forward-> RoboTwin forward

回放策略:
  - 继续使用相邻两帧的增量 replay，保证平滑
  - 平移增量用统一标定矩阵映射到世界系
  - 姿态增量使用 SciPy 处理，通过标定矩阵完成相似变换
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================
# 配置
# ============================================================

REPLAY_ARM = "right"
# 由 `pose_3` 到 `pose_8` 的旋转标定数据验证，原始四元数以 `xyzw` 解释最自洽。
RAW_QUAT_ORDER = "xyzw"
TRANSLATION_SCALE = 1.0

# 由 `base_data/pose_0/pose_1/pose_2` 的主运动方向拟合得到的统一标定矩阵。
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
    """将采集的原始四元数统一转为 SciPy 需要的 xyzw 格式"""
    raw_quat = np.asarray(raw_quat, dtype=np.float64)
    if RAW_QUAT_ORDER == "xyzw":
        return raw_quat
    if RAW_QUAT_ORDER == "wxyz":
        return wxyz_to_xyzw(raw_quat)
    raise ValueError(f"Unsupported raw quaternion order: {RAW_QUAT_ORDER}")


def ensure_quat_continuity(quat_wxyz, ref_quat_wxyz):
    """保证四元数符号连续性 (防止出现 360 度翻转导致的剧烈运动)"""
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    ref_quat_wxyz = np.asarray(ref_quat_wxyz, dtype=np.float64)
    if np.dot(quat_wxyz, ref_quat_wxyz) < 0:
        return -quat_wxyz
    return quat_wxyz


def transform_translation_delta(delta_xyz):
    """将设备系下的平移增量映射到世界系"""
    delta_xyz = np.asarray(delta_xyz, dtype=np.float64)
    return TRANSLATION_SCALE * (RAW_TO_WORLD_ROT @ delta_xyz)


def estimate_quat_drift_deg(raw_quat):
    """评估原始序列的总角度漂移量 (用于打分和Debug)"""
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
# RoboTwin 接口与核心逻辑
# ============================================================

def get_active_arm_pose(sim_obs, arm):
    """获取活动手臂在仿真器中的位姿 (返回格式: xyz, quat_wxyz, gripper)"""
    if arm == "right":
        endpose = sim_obs["endpose"]["right_endpose"]
        gripper = float(sim_obs["endpose"]["right_gripper"])
    else:
        endpose = sim_obs["endpose"]["left_endpose"]
        gripper = float(sim_obs["endpose"]["left_gripper"])
    xyz = np.array(endpose[:3], dtype=np.float64)
    quat_wxyz = np.array(endpose[3:7], dtype=np.float64)
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz) # 确保单位四元数
    return xyz, quat_wxyz, gripper


def get_idle_arm_action(sim_obs, active_arm):
    """锁定非活动手臂保持当前状态"""
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
    """提取录制数据，统一输出 xyzw 供 SciPy 使用"""
    xyz = np.asarray(raw_pose_7d[:3], dtype=np.float64)
    quat_xyzw = raw_quat_to_scipy_xyzw(raw_pose_7d[3:7])
    return xyz, quat_xyzw


def build_relative_target_action(curr_pose_7d, next_pose_7d, sim_obs, arm):
    """基于 SciPy 的核心：计算相对位姿并映射到机器人世界系"""
    global ACTIVE_GRIPPER_HOLD, IDLE_ACTION

    # 1. 解析原始数据
    curr_xyz, curr_quat_xyzw = extract_source_pose(curr_pose_7d)
    next_xyz, next_quat_xyzw = extract_source_pose(next_pose_7d)

    # 2. 获取仿真器当前绝对状态
    sim_xyz, sim_quat_wxyz, _ = get_active_arm_pose(sim_obs, arm)

    # 3. 处理平移：直接作差并进行基变换
    translation_delta = next_xyz - curr_xyz
    target_xyz = sim_xyz + transform_translation_delta(translation_delta)

    # 4. 处理旋转：使用 SciPy (极其优雅)
    r_curr = R.from_quat(curr_quat_xyzw)
    r_next = R.from_quat(next_quat_xyzw)
    
    # 4.1 提取设备系下的相对旋转矩阵
    r_rel_mat = (r_curr.inv() * r_next).as_matrix()
    
    # 4.2 将相对旋转矩阵变换到世界系 (相似变换)
    r_rel_world_mat = RAW_TO_WORLD_ROT @ r_rel_mat @ WORLD_TO_RAW_ROT
    r_rel_world = R.from_matrix(r_rel_world_mat)
    
    # 4.3 结合机器人当前姿态，计算目标姿态
    r_sim = R.from_quat(wxyz_to_xyzw(sim_quat_wxyz))
    r_target = r_sim * r_rel_world
    
    # 4.4 转换回仿真器需要的 wxyz 格式并处理跳变
    target_quat_wxyz = xyzw_to_wxyz(r_target.as_quat())
    target_quat_wxyz = ensure_quat_continuity(target_quat_wxyz, sim_quat_wxyz)

    # 5. 组装动作指令
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

    print(f"[Pose Unified - SciPy] Config: arm={REPLAY_ARM}")
    print(f"[Pose Unified - SciPy] raw_quat_order={RAW_QUAT_ORDER}")
    print(f"[Pose Unified - SciPy] translation_scale={TRANSLATION_SCALE:.4f}")
    print(f"[Pose Unified - SciPy] raw_to_world_rot=\n{RAW_TO_WORLD_ROT}")
    if hold_gripper is not None:
        print(f"[Pose Unified - SciPy] hold_gripper={float(hold_gripper):.4f}")
    else:
        print("[Pose Unified - SciPy] hold_gripper=<use sim init gripper>")

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
        raise RuntimeError(
            "REPLAY_DATA is not initialized. Please make sure get_model() runs before eval(), "
            "and reset_model() does not clear replay data."
        )

    poses_raw = REPLAY_DATA["poses"]
    total_steps = REPLAY_DATA["length"]

    if total_steps == 0:
        print("[Pose Unified] Empty sequence, skip replay")
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

        first_xyz, first_quat = extract_source_pose(poses_raw[0])
        print(f"[Pose Unified] Hold gripper : {ACTIVE_GRIPPER_HOLD:.4f}")
        print(f"[Pose Unified] User first xyz(raw): {np.round(first_xyz, 6)}")
        print(f"[Pose Unified] User first quat(xyzw): {np.round(first_quat, 6)}")
        print("[Pose Unified] Mode: incremental relative pose replay with unified raw->world mapping")

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

    STEP_IDX = 0
    WARMUP_DONE = False
    LAST_FULL_ACTION = None
    IDLE_ACTION = None
    ACTIVE_GRIPPER_HOLD = None
    HOLD_GRIPPER_OVERRIDE = getattr(model, "hold_gripper", None)
    model.reset()