#!/usr/bin/env python3
"""Probe simple base-frame IK targets for Flexiv Rizon 4 in RoboTwin.

This follows the same idea as RoboTwin's "find valid pose" calibration step:
scan a grid of end-effector poses in the robot base frame and report whether
cuRobo / MPLib can solve them at all. This separates embodiment/planner
configuration issues from task-specific grasp pose generation.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import CONFIGS_PATH  # noqa: E402
from envs.robot.planner import MplibPlanner  # noqa: E402


def class_decorator(task_name: str):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_camera_config(camera_type: str):
    camera_config_path = ROOT / "task_config" / "_camera_config.yml"
    with camera_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args[camera_type]


def get_embodiment_config(robot_file: str):
    robot_config_file = ROOT / robot_file / "config.yml"
    with robot_config_file.open("r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def load_task_args(task_name: str, task_config: str) -> dict:
    with (ROOT / "task_config" / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["render_freq"] = 0

    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(CONFIGS_PATH) / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as f:
        embodiment_index = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name: str) -> str:
        robot_file = embodiment_index[embodiment_name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment files for {embodiment_name}")
        return robot_file

    head_camera_type = args["camera"]["head_camera_type"]
    camera_config = get_camera_config(head_camera_type)
    args["head_camera_h"] = camera_config["h"]
    args["head_camera_w"] = camera_config["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def round_list(values, digits: int = 6):
    return [round(float(v), digits) for v in values]


def extract_arm_planner_qpos(entity, arm_joint_names: list[str]) -> np.ndarray:
    active_joints = entity.get_active_joints()
    entity_qpos = entity.get_qpos()
    joint_index = {joint.get_name(): idx for idx, joint in enumerate(active_joints)}
    missing = [name for name in arm_joint_names if name not in joint_index]
    if missing:
        raise KeyError(
            "Could not locate expected arm joints in articulation qpos order: "
            + ", ".join(missing)
        )
    dtype = getattr(entity_qpos, "dtype", np.float32)
    return np.asarray([entity_qpos[joint_index[name]] for name in arm_joint_names], dtype=dtype)


def extract_urdf_joint_order_qpos(entity, urdf_path: str) -> tuple[list[str], np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    urdf_joint_names: list[str] = []
    for joint in root.findall("joint"):
        joint_type = joint.attrib.get("type")
        joint_name = joint.attrib.get("name")
        if not joint_name or joint_type == "fixed":
            continue
        urdf_joint_names.append(joint_name)

    active_joints = entity.get_active_joints()
    entity_qpos = entity.get_qpos()
    joint_index = {joint.get_name(): idx for idx, joint in enumerate(active_joints)}
    filtered_joint_names = [name for name in urdf_joint_names if name in joint_index]
    dtype = getattr(entity_qpos, "dtype", np.float32)
    qpos = np.asarray([entity_qpos[joint_index[name]] for name in filtered_joint_names], dtype=dtype)
    return filtered_joint_names, qpos


def safe_getattr_chain(obj, path: str):
    cur = obj
    for token in path.split("."):
        if cur is None or not hasattr(cur, token):
            return None
        cur = getattr(cur, token)
    return cur


def summarize_collision_results(results) -> list[str]:
    summary: list[str] = []
    for item in results or []:
        if isinstance(item, str):
            summary.append(item)
            continue
        parts = []
        for key in (
            "link_name1",
            "link_name2",
            "object_name1",
            "object_name2",
            "reason",
            "collision_type",
        ):
            if hasattr(item, key):
                value = getattr(item, key)
                if value is not None:
                    parts.append(f"{key}={value}")
        summary.append(", ".join(parts) if parts else repr(item))
    return summary


def build_joint_state_map(joint_names: list[str], qpos: np.ndarray) -> dict[str, float]:
    return {name: round(float(value), 6) for name, value in zip(joint_names, qpos)}


def expected_gn01_mimic_states(finger_width: float) -> dict[str, float]:
    return {
        "left_outer_knuckle_joint": round(9.404 * finger_width - 0.155, 6),
        "left_inner_knuckle_joint": round(9.404 * finger_width - 0.155, 6),
        "left_inner_finger_joint": round(-9.404 * finger_width + 0.155, 6),
        "right_outer_knuckle_joint": round(9.404 * finger_width - 0.155, 6),
        "right_inner_knuckle_joint": round(9.404 * finger_width - 0.155, 6),
        "right_inner_finger_joint": round(-9.404 * finger_width + 0.155, 6),
    }


def force_robot_to_nominal_state(robot, left_gripper_pos: float = 1.0, right_gripper_pos: float = 1.0):
    robot.move_to_homestate()
    robot.left_gripper_val = left_gripper_pos
    robot.right_gripper_val = right_gripper_pos

    def _apply(entity, arm_joints, homestate, gripper, gripper_scale, gripper_val):
        active_joints = entity.get_active_joints()
        joint_index = {joint.get_name(): idx for idx, joint in enumerate(active_joints)}
        qpos = np.array(entity.get_qpos(), dtype=np.float64)
        qvel = np.zeros_like(qpos)

        for joint, target in zip(arm_joints, homestate):
            idx = joint_index[joint.get_name()]
            qpos[idx] = target
            joint.set_drive_target(target)
            joint.set_drive_velocity_target(0.0)

        real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])
        for real_joint, multiplier, offset in gripper:
            idx = joint_index[real_joint.get_name()]
            target = real_gripper_val * multiplier + offset
            qpos[idx] = target
            real_joint.set_drive_target(target)
            real_joint.set_drive_velocity_target(0.0)

        entity.set_qpos(qpos)
        entity.set_qvel(qvel)

    _apply(
        robot.left_entity,
        robot.left_arm_joints,
        robot.left_homestate,
        robot.left_gripper,
        robot.left_gripper_scale,
        robot.left_gripper_val,
    )
    _apply(
        robot.right_entity,
        robot.right_arm_joints,
        robot.right_homestate,
        robot.right_gripper,
        robot.right_gripper_scale,
        robot.right_gripper_val,
    )


def base_pose_to_world(base_pose: sapien.Pose, local_xyz: np.ndarray, local_quat: np.ndarray) -> sapien.Pose:
    base_rot = t3d.quaternions.quat2mat(np.array(base_pose.q, dtype=np.float64))
    local_rot = t3d.quaternions.quat2mat(local_quat)
    world_pos = np.array(base_pose.p, dtype=np.float64) + base_rot @ local_xyz
    world_rot = base_rot @ local_rot
    world_quat = t3d.quaternions.mat2quat(world_rot)
    return sapien.Pose(world_pos, world_quat)


def world_pose_to_base(base_pose: sapien.Pose, world_pose: sapien.Pose) -> list[float]:
    base_rot = t3d.quaternions.quat2mat(np.array(base_pose.q, dtype=np.float64))
    world_rot = t3d.quaternions.quat2mat(np.array(world_pose.q, dtype=np.float64))
    rel_pos = np.array(world_pose.p, dtype=np.float64) - np.array(base_pose.p, dtype=np.float64)
    local_pos = base_rot.T @ rel_pos
    local_rot = base_rot.T @ world_rot
    local_quat = t3d.quaternions.mat2quat(local_rot)
    return local_pos.tolist() + local_quat.tolist()


def linspace_triplet(start: float, end: float, num: int) -> list[float]:
    return np.linspace(start, end, num).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--arm", choices=["left", "right", "auto"], default="auto")
    parser.add_argument("--quat", nargs=4, type=float, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--x-min", type=float, default=0.25)
    parser.add_argument("--x-max", type=float, default=0.55)
    parser.add_argument("--x-num", type=int, default=7)
    parser.add_argument("--y-min", type=float, default=0.05)
    parser.add_argument("--y-max", type=float, default=0.30)
    parser.add_argument("--y-num", type=int, default=6)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=0.35)
    parser.add_argument("--z-num", type=int, default=7)
    parser.add_argument("--max-success", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    args = load_task_args(cli_args.task_name, cli_args.task_config)
    env = class_decorator(cli_args.task_name)
    env.check_stable = lambda: (True, [])
    env.together_open_gripper = lambda *args, **kwargs: True

    try:
        env.setup_demo(now_ep_num=cli_args.episode_id, seed=cli_args.seed, is_test=True, **args)
        force_robot_to_nominal_state(env.robot)

        cup_pose = env.cup.get_pose().p
        arm_tag = cli_args.arm
        if arm_tag == "auto":
            arm_tag = "right" if cup_pose[0] > 0 else "left"

        if arm_tag == "left":
            curobo_planner = env.robot.left_planner
            mplib_planner = getattr(env.robot, "left_mplib_planner", None)
            full_qpos = env.robot.left_entity.get_qpos()
            active_joint_names = [joint.get_name() for joint in env.robot.left_entity.get_active_joints()]
            arm_qpos = extract_arm_planner_qpos(
                env.robot.left_entity,
                env.robot.left_arm_joints_name,
            )
            base_pose = env.robot.left_entity_origion_pose
            base_link = env.robot.left_entity.find_link_by_name("base_link")
            current_link = env.robot.left_entity.find_link_by_name(env.robot.left_move_group)
            move_group_name = env.robot.left_move_group
            ee_joint_name = getattr(env.robot, "left_ee_joint_name", None)
            arm_joint_names = env.robot.left_arm_joints_name
            mplib_joint_names, mplib_qpos = extract_urdf_joint_order_qpos(
                env.robot.left_entity,
                env.robot.left_urdf_path,
            )
            raw_mplib_planner = MplibPlanner(
                env.robot.left_urdf_path,
                env.robot.left_srdf_path,
                env.robot.left_move_group,
                env.robot.left_entity_origion_pose,
                env.robot.left_entity,
                env.robot.left_planner_type,
                scene=None,
            )
        else:
            curobo_planner = env.robot.right_planner
            mplib_planner = getattr(env.robot, "right_mplib_planner", None)
            full_qpos = env.robot.right_entity.get_qpos()
            active_joint_names = [joint.get_name() for joint in env.robot.right_entity.get_active_joints()]
            arm_qpos = extract_arm_planner_qpos(
                env.robot.right_entity,
                env.robot.right_arm_joints_name,
            )
            base_pose = env.robot.right_entity_origion_pose
            base_link = env.robot.right_entity.find_link_by_name("base_link")
            current_link = env.robot.right_entity.find_link_by_name(env.robot.right_move_group)
            move_group_name = env.robot.right_move_group
            ee_joint_name = getattr(env.robot, "right_ee_joint_name", None)
            arm_joint_names = env.robot.right_arm_joints_name
            mplib_joint_names, mplib_qpos = extract_urdf_joint_order_qpos(
                env.robot.right_entity,
                env.robot.right_urdf_path,
            )
            raw_mplib_planner = MplibPlanner(
                env.robot.right_urdf_path,
                env.robot.right_srdf_path,
                env.robot.right_move_group,
                env.robot.right_entity_origion_pose,
                env.robot.right_entity,
                env.robot.right_planner_type,
                scene=None,
            )

        x_values = linspace_triplet(cli_args.x_min, cli_args.x_max, cli_args.x_num)
        y_values = linspace_triplet(cli_args.y_min, cli_args.y_max, cli_args.y_num)
        z_values = linspace_triplet(cli_args.z_min, cli_args.z_max, cli_args.z_num)
        quat = np.array(cli_args.quat, dtype=np.float64)
        quat = quat / np.linalg.norm(quat)

        print(f"task={cli_args.task_name}, task_config={cli_args.task_config}, seed={cli_args.seed}")
        print(f"cup_pose={round_list(cup_pose.tolist())}, chosen_arm={arm_tag}")
        print(f"base_pose={round_list(list(base_pose.p) + list(base_pose.q))}")
        print(f"quat(base_frame)={round_list(quat.tolist())}")
        print(f"move_group={move_group_name}, ee_joint={ee_joint_name}")
        print(f"arm_joint_names={arm_joint_names}")
        print(f"active_joint_names={active_joint_names}")
        active_joint_state_map = build_joint_state_map(active_joint_names, full_qpos)
        print(f"active_joint_qpos={active_joint_state_map}")
        if "finger_width_joint" in active_joint_state_map:
            expected_mimic = expected_gn01_mimic_states(active_joint_state_map["finger_width_joint"])
            print(f"expected_gn01_mimic_qpos={expected_mimic}")
        print(
            "curobo_joint_names="
            f"{safe_getattr_chain(curobo_planner, 'joint_names') or safe_getattr_chain(curobo_planner, 'motion_gen.kinematics.joint_names')}"
        )
        print(
            "mplib_joint_names="
            f"{safe_getattr_chain(mplib_planner, 'joint_names') or safe_getattr_chain(mplib_planner, 'planner.joint_names') or mplib_joint_names}"
        )
        print(f"qpos_dim: full={len(full_qpos)}, curobo={len(arm_qpos)}, mplib={len(mplib_qpos)}")
        if base_link is not None:
            sim_base_pose = base_link.entity_pose
            print(f"sim_base_link_world={round_list(list(sim_base_pose.p) + list(sim_base_pose.q))}")
            print(f"configured_base_pose={round_list(list(base_pose.p) + list(base_pose.q))}")
        print(
            f"scan: x in [{cli_args.x_min}, {cli_args.x_max}] x{cli_args.x_num}, "
            f"y in [{cli_args.y_min}, {cli_args.y_max}] x{cli_args.y_num}, "
            f"z in [{cli_args.z_min}, {cli_args.z_max}] x{cli_args.z_num}"
        )
        if current_link is not None:
            current_world_pose = current_link.entity_pose
            current_base_pose = world_pose_to_base(base_pose, current_world_pose)
            current_wrap_ok = "Unavailable"
            current_self_collisions = []
            planner_core = getattr(raw_mplib_planner, "planner", None)
            if planner_core is not None:
                if hasattr(planner_core, "wrap_joint_limit"):
                    qpos_for_wrap = np.array(mplib_qpos, copy=True)
                    current_wrap_ok = planner_core.wrap_joint_limit(qpos_for_wrap)
                if hasattr(planner_core, "check_for_self_collision"):
                    current_self_collisions = summarize_collision_results(
                        planner_core.check_for_self_collision(mplib_qpos)
                    )
            current_curobo = curobo_planner.plan_path(arm_qpos, current_world_pose, arms_tag=arm_tag).get(
                "status",
                "Unknown",
            )
            if mplib_planner is not None:
                current_mplib = mplib_planner.plan_path(mplib_qpos, current_world_pose, arms_tag=arm_tag, log=False).get(
                    "status",
                    "Unknown",
                )
            else:
                current_mplib = "Unavailable"
            current_raw_mplib = raw_mplib_planner.plan_path(
                mplib_qpos,
                current_world_pose,
                arms_tag=arm_tag,
                log=False,
            ).get("status", "Unknown")
            print(f"current_endlink_world={round_list(list(current_world_pose.p) + list(current_world_pose.q))}")
            print(f"current_endlink_base={round_list(current_base_pose)}")
            print(f"current_pose_plan_status: curobo={current_curobo}, mplib={current_mplib}")
            print(f"current_pose_raw_mplib_status: {current_raw_mplib}")
            print(f"current_raw_mplib_wrap_joint_limit={current_wrap_ok}")
            print(f"current_raw_mplib_self_collisions={current_self_collisions}")
        print()

        success_count = 0
        total_count = 0
        for x in x_values:
            for y in y_values:
                for z in z_values:
                    total_count += 1
                    world_pose = base_pose_to_world(base_pose, np.array([x, y, z], dtype=np.float64), quat)
                    curobo_result = curobo_planner.plan_path(arm_qpos, world_pose, arms_tag=arm_tag)
                    curobo_status = curobo_result.get("status", "Unknown")

                    mplib_status = "Unavailable"
                    if mplib_planner is not None:
                        mplib_result = mplib_planner.plan_path(mplib_qpos, world_pose, arms_tag=arm_tag, log=False)
                        mplib_status = mplib_result.get("status", "Unknown")

                    if curobo_status == "Success" or mplib_status == "Success":
                        success_count += 1
                        print(
                            f"SUCCESS base_pose={[round(x, 4), round(y, 4), round(z, 4)]} "
                            f"world_pose={round_list(world_pose.p.tolist() + world_pose.q.tolist())} "
                            f"curobo={curobo_status} mplib={mplib_status}"
                        )
                        if success_count >= cli_args.max_success:
                            print(f"Reached max_success={cli_args.max_success}, stopping early.")
                            return 0

        print(f"Finished {total_count} probes, success_count={success_count}")
        if success_count == 0:
            print("No valid IK target was found in the scanned base-frame region.")
        return 0
    finally:
        if hasattr(env, "close_env"):
            env.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
