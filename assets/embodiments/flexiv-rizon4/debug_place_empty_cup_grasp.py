#!/usr/bin/env python3
"""Debug grasp candidate generation for Flexiv Rizon 4 in RoboTwin.

This script reproduces the `place_empty_cup` grasp setup for one seed and
prints the exact candidate poses that RoboTwin sends to cuRobo/MPLib after
applying:

- contact point -> seed grasp pose conversion
- embodiment `rotate_lim`
- `_trans_from_gripper_to_endlink()`
- world -> robot-base frame conversion

It is meant to run inside a full RoboTwin checkout on the target server.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import sapien.core as sapien
import transforms3d as t3d
import yaml


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs import CONFIGS_PATH  # noqa: E402
import envs._GLOBAL_CONFIGS as GLOBAL_CONFIGS  # noqa: E402


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


def pose_to_list(pose: sapien.Pose) -> list[float]:
    return pose.p.tolist() + pose.q.tolist()


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


def compute_seed_pose(actor, contact_point_id: int, pre_dis: float) -> tuple[list[float], list[float]]:
    contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
    if contact_matrix is None:
        raise ValueError(f"Contact point {contact_point_id} is missing matrix data")

    global_contact_pose_matrix = contact_matrix @ np.array(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    global_contact_pose_matrix_q = global_contact_pose_matrix[:3, :3]
    global_grasp_pose_p = (
        global_contact_pose_matrix[:3, 3]
        + global_contact_pose_matrix_q @ np.array([-0.12 - pre_dis, 0, 0], dtype=np.float64).T
    )
    global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
    res_pose = list(global_grasp_pose_p) + list(global_grasp_pose_q)
    center_pose = actor.get_contact_point(contact_point_id, "list")
    return res_pose, center_pose


def world_to_base(planner, base_pose: sapien.Pose, target_pose: sapien.Pose):
    world_base_pose = np.concatenate([np.array(base_pose.p), np.array(base_pose.q)])
    world_target_pose = np.concatenate([np.array(target_pose.p), np.array(target_pose.q)])
    return planner._trans_from_world_to_base(world_base_pose, world_target_pose)


def debug_contact_point(env, arm_tag: str, contact_point_id: int, pre_dis: float, limit: int):
    actor = env.cup
    seed_pose, center_pose = compute_seed_pose(actor, contact_point_id, pre_dis=pre_dis)
    target_lst = env.robot.create_target_pose_list(seed_pose, center_pose, arm_tag)

    if arm_tag == "left":
        curobo_planner = env.robot.left_planner
        mplib_planner = getattr(env.robot, "left_mplib_planner", None)
        full_qpos = env.robot.left_entity.get_qpos()
        arm_qpos = extract_arm_planner_qpos(
            env.robot.left_entity,
            env.robot.left_arm_joints_name,
        )
        base_pose = env.robot.left_entity_origion_pose
        rotate_lim = env.robot.left_rotate_lim
    else:
        curobo_planner = env.robot.right_planner
        mplib_planner = getattr(env.robot, "right_mplib_planner", None)
        full_qpos = env.robot.right_entity.get_qpos()
        arm_qpos = extract_arm_planner_qpos(
            env.robot.right_entity,
            env.robot.right_arm_joints_name,
        )
        base_pose = env.robot.right_entity_origion_pose
        rotate_lim = env.robot.right_rotate_lim

    rotate_step = (rotate_lim[1] - rotate_lim[0]) / GLOBAL_CONFIGS.ROTATE_NUM

    print("=" * 80)
    print(f"arm_tag: {arm_tag}")
    print(f"contact_point_id: {contact_point_id}")
    print(f"center_pose: {round_list(center_pose)}")
    print(f"seed_pose:   {round_list(seed_pose)}")
    print(f"base_pose:   {round_list(pose_to_list(base_pose))}")
    print(f"rotate_lim:  {rotate_lim}, rotate_step={rotate_step}")
    print(f"qpos_dim:    full={len(full_qpos)}, curobo={len(arm_qpos)}")
    print("=" * 80)

    for idx, target_pose in enumerate(target_lst[:limit]):
        endlink_pose = env.robot._trans_from_gripper_to_endlink(target_pose, arm_tag=arm_tag)
        base_target_pose_p, base_target_pose_q = world_to_base(curobo_planner, base_pose, endlink_pose)

        curobo_result = curobo_planner.plan_path(arm_qpos, endlink_pose, arms_tag=arm_tag)
        curobo_status = curobo_result.get("status", "Unknown")

        mplib_status = "Unavailable"
        if mplib_planner is not None:
            mplib_result = mplib_planner.plan_path(full_qpos, endlink_pose, arms_tag=arm_tag, log=False)
            mplib_status = mplib_result.get("status", "Unknown")

        theta = rotate_step * idx + rotate_lim[0]
        print(f"[candidate {idx}] theta={round(theta, 6)}")
        print(f"  grasp_target_world: {round_list(target_pose)}")
        print(f"  endlink_world:      {round_list(pose_to_list(endlink_pose))}")
        print(
            "  endlink_base:       "
            f"{round_list(base_target_pose_p.tolist() + base_target_pose_q.tolist())}"
        )
        print(f"  curobo={curobo_status}, mplib={mplib_status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--arm", choices=["left", "right", "auto"], default="auto")
    parser.add_argument("--contact-point-id", type=int, default=None)
    parser.add_argument("--pre-grasp-dis", type=float, default=0.1)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--all-contacts", action="store_true")
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

        if cli_args.all_contacts:
            contact_ids = [idx for idx, _ in env.cup.iter_contact_points()]
        elif cli_args.contact_point_id is not None:
            contact_ids = [cli_args.contact_point_id]
        else:
            # Match place_empty_cup.play_once()
            contact_ids = [[0], [2]][int(arm_tag == "left")]

        print(f"task={cli_args.task_name}, task_config={cli_args.task_config}, seed={cli_args.seed}")
        print(f"cup_pose={round_list(cup_pose.tolist())}, chosen_arm={arm_tag}, contacts={contact_ids}")
        print()

        for contact_id in contact_ids:
            debug_contact_point(
                env,
                arm_tag=arm_tag,
                contact_point_id=contact_id,
                pre_dis=cli_args.pre_grasp_dis,
                limit=cli_args.candidate_limit,
            )
            print()
    finally:
        if hasattr(env, "close_env"):
            env.close_env()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
