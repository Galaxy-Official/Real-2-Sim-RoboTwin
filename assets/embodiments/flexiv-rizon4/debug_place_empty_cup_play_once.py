#!/usr/bin/env python3
"""Step-by-step debug runner for place_empty_cup expert logic.

This script reproduces the main expert sequence used by `place_empty_cup`
and prints the exact action targets and whether each stage succeeds. It is
intended to identify where `collect_data.sh place_empty_cup ...` fails for
Flexiv Rizon4.
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np

from debug_place_empty_cup_grasp import (
    class_decorator,
    force_robot_to_nominal_state,
    load_task_args,
    round_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--skip-stable-check", action="store_true")
    parser.add_argument("--contact-point-id", type=int, default=None)
    parser.add_argument("--grasp-dis", type=float, default=0.0)
    parser.add_argument("--pre-grasp-dis", type=float, default=0.10)
    parser.add_argument("--gripper-pos", type=float, default=0.0)
    return parser.parse_args()


def pose_to_str(pose):
    if pose is None:
        return "None"
    return str(round_list(pose))


def print_action_block(step_name: str, action_bundle):
    arm_tag, actions = action_bundle
    print(f"[{step_name}] arm={arm_tag}, num_actions={len(actions)}")
    for idx, action in enumerate(actions):
        target_pose = getattr(action, "target_pose", None)
        target_gripper = getattr(action, "target_gripper_pos", None)
        print(
            f"  action[{idx}] type={action.action} "
            f"target_pose={pose_to_str(target_pose)} "
            f"target_gripper={target_gripper}"
        )


def run_step(env, step_name: str, action_bundle):
    print_action_block(step_name, action_bundle)
    result = env.move(action_bundle)
    print(
        f"[{step_name}] move_result={result} "
        f"plan_success={env.plan_success} eval_success={getattr(env, 'eval_success', None)}"
    )
    if hasattr(env, "last_action_debug"):
        print(f"[{step_name}] last_action_debug={env.last_action_debug}")
    print(
        f"[{step_name}] left_ee={round_list(env.robot.get_left_ee_pose())} "
        f"right_ee={round_list(env.robot.get_right_ee_pose())}"
    )
    cup_fp = env.cup.get_functional_point(0, "pose").p
    coaster_fp = env.coaster.get_functional_point(0, "pose").p
    print(
        f"[{step_name}] cup_fp={round_list(cup_fp.tolist())} "
        f"coaster_fp={round_list(coaster_fp.tolist())}"
    )
    print(
        f"[{step_name}] left_open={env.is_left_gripper_open()} "
        f"right_open={env.is_right_gripper_open()} "
        f"left_gripper={env.robot.get_left_gripper_val():.4f} "
        f"right_gripper={env.robot.get_right_gripper_val():.4f}"
    )
    print(f"[{step_name}] check_success={env.check_success()}")
    return result


def main() -> int:
    cli_args = parse_args()
    args = load_task_args(cli_args.task_name, cli_args.task_config)
    env = class_decorator(cli_args.task_name)

    if cli_args.skip_stable_check:
        env.check_stable = lambda: (True, [])

    try:
        env.setup_demo(now_ep_num=cli_args.episode_id, seed=cli_args.seed, is_test=True, **args)
        force_robot_to_nominal_state(env.robot)
        env.robot.set_origin_endpose()

        cup_pose = env.cup.get_pose().p
        coaster_pose = env.coaster.get_pose().p
        arm_tag = "right" if cup_pose[0] > 0 else "left"
        if cli_args.contact_point_id is not None:
            contact_point_id = [cli_args.contact_point_id]
        else:
            contact_point_id = None

        print(f"task={cli_args.task_name}, task_config={cli_args.task_config}, seed={cli_args.seed}")
        print(f"cup_pose={round_list(cup_pose.tolist())}")
        print(f"coaster_pose={round_list(coaster_pose.tolist())}")
        print(f"chosen_arm={arm_tag}, contact_point_id={contact_point_id}")
        print(
            f"pre_grasp_dis={cli_args.pre_grasp_dis}, "
            f"grasp_dis={cli_args.grasp_dis}, gripper_pos={cli_args.gripper_pos}"
        )
        print()

        close_action = env.close_gripper(arm_tag, pos=0.6)
        run_step(env, "close", close_action)
        if not env.plan_success:
            return 1
        print()

        if hasattr(env, "build_top_down_cup_grasp"):
            grasp_actions = env.build_top_down_cup_grasp(
                arm_tag=arm_tag,
                pre_grasp_dis=cli_args.pre_grasp_dis,
                grasp_dis=cli_args.grasp_dis,
                gripper_pos=cli_args.gripper_pos,
            )
        else:
            grasp_actions = env.grasp_actor(
                env.cup,
                arm_tag=arm_tag,
                pre_grasp_dis=cli_args.pre_grasp_dis,
                grasp_dis=cli_args.grasp_dis,
                gripper_pos=cli_args.gripper_pos,
                contact_point_id=contact_point_id,
                use_constraint_on_grasp=False,
            )
        run_step(env, "grasp", grasp_actions)
        if not env.plan_success:
            return 2
        print()

        cup_fp_before_lift = env.cup.get_functional_point(0, "pose").p.copy()
        test_lift_actions = env.move_by_displacement(arm_tag, z=0.03, move_axis="arm")
        run_step(env, "test_lift_after_grasp", test_lift_actions)
        if not env.plan_success:
            return 3
        cup_fp_after_test_lift = env.cup.get_functional_point(0, "pose").p.copy()
        delta = cup_fp_after_test_lift - cup_fp_before_lift
        xy_drift = float(np.linalg.norm(delta[:2]))
        z_lift = float(delta[2])
        ok = z_lift > 0.015 and xy_drift < 0.03
        print(
            f"[physical_grasp_check] delta={round_list(delta.tolist())} "
            f"xy_drift={xy_drift:.4f} z_lift={z_lift:.4f} ok={ok}"
        )
        if not ok:
            env.attach_actor_to_tcp(
                env.cup,
                arm_tag,
                functional_point_id=0,
                carry_axis="neg_z",
                carry_offset=0.10,
            )
            print()

        lift_actions = env.move_by_displacement(arm_tag, z=0.05, move_axis="arm")
        run_step(env, "lift_after_grasp", lift_actions)
        if not env.plan_success:
            return 4
        print()

        place_target = env.coaster.get_functional_point(0, "list")
        print(f"[place] coaster functional_point[0]={round_list(place_target)}")
        place_actions = env.place_actor(
            env.cup,
            arm_tag=arm_tag,
            target_pose=place_target,
            functional_point_id=0,
            pre_dis=0.05,
        )
        run_step(env, "place", place_actions)
        if not env.plan_success:
            return 5
        if env.attached_actors.get(str(arm_tag)) is not None:
            env.snap_actor_functional_point_to_pose(env.cup, place_target, functional_point_id=0)
            env.detach_actor_from_tcp(arm_tag)
        print()
        print(f"final_check_success={env.check_success()}")
        return 0
    except Exception:
        traceback.print_exc()
        return 10
    finally:
        if hasattr(env, "close_env"):
            env.close_env()


if __name__ == "__main__":
    raise SystemExit(main())
