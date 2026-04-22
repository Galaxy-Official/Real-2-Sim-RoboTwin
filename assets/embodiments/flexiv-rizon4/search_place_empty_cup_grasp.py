#!/usr/bin/env python3
"""Brute-force a few grasp parameters for Flexiv place_empty_cup.

Runs a reduced version of the expert sequence:
close -> grasp -> lift
and reports whether the cup is actually lifted.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from debug_place_empty_cup_grasp import (
    class_decorator,
    force_robot_to_nominal_state,
    load_task_args,
    round_list,
)


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", default="place_empty_cup")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--contact-points", default="0,1,2,3")
    parser.add_argument("--grasp-dis", default="-0.04,-0.02,0.0,0.02,0.04")
    parser.add_argument("--pre-grasp-dis", default="0.10")
    parser.add_argument("--gripper-pos", default="0.0,0.05,0.1,0.15")
    parser.add_argument("--lift-z", type=float, default=0.08)
    parser.add_argument("--max-grasp-xy-drift", type=float, default=0.05)
    parser.add_argument("--max-lift-xy-drift", type=float, default=0.10)
    parser.add_argument("--max-grasp-z-drift", type=float, default=0.05)
    parser.add_argument("--stop-on-stable", action="store_true")
    return parser.parse_args()


def main() -> int:
    cli_args = parse_args()
    args = load_task_args(cli_args.task_name, cli_args.task_config)

    contact_points = parse_csv_ints(cli_args.contact_points)
    grasp_dis_list = parse_csv_floats(cli_args.grasp_dis)
    pre_grasp_dis_list = parse_csv_floats(cli_args.pre_grasp_dis)
    gripper_pos_list = parse_csv_floats(cli_args.gripper_pos)

    best = None
    results = []

    for contact_point_id, pre_grasp_dis, grasp_dis, gripper_pos in itertools.product(
        contact_points, pre_grasp_dis_list, grasp_dis_list, gripper_pos_list
    ):
        env = class_decorator(cli_args.task_name)
        env.check_stable = lambda: (True, [])
        try:
            env.setup_demo(now_ep_num=cli_args.episode_id, seed=cli_args.seed, is_test=True, **args)
            force_robot_to_nominal_state(env.robot)
            env.robot.set_origin_endpose()

            cup_fp0 = np.array(env.cup.get_functional_point(0, "pose").p, dtype=np.float64)
            cup_pose0 = np.array(env.cup.get_pose().p, dtype=np.float64)
            arm_tag = "right" if cup_pose0[0] > 0 else "left"

            env.move(env.close_gripper(arm_tag, pos=0.6))
            if not env.plan_success:
                status = "close_fail"
            else:
                env.move(
                    env.grasp_actor(
                        env.cup,
                        arm_tag=arm_tag,
                        pre_grasp_dis=pre_grasp_dis,
                        grasp_dis=grasp_dis,
                        gripper_pos=gripper_pos,
                        contact_point_id=[contact_point_id],
                    )
                )
                if not env.plan_success:
                    status = "grasp_fail"
                else:
                    cup_fp_grasp = np.array(env.cup.get_functional_point(0, "pose").p, dtype=np.float64)
                    env.move(env.move_by_displacement(arm_tag, z=cli_args.lift_z, move_axis="arm"))
                    cup_fp_lift = np.array(env.cup.get_functional_point(0, "pose").p, dtype=np.float64)
                    if not env.plan_success:
                        status = "lift_fail"
                    else:
                        lift_delta = cup_fp_lift - cup_fp0
                        grasp_delta = cup_fp_grasp - cup_fp0
                        grasp_xy = float(np.linalg.norm(grasp_delta[:2]))
                        lift_xy = float(np.linalg.norm(lift_delta[:2]))
                        grasp_z = float(grasp_delta[2])
                        lift_z = float(lift_delta[2])

                        stable_lift = (
                            lift_z > 0.03
                            and grasp_xy < cli_args.max_grasp_xy_drift
                            and lift_xy < cli_args.max_lift_xy_drift
                            and abs(grasp_z) < cli_args.max_grasp_z_drift
                        )
                        if stable_lift:
                            status = "stable_lift"
                        elif lift_z > 0.03:
                            status = "flung_up"
                        else:
                            status = "not_lifted"

                        score = float(
                            lift_z
                            - 0.5 * lift_xy
                            - 0.25 * grasp_xy
                            - 0.5 * abs(grasp_z)
                        )
                        results.append(
                            {
                                "contact_point_id": contact_point_id,
                                "pre_grasp_dis": pre_grasp_dis,
                                "grasp_dis": grasp_dis,
                                "gripper_pos": gripper_pos,
                                "status": status,
                                "grasp_delta": grasp_delta,
                                "lift_delta": lift_delta,
                                "grasp_xy": grasp_xy,
                                "lift_xy": lift_xy,
                                "score": score,
                            }
                        )
                        if best is None or score > best["score"]:
                            best = results[-1]
                        print(
                            f"contact={contact_point_id} pre={pre_grasp_dis:.3f} grasp={grasp_dis:.3f} "
                            f"gripper={gripper_pos:.3f} status={status} "
                            f"grasp_delta={round_list(grasp_delta.tolist())} "
                            f"lift_delta={round_list(lift_delta.tolist())} "
                            f"grasp_xy={grasp_xy:.4f} lift_xy={lift_xy:.4f}"
                        )
                        if stable_lift and cli_args.stop_on_stable:
                            print()
                            print("=== BEST ===")
                            print(best)
                            return 0
                        continue

            print(
                f"contact={contact_point_id} pre={pre_grasp_dis:.3f} grasp={grasp_dis:.3f} "
                f"gripper={gripper_pos:.3f} status={status}"
            )
        finally:
            if hasattr(env, "close_env"):
                env.close_env()

    print()
    print("=== BEST ===")
    if best is None:
        print("No successful grasp/lift candidate found.")
    else:
        print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
