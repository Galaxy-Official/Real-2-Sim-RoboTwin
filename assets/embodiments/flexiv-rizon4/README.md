# Flexiv Rizon 4 For RoboTwin 2.0

This embodiment scaffold lets you add a single-arm `flexiv-rizon4` robot to RoboTwin 2.0 and combine two instances through RoboTwin's single-arm embodiment workflow.

Files in this folder:

- `build_rizon4_robotwin.py`: downloads `flexiv_description`, rewrites ROS package paths, rescales the official millimeter meshes into meter meshes for RoboTwin/MPLib, and generates both `rizon4_robotwin.urdf` and `rizon4_curobo.urdf`.
- `config.yml`: RoboTwin embodiment config.
- `curobo_tmp.yml`: CuRobo template config that RoboTwin will expand into `curobo.yml`.
- `collision_rizon4.yml`: coarse collision spheres for bootstrap planning.
- `rizon4.srdf`: minimal SRDF used by RoboTwin / MPLib.
- `_embodiment_config.snippet.yml`: snippet to append to RoboTwin's `task_config/_embodiment_config.yml`.

Important notes:

1. This package assumes the Flexiv GN01 gripper is loaded in simulation.
2. GN01 already exposes a fixed `closed_fingers_tcp` link, and RoboTwin's grasp generator already emits a TCP-centered target with a built-in `0.12 m` convention. This template therefore uses `closed_fingers_tcp` directly as the planning end link, with `gripper_bias: 0.12` and identity `delta_matrix`.
3. `global_trans_matrix` stays identity in the bootstrap config. If end-effector pose readback still looks rotated in visualization, calibrate it in a desktop RoboTwin environment, but avoid compensating GN01 with a non-identity `delta_matrix` unless you have direct evidence it is needed.

If `place_empty_cup` still fails with `target_pose cannot be None for move action`, run `debug_place_empty_cup_grasp.py` from the RoboTwin root after copying this embodiment directory to the server:

```bash
python assets/embodiments/flexiv-rizon4/debug_place_empty_cup_grasp.py \
  --task-config demo_clean_flexiv \
  --seed 0 \
  --candidate-limit 10
```

The script prints the exact grasp seed pose, all rotated candidate poses, their transformed `flange` targets in both world and robot-base frames, and the `curobo` / `mplib` status for each candidate. That output is much more actionable than the original `seed pose` log because it shows the real poses that RoboTwin is handing to the planners after `rotate_lim`, `gripper_bias`, and `delta_matrix` are applied.

If all task-derived grasp candidates still fail, run the direct IK probe from the RoboTwin root:

```bash
python assets/embodiments/flexiv-rizon4/probe_rizon4_base_ik.py \
  --task-config demo_clean_flexiv \
  --seed 0 \
  --arm right
```

This scans a simple xyz grid in the arm base frame with a fixed quaternion and reports whether `curobo` / `mplib` can solve any pose at all. It mirrors the "find valid pose" step in RoboTwin's official new-embodiment calibration flow and helps distinguish:

- task grasp pose generation issues
- `delta_matrix` / orientation issues
- deeper URDF / planner / base-placement problems

If `collect_data.sh place_empty_cup ...` still fails but setup succeeds, run the step-by-step expert debugger:

```bash
python assets/embodiments/flexiv-rizon4/debug_place_empty_cup_play_once.py \
  --task-config demo_clean_flexiv \
  --seed 0
```

It prints the exact `close -> grasp -> lift -> place -> retreat` action sequence used by `place_empty_cup`, along with the generated target poses and any planning failure status from `_base_task.py`.
4. `collision_rizon4.yml` is hand-authored and intentionally coarse. It is good for initial bring-up, but for best planning quality you should later replace it with Isaac Sim annotated spheres as recommended by cuRobo and RoboTwin.
5. `rizon4_curobo.urdf` intentionally removes articulated GN01 finger joints and replaces them with one fixed TCP link. This avoids a known cuRobo failure mode on robots that contain extra non-planning joints.
6. The official Flexiv mesh assets are authored in millimeters and referenced with `scale="0.001 0.001 0.001"`. RoboTwin's MPLib bridge expects unit-scale meshes, so the build script bakes the `0.001` factor into the mesh vertices and rewrites the URDF/Xacro scale attributes to `1 1 1`.
7. The build script also injects a fixed `camera` link into `rizon4_robotwin.urdf` so RoboTwin can attach wrist cameras without falling back to the robot base link.
8. In RoboTwin, the embodiment `planner` field controls the MPLib fallback mode, not whether cuRobo is enabled. Keep it as `mplib_RRT` unless you intentionally want MPLib screw planning.

Quick start after copying this folder into RoboTwin's `assets/embodiments/`:

```bash
cd /path/to/RoboTwin
python assets/embodiments/flexiv-rizon4/build_rizon4_robotwin.py
python script/update_embodiment_config_path.py
```

Then:

1. Append `_embodiment_config.snippet.yml` into `task_config/_embodiment_config.yml`.
2. Set your task config embodiment to `[flexiv-rizon4, flexiv-rizon4, 0.8]`.
3. Keep `render_freq: 0` on headless servers.
