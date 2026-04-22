# Replay_Policy Auto Init

This folder contains the local files that should be uploaded into the RoboTwin
server's `policy/Replay_Policy/` directory.

## What This Adds

1. A LeRobot parquet replay loader for `data/block_stack`.
2. A first-frame auto-init pipeline that:
   - extracts the wrist-camera first frame
   - runs Depth Anything V2 externally
   - runs FoundationPose externally
   - combines the result with the fixed ALOHA wrist-camera-to-eef transform
   - writes `init_meta.json`
3. A new RoboTwin task `envs/replay_block_auto_init.py` that:
   - loads `init_meta.json`
   - places the target block actor into the first-frame pose
   - moves the active arm to the first-frame end-effector pose
   - then lets `Replay_Policy` continue replay from that state
4. Two uploadable wrappers under `policy/Replay_Policy/auto_init/`:
   - `run_depth_anything_metric.py`: runs Depth Anything V2 on one RGB frame and saves a `.npy` depth map
   - `run_foundationpose_once.py`: runs FoundationPose registration on one `rgb + depth + mask + K + mesh` sample and saves pose JSON

## Files To Upload

- `policy/Replay_Policy/*`
- `envs/replay_block_auto_init.py`
- `task_config/replay_block_auto_init.yml`

## Required Manual Edits Before Running

1. Update `policy/Replay_Policy/object_configs/block_stack_default.yml`
   - `modelname`
   - `mesh_path`
2. Update `policy/Replay_Policy/deploy_policy.yml`
   - camera intrinsics
   - `depth_anything.command`
   - `foundationpose.command`
   - optional mask template path
   - verify the default ALOHA OpenCV `T_camera_to_eef` matches your setup
3. Update `task_config/replay_block_auto_init.yml`
   - `embodiment`
   - `target_object_modelname`
   - optional static scene objects

## Suggested Server Setup

Clone the external repos under `RoboTwin/third_party/`:

```bash
cd /path/to/RoboTwin
mkdir -p third_party
cd third_party
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
git clone https://github.com/NVlabs/FoundationPose.git
```

The bundled `deploy_policy.yml` already contains the ALOHA wrist-camera-to-eef
transform in the OpenCV convention expected by FoundationPose:

```text
T_camera_to_eef =
[[ 0.000001, -0.000796, -1.000000,  0.000000],
 [ 0.905971,  0.423340, -0.000337,  0.136103],
 [ 0.423341, -0.905970,  0.000722,  0.074117],
 [ 0.000000,  0.000000,  0.000000,  1.000000]]
```

The equivalent `translation` and Rodrigues `rotvec` are:

```text
translation = [0.000000, 0.136103, 0.074117]
rotvec      = [-0.880866, -1.384416, 0.881969]
```

## Wrapper Roles

`run_depth_anything_metric.py`

- Purpose: convert one wrist-camera RGB frame into a depth map that the replay
  pipeline can hand to FoundationPose.
- Input: `--repo-root`, `--image`, `--output`
- Output: one `.npy` depth file
- Notes:
  - defaults to metric mode because FoundationPose needs meaningful translation scale
  - supports explicit `--checkpoint`
  - can optionally write a small metadata JSON with `--meta-output`

`run_foundationpose_once.py`

- Purpose: run the first-frame registration step from FoundationPose and export
  `T_cam_obj` in a JSON format that `build_init_meta.py` already understands.
- Input: `--repo-root`, `--rgb`, `--depth`, `--mask`, `--intrinsics`, `--mesh`, `--output`
- Output: one JSON file containing `matrix`, `T_cam_obj`, `pose6d_rotvec`, and `pose7d_wxyz`
- Notes:
  - this is a single-frame registration wrapper, not a tracker loop
  - depth/mask/RGB must share the same image resolution
  - the FoundationPose environment must already be built successfully

## Suggested Commands In `deploy_policy.yml`

```yaml
auto_init:
  depth_anything:
    mode: command
    output_template: "{cache_dir}/episode_{episode_index:06d}_depth.npy"
    command:
      - conda
      - run
      - -n
      - depth-anything
      - python
      - policy/Replay_Policy/auto_init/run_depth_anything_metric.py
      - --repo-root
      - third_party/Depth-Anything-V2
      - --image
      - "{image_path}"
      - --output
      - "{output_path}"
      - --encoder
      - vitl
      - --mode
      - metric
      - --metric-dataset
      - hypersim
      - --checkpoint
      - third_party/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
  foundationpose:
    mode: command
    output_template: "{cache_dir}/episode_{episode_index:06d}_foundationpose.json"
    command:
      - conda
      - run
      - -n
      - foundationpose
      - python
      - policy/Replay_Policy/auto_init/run_foundationpose_once.py
      - --repo-root
      - third_party/FoundationPose
      - --rgb
      - "{image_path}"
      - --depth
      - "{depth_path}"
      - --mask
      - "{mask_path}"
      - --intrinsics
      - "{intrinsics_path}"
      - --mesh
      - "{mesh_path}"
      - --output
      - "{output_path}"
      - --debug-dir
      - "{cache_dir}/foundationpose_debug_{episode_index:06d}"
```

Create Python environments or install into the one used by RoboTwin. At minimum
the replay/auto-init scripts expect:

```bash
pip install numpy scipy pyyaml pyarrow
```

You will also need the dependencies required by the two cloned repos, following
their official READMEs.

## Expected Runtime Flow

```bash
cd /path/to/RoboTwin/policy/Replay_Policy
bash eval.sh replay_block_auto_init replay_block_auto_init /path/to/block_stack 0 right 0
```

`eval.sh` first builds `policy/Replay_Policy/init_meta/episode_000000.json`,
then launches `script/eval_policy.py`.
