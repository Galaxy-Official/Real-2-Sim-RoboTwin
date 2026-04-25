# Replay_Policy Auto Init

This folder contains the local files that should be uploaded into the RoboTwin
server's `policy/Replay_Policy/` directory.

## What This Adds

1. A LeRobot parquet replay loader for `data/block_stack`.
2. A first-frame auto-init pipeline that:
   - extracts the wrist-camera first frame
   - runs Depth Anything 3 externally
   - runs FoundationPose externally
   - combines the result with the fixed ALOHA wrist-camera-to-eef transform
   - writes `init_meta.json`
3. A new RoboTwin task `envs/replay_block_auto_init.py` that:
   - loads `init_meta.json`
   - places the target block actor into the first-frame pose
   - moves the active arm to the first-frame end-effector pose
   - then lets `Replay_Policy` continue replay from that state
4. Two uploadable wrappers under `policy/Replay_Policy/auto_init/`:
   - `run_depth_anything_metric.py`: runs Depth Anything 3 on one RGB frame and saves a `.npy` depth map
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
   - camera calibration path and `calibration_image_size`
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
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
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

## Camera Calibration

The first-frame pipeline now supports the fisheye calibration file
`policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz`.

The current default bundled values are:

```text
K_new =
[[290.58525666,   0.00000000, 320.00000000],
 [  0.00000000, 387.44700888, 240.00000000],
 [  0.00000000,   0.00000000,   1.00000000]]

D_raw = [-0.10000000, 0.05000000, -0.01000000, 0.00100000]
rms = 0.5
```

The current runtime path treats `K_new` as the pinhole intrinsics for the raw
640x480 wrist frame. `build_init_meta.py` therefore uses the extracted raw
frame directly and writes `episode_xxxxxx_intrinsics.json` from `K_new` in the
calibration `.npz`.

The fisheye `D_raw` values are kept only as calibration provenance. They are not
used by the default replay pipeline because step-2 debugging showed that
runtime fisheye undistortion over-warps the image.

The old fisheye runtime path used this calibration to:

- undistort the extracted wrist RGB frame
- undistort the object mask with nearest-neighbor interpolation
- compute the new pinhole camera matrix `new_K`
- write `episode_xxxxxx_intrinsics.json` for FoundationPose

LeRobot metadata reports the wrist video as `640x480`. The current default
`K_new` has principal point `(320, 240)`, which is geometrically consistent
with that frame size. If a future calibration is produced at a different source
resolution, set:

```yaml
auto_init:
  camera_calibration:
    calibration_image_size: [SOURCE_WIDTH, SOURCE_HEIGHT]
```

Without the correct source resolution, `K` cannot be scaled reliably and
FoundationPose translation can be biased.

### Updating Camera Intrinsics After Recalibration

When the wrist camera is recalibrated, update the auto-init camera config in
`policy/Replay_Policy/deploy_policy.yml`.

Recommended path:

1. Export the new calibration as an `.npz` file with these keys:
   - `K_new`: `3x3` camera matrix
   - `D_raw`: fisheye distortion coefficients, usually `4x1` or `4`
   - `rms`: optional calibration error
2. Put the file here:
   - `policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz`
3. Check the video resolution:
   - current wrist videos are `640x480`
4. If `K_new` is already the pinhole matrix for the replay wrist frame, keep:

```yaml
auto_init:
  camera_calibration:
    type: pinhole
    path: policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz
  undistort:
    enabled: false
    mask: false
```

5. If a future calibration provides raw fisheye intrinsics and distortion,
   re-enable runtime undistortion and set the original calibration resolution
   explicitly when needed:

```yaml
auto_init:
  camera_calibration:
    type: fisheye
    path: policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz
    calibration_image_size: [SOURCE_WIDTH, SOURCE_HEIGHT]
  undistort:
    enabled: true
    mask: true
```

For example, if calibration was done at `1280x960` but replay videos are
`640x480`:

```yaml
auto_init:
  camera_calibration:
    type: fisheye
    calibration_image_size: [1280, 960]
```

The code will scale `K` from the calibration resolution to the actual extracted
video frame resolution before computing the undistorted pinhole `new_K`.

### Debugging Calibration Semantics

Step-2 debugging has selected the raw + `K_new` pinhole path for the current
dataset. These scripts remain useful when comparing or revisiting the
alternative raw-fisheye interpretation.

From `policy/Replay_Policy`:

```bash
python auto_init/debug_calibration_semantics.py \
  --config deploy_policy.yml \
  --data-dir ../../replay_data/block_stack \
  --episode-index 0
```

The script writes diagnostics to:

```text
init_meta/cache/step2_calibration_debug/
```

Inspect these files first:

- `episode_xxxxxx_raw_vs_fisheye_assumption.png`
- `episode_xxxxxx_undistort_displacement_heatmap.png`
- `episode_xxxxxx_calibration_semantics.json`

For the current calibration, the fisheye runtime output looked like an enlarged
version of the raw frame and had severe border loss, so the default config keeps
`undistort.enabled: false` and uses `K_new` directly.

To generate the alternate pinhole-output assumption inputs directly, run:

```bash
python auto_init/debug_calibration_pinhole_assumption.py \
  --config deploy_policy.yml \
  --data-dir ../../replay_data/block_stack \
  --episode-index 0
```

This writes:

```text
init_meta/cache/step2_pinhole_assumption_debug/
```

If `debug_calibration_semantics.py` has already been run, the alternate script
also writes `episode_xxxxxx_pinhole_vs_fisheye_assumption.png`, which places
the raw+pinhole assumption next to the runtime-undistorted fisheye assumption.

The current default already uses the raw + `K_new` pinhole logic:

```yaml
auto_init:
  camera_calibration:
    type: pinhole
    path: policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz
  undistort:
    enabled: false
    mask: false
```

In this mode, the pipeline passes the extracted RGB frame directly to Depth
Anything 3 and FoundationPose, and `episode_xxxxxx_intrinsics.json` is
written from `K_new` in the calibration `.npz`.

## Wrapper Roles

```
run_depth_anything_metric.py
```

- Purpose: convert one wrist-camera RGB frame into a depth map that the replay
  pipeline can hand to FoundationPose.
- Input: `--repo-root`, `--image`, `--output`
- Output: one `.npy` depth file
- Notes:
  - defaults to metric mode because FoundationPose needs meaningful translation scale
  - supports explicit `--checkpoint`
  - can optionally write a small metadata JSON with `--meta-output`

```
run_foundationpose_once.py
```

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
      - depth-anything-3
      - python
      - policy/Replay_Policy/auto_init/run_depth_anything_metric.py
      - --repo-root
      - third_party/Depth-Anything-3
      - --image
      - "{image_path}"
      - --output
      - "{output_path}"
      - --intrinsics
      - "{intrinsics_path}"
      - --model-dir
      - depth-anything/da3metric-large
      - --metric-scale
      - da3metric
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
pip install numpy scipy pyyaml pyarrow pillow opencv-python
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
