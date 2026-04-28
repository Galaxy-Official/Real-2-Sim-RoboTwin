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

The current runtime path treats `K_new + D_raw` as fisheye calibration for the
raw 640x480 wrist frame. `build_init_meta.py` undistorts the extracted wrist
RGB frame and mask before running Depth Anything 3 and FoundationPose. This
change was made because metric depth on the raw fisheye frame was observed to
be much larger than the real wrist-camera-to-object distance.

The current fisheye runtime path uses this calibration to:

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
4. The current default treats `K_new + D_raw` as fisheye calibration for the raw
   replay wrist frame and undistorts RGB/mask before Depth Anything 3 and
   FoundationPose:

```yaml
auto_init:
  camera_calibration:
    type: fisheye
    path: policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz
  undistort:
    enabled: true
    mask: true
```

5. If `K_new` is later confirmed to already be the pinhole matrix for the replay
   wrist frame, disable runtime undistortion:

```yaml
auto_init:
  camera_calibration:
    type: pinhole
    path: policy/Replay_Policy/auto_init/fisheye_calib_result_resized.npz
  undistort:
    enabled: false
    mask: false
```

6. If a future calibration provides raw fisheye intrinsics at a different
   resolution, set the original calibration resolution explicitly when needed:

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

Step-2 originally compared raw + `K_new` pinhole against runtime fisheye
undistortion. Depth Anything 3 scale checks later showed that running DA3 on
the raw fisheye frame can produce depth that is far too large, so the current
default uses runtime fisheye undistortion.

From `policy/Replay_Policy`:

```bash
python auto_init/debug_calibration_semantics.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
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
  --data-dir data/handcap2603/block_stack_0401 \
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

### Debugging Camera Calibration Outputs

After selecting the calibration semantics, verify the concrete outputs produced
by `camera_calibration.py`:

```bash
python auto_init/debug_camera_calibration_outputs.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0
```

If the configured mask path is not present yet, either provide it explicitly:

```bash
python auto_init/debug_camera_calibration_outputs.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --mask-image /path/to/episode_000000_mask.png
```

or skip mask checks while validating the RGB frame and intrinsics:

```bash
python auto_init/debug_camera_calibration_outputs.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --allow-missing-mask
```

The script writes:

```text
init_meta/cache/step3_camera_calibration_debug/
```

For the current fisheye-undistort logic, expected results are:

- `episode_xxxxxx_current_config_raw_vs_runtime_frame.png` should show the undistorted runtime frame
- `episode_xxxxxx_current_config_raw_vs_runtime_mask.png` should show the undistorted runtime mask, if a mask is available
- `episode_xxxxxx_current_config_intrinsics.json` should have `source: fisheye_undistorted`
- `episode_xxxxxx_camera_calibration_outputs.json` should report `frame_undistorted: true`

To additionally force a fisheye reference pass while testing a non-fisheye
configuration, add:

```bash
python auto_init/debug_camera_calibration_outputs.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --include-fisheye-reference
```

## Generating Object Masks

The mask is a binary image used by FoundationPose to isolate the target object
in the first wrist-camera frame. The default path is controlled by
`auto_init.mask.template` in `deploy_policy.yml`:

```text
{data_dir}/masks/episode_{episode_index:06d}.png
```

For the current server layout, run from `policy/Replay_Policy` with
`--data-dir data/handcap2603/block_stack_0401`; episode `0` mask is:

```text
data/handcap2603/block_stack_0401/masks/episode_000000.png
```

### Installing SAM2 On The Server

Install SAM2 under the RoboTwin root so all third-party code stays in one
place:

```bash
cd /path/to/RoboTwin
mkdir -p third_party
cd third_party
git clone https://github.com/facebookresearch/sam2.git
cd sam2
```

Create a dedicated conda environment. Pick the PyTorch CUDA wheel that matches
your server driver; the example below uses CUDA 12.1 wheels:

```bash
conda create -n sam2 python=3.10 -y
conda activate sam2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[notebooks]"
pip install pyyaml pillow numpy opencv-python
```

Confirm `pip` and `python` are from the same `sam2` environment:

```bash
which python
which pip
python -m pip show SAM-2
python - <<'PY'
import sam2
print("sam2 package path:", sam2.__file__)
PY
```

If the SAM2 CUDA extension build fails but the package installs, continue; image
mask generation still usually works. Then download checkpoints:

```bash
cd /path/to/RoboTwin/third_party/sam2/checkpoints
bash download_ckpts.sh
```

Verify the environment:

```bash
conda activate sam2
python - <<'PY'
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sam2 import ok", SAM2ImagePredictor)
PY
```

The commands below assume:

```text
SAM2 repo:   /path/to/RoboTwin/third_party/sam2
Checkpoint:  /path/to/RoboTwin/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
Config:      configs/sam2.1/sam2.1_hiera_l.yaml
```

Use `generate_sam_mask.py` to create this mask with SAM/SAM2. First create a
prompt template and inspect the extracted first frame:

```bash
cd /path/to/RoboTwin/policy/Replay_Policy
conda activate sam2

python auto_init/generate_sam_mask.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --write-prompt-template init_meta/cache/mask_generation_debug/episode_000000_prompt.json
```

This also writes a coordinate guide image:

```text
init_meta/cache/mask_generation_debug/episode_000000_coordinate_guide.png
```

Use this guide instead of guessing raw pixel coordinates. Yellow vertical lines
show `x`; cyan horizontal lines show `y`.

Edit the JSON with:

- `box`: tight `[x1, y1, x2, y2]` rectangle around the target object
- `positive_points`: one or more points safely inside the target object
- `negative_points`: points on distractors such as gripper fingers, table, or other blocks

Coordinate convention:

- origin `(0, 0)` is the top-left corner of the image
- `x` increases to the right
- `y` increases downward
- wrist images are currently `640x480`, so valid `x` is `[0, 639]` and valid `y` is `[0, 479]`

Example prompt JSON:

```json
{
  "image_path": "init_meta/cache/mask_generation_debug/episode_000000_wrist_first_frame.png",
  "coordinate_guide_path": "init_meta/cache/mask_generation_debug/episode_000000_coordinate_guide.png",
  "box": [260, 150, 360, 250],
  "positive_points": [[310, 200]],
  "negative_points": [[310, 125], [390, 205]]
}
```

How to choose values:

- `box`: read the approximate left, top, right, bottom edges of the target object from the coordinate guide
- `positive_points`: choose one point near the visible center of the target object
- `negative_points`: choose points inside nearby distractors if SAM includes them in the mask
- The numbers do not need to be pixel-perfect; a tight bbox plus one good positive point is usually enough

Edit it on the server:

```bash
nano init_meta/cache/mask_generation_debug/episode_000000_prompt.json
```

If you prefer not to edit JSON, pass the same prompts directly on the command
line:

```bash
--box 260 150 360 250 \
--positive-point 310 200 \
--negative-point 310 125 \
--negative-point 390 205
```

Then run SAM2:

```bash
cd /path/to/RoboTwin/policy/Replay_Policy
conda activate sam2

python auto_init/generate_sam_mask.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --prompt-json init_meta/cache/mask_generation_debug/episode_000000_prompt.json \
  --backend sam2 \
  --sam2-repo ../../third_party/sam2 \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --checkpoint ../../third_party/sam2/checkpoints/sam2.1_hiera_large.pt \
  --device cuda
```

Or run SAM v1 if your environment uses `segment_anything`:

```bash
python auto_init/generate_sam_mask.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --prompt-json init_meta/cache/mask_generation_debug/episode_000000_prompt.json \
  --backend sam \
  --model-type vit_h \
  --checkpoint /path/to/sam_vit_h.pth \
  --device cuda
```

The script writes the binary mask to the default mask path and debug files to:

```text
init_meta/cache/mask_generation_debug/
```

Inspect `episode_xxxxxx_sam_mask_overlay.png`. The green region should include
the complete target object and exclude the gripper, table, and neighboring
objects. If it is wrong, tighten the bbox or add negative points and rerun.

After the overlay looks correct, verify that the mask was saved to the default
path:

```bash
ls -lh data/handcap2603/block_stack_0401/masks/episode_000000.png
```

Then rerun step 3 without `--allow-missing-mask`:

```bash
python auto_init/debug_camera_calibration_outputs.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0
```

## Debugging Depth Anything 3

Step 4 verifies the single-frame RGB + K input to Depth Anything 3 and the
resulting metric depth `.npy` file before FoundationPose is introduced.

If Depth Anything 3 is not installed on the server yet, install it under the
RoboTwin root:

```bash
cd /path/to/RoboTwin
mkdir -p third_party
cd third_party
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git

conda create -n depth-anything-3 python=3.10 -y
conda activate depth-anything-3
cd /path/to/RoboTwin/third_party/Depth-Anything-3
python -m pip install -e .
python -m pip install pillow numpy pyyaml opencv-python
```

Verify the import:

```bash
conda activate depth-anything-3
python - <<'PY'
import torch
from depth_anything_3.api import DepthAnything3
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("DepthAnything3 import ok", DepthAnything3)
PY
```

From `policy/Replay_Policy`, first verify input preparation only:

```bash
cd /path/to/RoboTwin/policy/Replay_Policy

python auto_init/debug_depth_anything.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --prepare-only
```

Then run Depth Anything 3 through the command configured in
`deploy_policy.yml`:

```bash
python auto_init/debug_depth_anything.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0
```

The script writes:

```text
init_meta/cache/step4_depth_anything_debug/
```

Inspect:

- `episode_xxxxxx_depth.npy`: metric depth array for FoundationPose
- `episode_xxxxxx_depth_vis.png`: normalized depth visualization
- `episode_xxxxxx_depth_anything_debug.json`: shape, finite ratio, positive ratio, and mask-region depth stats

Expected checks:

- `first_frame_undistortion_matches_config: true`
- `intrinsics_source_matches_config: true`
- `depth_shape_matches_image: true`
- `depth_global_finite_ratio_ok: true`
- `depth_global_positive_ratio_ok: true`
- if a mask exists, `depth_mask_finite_ratio_ok: true` and `depth_mask_positive_ratio_ok: true`

If the server cannot access Hugging Face during the first run, download the
`depth-anything/da3metric-large` model manually and replace `--model-dir` in
`deploy_policy.yml` with that local model directory.

## Debugging FoundationPose

Step 5 verifies the exact single-frame inputs to FoundationPose, then runs
FoundationPose registration and checks the pose JSON.

FoundationPose is not installed by this repository. The official project
recommends Docker; its Conda setup is marked experimental. If you use Conda on
the server, install and verify it before running step 5.

From the RoboTwin root:

```bash
cd /inspire/hdd/project/robot-reasoning/xuyue-p-xuyue/lihong_workspace/lihong/RoboTwin
mkdir -p third_party
cd third_party

# Skip this if the directory already exists.
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
```

Create and enter the environment. The simplest reliable rule is: the PyTorch
CUDA version must match the CUDA toolkit reported by `nvcc -V`, because
`nvdiffrast` is compiled locally.

```bash
nvcc -V

conda create -n foundationpose python=3.9 -y
conda activate foundationpose
```

If `nvcc -V` reports CUDA 12.4, use this PyTorch build. Use PyTorch 2.4.1 here
instead of newer PyTorch releases because PyTorch3D's official install support
currently includes PyTorch 2.4.1:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

If your server explicitly provides CUDA 11.8 instead, use the official
FoundationPose baseline:

```bash
conda install pytorch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Before installing `nvdiffrast`, verify that this exact environment has PyTorch
and that `torch.version.cuda` matches `nvcc -V`:

```bash
python - <<'PY'
import sys
print(sys.executable)
try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
```

Then install build tools. Use an older setuptools line that still provides
`pkg_resources`, which some FoundationPose dependencies still import during
build:

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools<70" wheel ninja pybind11
```

Then install the remaining Python dependencies. Do not run the official
`requirements.txt` directly after installing a CUDA 12.4 PyTorch build, because
that file pins `torch/torchvision/torchaudio` to CUDA 11.8. Filter those lines
out. Also use `--no-build-isolation`, otherwise older packages such as `visdom`
may fail in pip's temporary build environment with `No module named
'pkg_resources'`:

```bash
grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==)' requirements.txt \
  > /tmp/foundationpose_requirements_no_torch.txt
python -m pip install --no-build-isolation -r /tmp/foundationpose_requirements_no_torch.txt
```

Install PyTorch3D from source against the active PyTorch/CUDA pair:

```bash
python -m pip install fvcore iopath
MAX_JOBS=8 FORCE_CUDA=1 python -m pip install --no-cache-dir --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If the server runs out of memory while compiling PyTorch3D, rerun the same
command with `MAX_JOBS=4`.

Install `nvdiffrast` only after PyTorch imports correctly. The
`--no-build-isolation` flag is required so the CUDA extension builds against the
PyTorch installed in the active Conda environment:

```bash
python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
```

If this command still fails, first check that the server has CUDA compiler
tools, not only CUDA runtime:

```bash
which nvcc
nvcc -V
```

Do not use `sudo` for the build commands; it can switch away from the active
Conda Python and make `torch` disappear during extension builds.

Verify `nvdiffrast`:

```bash
python - <<'PY'
import torch
import nvdiffrast.torch as dr
from pytorch3d.transforms import so3_log_map
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("nvdiffrast and pytorch3d import ok")
PY
```

Build FoundationPose extensions with the repository helper. This avoids two
common failures in the official `build_all_conda.sh`: missing Boost discovery
and pip editable builds that cannot see the active environment's `torch`.

```bash
cd /path/to/RoboTwin
conda activate foundationpose
bash policy/Replay_Policy/auto_init/setup_foundationpose_build.sh \
  --foundationpose-root third_party/FoundationPose \
  --max-jobs 8
```

Verify FoundationPose imports from `third_party/FoundationPose`:

```bash
python - <<'PY'
import nvdiffrast.torch as dr
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
print("FoundationPose import ok")
PY
```

FoundationPose also needs its pretrained weights under:

```text
third_party/FoundationPose/weights/2023-10-28-18-33-37
third_party/FoundationPose/weights/2024-01-11-20-02-45
```

Download them from the RoboTwin root:

```bash
conda activate foundationpose
bash policy/Replay_Policy/auto_init/setup_foundationpose_weights.sh \
  --foundationpose-root third_party/FoundationPose
```

The helper uses the Hugging Face mirror `gpue/foundationpose-weights` and
verifies that both `config.yml` and `model_best.pth` exist for the scorer and
refiner runs.

If the environment name is not `foundationpose`, update
`auto_init.foundationpose.command` in `deploy_policy.yml` and replace the
environment name after `conda run -n`.

The current block-stack data uses `121_orange-block`. The default object config
is:

```yaml
name: block_stack_target
modelname: 121_orange-block
mesh_path: assets/objects/121_orange-block/visual/base0.glb
symmetry: none
```

This assumes the official RoboTwin asset exists at
`RoboTwin/assets/objects/121_orange-block/visual/base0.glb`.

RoboTwin object assets are usually loaded with the per-object scale from
`model_data*.json`. The FoundationPose wrapper uses `--mesh-scale auto` by
default, which reads that scale and applies it to the mesh before registration.
For `121_orange-block`, this should apply:

```text
assets/objects/121_orange-block/model_data0.json
scale: [0.034, 0.034, 0.034]
```

From `policy/Replay_Policy`, first validate inputs only:

```bash
python auto_init/debug_foundationpose.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --prepare-only
```

If you want to test with a mesh without editing YAML yet, pass it explicitly:

```bash
python auto_init/debug_foundationpose.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --mesh /absolute/path/to/target_object_mesh.obj \
  --prepare-only
```

Then run FoundationPose:

```bash
python auto_init/debug_foundationpose.py \
  --config deploy_policy.yml \
  --data-dir data/handcap2603/block_stack_0401 \
  --episode-index 0 \
  --mesh /absolute/path/to/target_object_mesh.obj
```

The script uses the step-4 depth by default:

```text
init_meta/cache/step4_depth_anything_debug/episode_000000_depth.npy
```

It writes:

```text
init_meta/cache/step5_foundationpose_debug/
```

Inspect:

- `episode_xxxxxx_foundationpose_input_overlay.png`
- `episode_xxxxxx_foundationpose_pose_overlay.png`
- `episode_xxxxxx_foundationpose.json`
- `episode_xxxxxx_foundationpose_debug.json`

The pose overlay projects the scaled mesh bounding box and object-frame axes
back onto the wrist RGB image. Yellow is the projected mesh box, red/green/blue
are the object `x/y/z` axes, and green is the SAM mask.

Expected checks:

- `mesh_exists: true`
- `mesh_not_placeholder: true`
- `rgb_depth_shape_match: true`
- `depth_mask_shape_match: true`
- `mask_nonempty: true`
- `first_frame_undistortion_matches_config: true`
- `intrinsics_source_matches_config: true`
- after running FoundationPose, `pose_present: true`
- `pose_translation_positive_z: true`
- `pose_rotation_det_ok: true`

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
