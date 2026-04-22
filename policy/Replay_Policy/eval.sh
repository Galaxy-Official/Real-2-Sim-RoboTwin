#!/bin/bash

set -euo pipefail

policy_name=Replay_Policy
task_name=${1}
task_config=${2}
data_dir=${3}
episode_index=${4:-0}
replay_arm=${5:-right}
gpu_id=${6:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}

echo -e "\033[33m========================================\033[0m"
echo -e "\033[33m  RoboTwin Replay With Auto Init\033[0m"
echo -e "\033[33m  Task:    ${task_name} (${task_config})\033[0m"
echo -e "\033[33m  Data:    ${data_dir}\033[0m"
echo -e "\033[33m  Episode: ${episode_index}\033[0m"
echo -e "\033[33m  Arm:     ${replay_arm}\033[0m"
echo -e "\033[33m  GPU:     ${gpu_id}\033[0m"
echo -e "\033[33m========================================\033[0m"

cd ../..

init_meta_dir="policy/${policy_name}/init_meta"
mkdir -p "${init_meta_dir}"
init_meta_path="${init_meta_dir}/episode_$(printf "%06d" "${episode_index}").json"

python -u "policy/${policy_name}/auto_init/build_init_meta.py" \
  --config "policy/${policy_name}/deploy_policy.yml" \
  --data-dir "${data_dir}" \
  --episode-index "${episode_index}" \
  --replay-arm "${replay_arm}" \
  --output "${init_meta_path}"

export REPLAY_INIT_META_PATH="${init_meta_path}"

python -u script/eval_policy.py --config "policy/${policy_name}/deploy_policy.yml" \
  --overrides \
  --task_name "${task_name}" \
  --task_config "${task_config}" \
  --data_dir "${data_dir}" \
  --episode_index "${episode_index}" \
  --replay_arm "${replay_arm}" \
  --policy_name "${policy_name}" \
  --seed 0 \
  --ckpt_setting replay \
  --expert_check False \
  --test_num 1
