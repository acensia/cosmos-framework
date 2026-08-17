#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Structured-TOML launch for action_policy_behavior1k_edge — Cosmos3-Edge
# BEHAVIOR-1K R1Pro action-policy SFT. Drives cosmos_framework.scripts.train
# against examples/toml/sft_config/action_policy_behavior1k_edge.toml.
#
# Point BEHAVIOR1K_ROOT at a local LeRobot v3 dir (a per-task RGB subset of
# behavior-1k/2026-challenge-demos works). The default recipe is FSDP 4x1
# (one 4-GPU node, NPROC_PER_NODE=4, global batch 512).
#
# Required env vars:
#   BEHAVIOR1K_ROOT       local BEHAVIOR-1K LeRobot dataset dir (no default)
#   EDGE_HF_PATH          local nvidia/Cosmos3-Edge HF snapshot (processor source; no default)
# Optional env vars (defaults below; override to relocate data/checkpoints):
#   BASE_CHECKPOINT_PATH  default: examples/checkpoints/Cosmos3-Edge-DCP
#   WAN_VAE_PATH          default: examples/checkpoints/wan22_vae/Wan2.2_VAE.pth
#   OUTPUT_ROOT           default: outputs/train
#
# Usage (single 4-GPU node):
#   BEHAVIOR1K_ROOT=<dir> EDGE_HF_PATH=<dir> NPROC_PER_NODE=4 \
#     bash examples/launch_sft_action_policy_behavior1k_edge.sh

# Overridable: point at action_policy_behavior1k_edge_state.toml for the
# robot-state-conditioned variant (use_state=True initial action row).
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_behavior1k_edge.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-DCP}"

# Behavior1KLeRobotDataset reads ${oc.env:BEHAVIOR1K_ROOT} directly (a LOCAL
# LeRobot dir) and the model config reads ${oc.env:EDGE_HF_PATH}; export both so
# torchrun (launched in this shell) inherits them.
export BEHAVIOR1K_ROOT="${BEHAVIOR1K_ROOT:-}"
export EDGE_HF_PATH="${EDGE_HF_PATH:-}"

EXTRA_DATASET_CHECK='[[ -f "$BEHAVIOR1K_ROOT/meta/info.json" ]] || { echo "ERROR: BEHAVIOR1K_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$BEHAVIOR1K_ROOT'\'')." >&2; exit 1; }; [[ -f "$EDGE_HF_PATH/processor_config.json" ]] || { echo "ERROR: EDGE_HF_PATH must be a local Cosmos3-Edge HF snapshot containing processor_config.json (got: '\''$EDGE_HF_PATH'\'')." >&2; exit 1; }'

# Extra Hydra overrides from the environment: a space-separated string word-split into
# the TAIL_OVERRIDES array, e.g. EXTRA_TAIL_OVERRIDES="trainer.max_iter=5".
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
