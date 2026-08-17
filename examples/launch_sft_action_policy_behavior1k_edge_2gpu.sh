#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# 2-GPU launch for action_policy_behavior1k_edge (FSDP 2x1, grad_accum 8 —
# same global batch 512 as the 4-GPU recipe). See
# launch_sft_action_policy_behavior1k_edge.sh for env var docs.
#
# Usage (single node, 2 GPUs):
#   BEHAVIOR1K_ROOT=<dir> EDGE_HF_PATH=<dir> \
#     bash examples/launch_sft_action_policy_behavior1k_edge_2gpu.sh

# Overridable: point at action_policy_behavior1k_edge_state_2gpu.toml for the
# robot-state-conditioned variant (use_state=True initial action row).
TOML_FILE="${TOML_FILE:-examples/toml/sft_config/action_policy_behavior1k_edge_2gpu.toml}"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge-DCP}"
: "${NPROC_PER_NODE:=2}"

export BEHAVIOR1K_ROOT="${BEHAVIOR1K_ROOT:-}"
export EDGE_HF_PATH="${EDGE_HF_PATH:-}"

EXTRA_DATASET_CHECK='[[ -f "$BEHAVIOR1K_ROOT/meta/info.json" ]] || { echo "ERROR: BEHAVIOR1K_ROOT must be a local LeRobot dir containing meta/info.json (got: '\''$BEHAVIOR1K_ROOT'\'')." >&2; exit 1; }; [[ -f "$EDGE_HF_PATH/processor_config.json" ]] || { echo "ERROR: EDGE_HF_PATH must be a local Cosmos3-Edge HF snapshot containing processor_config.json (got: '\''$EDGE_HF_PATH'\'')." >&2; exit 1; }'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
