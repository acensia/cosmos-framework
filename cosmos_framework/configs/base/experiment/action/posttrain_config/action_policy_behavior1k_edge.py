# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_behavior1k_edge`` — Cosmos3-Edge BEHAVIOR-1K action-policy SFT recipe.

Feeds ``Behavior1KLeRobotDataset`` (raw 23D absolute joint actions, embodiment
``behavior1k_lerobot`` / domain 22, 3-camera head+wrists composite on the
exact-aspect "2,3" 512x768 canvas) and fine-tunes the generation + action heads
from the public ``nvidia/Cosmos3-Edge`` base (converted to DCP). Point
``BEHAVIOR1K_ROOT`` at a local LeRobot v3 dir (a per-task subset of
``behavior-1k/2026-challenge-demos`` works). Mirrors
``action_policy_libero_nano`` with the Edge model config.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_behavior1k_sft_dataset
from cosmos_framework.data.generator.joint_dataloader import (
    PackingDataLoader,
    RankPartitionedDataLoader,
)
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_behavior1k_edge_model_config() -> dict:
    """BEHAVIOR-1K Edge model config: capped packed tokens, selective activation
    checkpointing, fresh diffusion-expert init, 10x vision flow-matching loss.
    ``encode_exact_durations=[17]`` matches ``chunk_length=16`` (+1 obs frame),
    like the DROID recipe pins ``[33]`` for ``chunk_length=32``."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)  # action_gen=True, max_action_dim=64, resolution="480"
    cfg["max_num_tokens_after_packing"] = 74000
    cfg["activation_checkpointing"]["mode"] = "selective"
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17]
    # Offline-safe processor: source the tokenizer/processor from the local Edge
    # snapshot (EDGE_HF_PATH) instead of a fresh nvidia/Cosmos3-Edge hub fetch.
    cfg["vlm_config"]["tokenizer"] = L(build_processor_lazy)(
        tokenizer_type="${oc.env:EDGE_HF_PATH}",
    )
    return cfg


action_policy_behavior1k_edge = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            # FusedAdam with fp32 master_weights + eps 1e-8 (bf16 params + eps 1e-6
            # diverged on the action loss in the LIBERO recipe).
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},  # linear LR decay
            {"override /checkpoint": "s3"},
            {
                "override /callbacks": [
                    "basic",
                    "optimization",
                    "job_monitor",
                ]
            },
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_behavior1k",
            group="action_sft",
            name="action_policy_behavior1k_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=_action_policy_behavior1k_edge_model_config(),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,  # popped by build_optimizer for FusedAdam (fused by construction)
            # Train the generation + action heads. k_norm_und_for_gen trains with
            # the gen pathway on Edge (use_und_k_norm_for_gen=True).
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "k_norm_und_for_gen",
            ],
            lr=5.0e-05,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],  # smoke: 100 iters (real run sets via TOML)
            f_max=[1.0],
            f_min=[0.0],
            f_start=[1.0e-06],
            verbosity_interval=0,
            warm_up_steps=[0],  # smoke (real run sets via TOML)
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,  # real run sets via TOML
            logging_iter=1,
            max_iter=100,  # smoke
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200, log_memory_detail=True, save_s3=False, step_size=1, upload_every_n_mul=5
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=100),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            # Skip net_ema (EMA warm-starts from net, see dcp.py) and the action
            # heads. The renewed public Edge base DOES ship trained action heads,
            # but they were trained for other domains' action layouts; the 23D
            # behavior1k joint layout shares no convention with them, so init the
            # heads fresh like the LIBERO recipe does.
            keys_to_skip_loading=[
                "net_ema.",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",  # Cosmos3-Edge DCP dir; supply via TOML/env
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,  # base init: tolerate key set differences
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_behavior1k",
            max_samples_per_batch=32,  # 512x768 canvas packs far fewer samples than LIBERO's 192x320
            max_sequence_length=None,  # None disables token packing (TOML can't express null)
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
                sampler=None,
                # Shuffling is handled by the dataset (iterable_shuffle=True below).
                datasets=dict(
                    behavior1k=dict(
                        ratio=1,
                        dataset=L(get_action_behavior1k_sft_dataset)(
                            # Local LeRobot v3 dir (per-task subset of
                            # behavior-1k/2026-challenge-demos is fine).
                            root="${oc.env:BEHAVIOR1K_ROOT}",
                            fps=30,  # metadata only (FPS-agnostic loader reads native fps from info.json)
                            chunk_length=16,
                            mode="wam",
                            action_space="joint_pos",
                            action_normalization=None,  # raw absolute joint targets (passthrough serving)
                            val_ratio=0.01,
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                            resolution="480",  # REQUIRED: "2,3" canvas exists only in the 480 tier
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            format_prompt_as_json=True,  # Edge checkpoints train with JSON-structured prompts
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


for _item in [action_policy_behavior1k_edge]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
