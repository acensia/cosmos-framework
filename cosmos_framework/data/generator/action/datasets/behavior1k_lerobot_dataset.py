# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""BEHAVIOR-1K R1Pro LeRobot dataset (23D absolute joint actions).

Reads the ``behavior-1k/2026-challenge-demos`` LeRobot v3 dataset (or any
per-task subset of it with the same layout): parquet frames read directly,
windows by frame index, video decoded at each frame's real timestamp
(mirroring ``LIBEROLeRobotDataset``, so it is FPS-agnostic).

Action layout (``joint_pos``): the stored 23D ``action`` is used as-is —
``[base(3), trunk(4), left_arm(7), left_gripper(1), right_arm(7),
right_gripper(1)]`` (embodiment ``behavior1k_lerobot``, domain id 22). Actions
are raw absolute joint targets; ``action_normalization=None`` by default (like
the DROID ``joint_pos`` recipe), matching a passthrough serving contract.

Camera composite (``concat_view``): the R1Pro's head camera (zed, 720x720) on
top with the left/right wrist cameras (realsense, 480x480) downscaled to
360x360 and tiled side by side below -> 720(W) x 1080(H) per frame. At the
"480" resolution tier this snaps to the purpose-registered exact-aspect "2,3"
512x768 canvas with no reflection padding (see
``cosmos_framework/data/generator/utils.py``).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_spec import ActionSpec, Gripper, Joint, build_action_spec
from cosmos_framework.data.generator.action.datasets.base_dataset import ActionBaseDataset
from cosmos_framework.utils import log

_ACTION_FEATURE = "action"
_HEAD_FEATURE = "observation.rgb.zed_link_camera_0"
_LEFT_WRIST_FEATURE = "observation.rgb.left_realsense_link_camera_0"
_RIGHT_WRIST_FEATURE = "observation.rgb.right_realsense_link_camera_0"
_NORMALIZERS_DIR = Path(__file__).parent.parent / "normalizer_stats"

_ACTION_DIM = 23

# Must stay byte-identical to the serving-side prompt sentence.
CONCAT_VIEW_LAYOUT_DESCRIPTION = (
    "The top row is from the head-mounted camera. "
    "The bottom row contains the left and right wrist-mounted camera views, concatenated horizontally."
)


class Behavior1KLeRobotDataset(ActionBaseDataset):
    """BEHAVIOR-1K R1Pro action-policy dataset with raw 23D joint actions."""

    EMBODIMENT_TYPE = "behavior1k_lerobot"

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        mode: str = "wam",
        tolerance_s: float = 1e-4,
        action_space: str = "joint_pos",
        embodiment_type: str = "behavior1k_lerobot",
        action_normalization: str | None = None,
        split: str = "train",
        val_ratio: float = 0.01,
        seed: int = 0,
        sample_stride: int = 1,
    ) -> None:
        if action_space != "joint_pos":
            raise NotImplementedError(
                f"This BEHAVIOR-1K dataset only supports action_space='joint_pos', got {action_space!r}."
            )
        split = split.lower().strip()
        if split not in {"train", "val", "valid", "validation", "eval", "test", "full"}:
            raise ValueError(f"Unsupported split={split!r}. Use train/val/full.")
        if chunk_length % 4 != 0:
            raise ValueError(f"chunk_length must be divisible by 4, got {chunk_length}.")

        super().__init__(
            root=root,
            domain_name=embodiment_type,
            fps=fps,
            chunk_length=chunk_length,
            mode=mode,
            pose_convention="backward_framewise",  # unused for joint_pos; satisfies the base assert
            tolerance_s=tolerance_s,
            viewpoint="concat_view",
            action_normalization=action_normalization,
            sample_stride=sample_stride,
        )
        # FPS-agnostic loader: trust the dataset's NATIVE fps (30 for the 2026
        # challenge demos) for conditioning_fps / prompt duration. Frame sampling
        # uses each frame's real timestamp regardless.
        info_fps = self._info.get("fps")
        if info_fps:
            if int(info_fps) != int(fps):
                log.info(f"Using dataset native fps={info_fps} for conditioning (requested {fps}).")
            self._fps = float(info_fps)
            self._dt = 1.0 / self._fps

        self._video_keys = [_HEAD_FEATURE, _LEFT_WRIST_FEATURE, _RIGHT_WRIST_FEATURE]

        # Compact, lazy frame index (mirrors LIBEROLeRobotDataset): read only the
        # columns the sample builder needs into contiguous arrays, ordered by global
        # frame index, so DataLoader worker forks share them copy-on-write.
        index_parts, episode_parts, task_parts, ts_parts, action_parts = [], [], [], [], []
        for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path, columns=["index", "episode_index", "task_index", "timestamp", _ACTION_FEATURE])
            index_parts.append(table["index"].to_numpy())
            episode_parts.append(table["episode_index"].to_numpy())
            task_parts.append(table["task_index"].to_numpy())
            ts_parts.append(table["timestamp"].to_numpy())
            action_parts.append(np.asarray(table[_ACTION_FEATURE].to_pylist(), dtype=np.float32))
        if not index_parts:
            raise FileNotFoundError(f"No data parquet found under {self._root / 'data'}.")
        order = np.argsort(np.concatenate(index_parts).astype(np.int64), kind="stable")
        self._row_episode = np.concatenate(episode_parts).astype(np.int64)[order]
        self._row_task = np.concatenate(task_parts).astype(np.int64)[order]
        self._row_timestamp = np.concatenate(ts_parts).astype(np.float64)[order]
        self._row_action = np.concatenate(action_parts, axis=0).astype(np.float32)[order]
        if self._row_action.shape[-1] != _ACTION_DIM:
            raise ValueError(
                f"Expected {_ACTION_DIM}D actions for behavior1k_lerobot, got {self._row_action.shape[-1]}D."
            )

        assert np.all(np.diff(self._row_episode) >= 0), "episode_index not contiguous after sorting by frame index"
        ep_vals, ep_starts, ep_counts = np.unique(self._row_episode, return_index=True, return_counts=True)

        # Deterministic per-episode train/val split (seeded; same on every rank).
        keep = self._split_episode_ids(ep_vals.tolist(), split, val_ratio, seed)
        kept = np.array([int(v) in keep for v in ep_vals], dtype=bool)
        self._ep_vals = ep_vals.astype(np.int64)[kept]
        self._ep_starts = ep_starts.astype(np.int64)[kept]
        kept_counts = ep_counts.astype(np.int64)[kept]
        # Within-episode windows only: total - n_kept_episodes * chunk_length valid samples.
        self._valid_cum = np.cumsum(np.maximum(0, kept_counts - self._chunk_length)).astype(np.int64)

        log.info(
            f"Loaded BEHAVIOR-1K dataset root={self._root} split={split!r} fps={self._fps} "
            f"kept_episodes={len(self._ep_vals)}/{len(ep_vals)} "
            f"valid_indices={int(self._valid_cum[-1]) if self._valid_cum.size else 0}"
        )

    # ---- spec / dims -------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    def _action_spec(self) -> ActionSpec:
        # Observed layout of the 2026-challenge-demos 23D action vector: grippers
        # sit at the tail of each arm block (dims 14 and 22), not appended at the end.
        return build_action_spec(
            Joint(n=3, label="base"),
            Joint(n=4, label="trunk"),
            Joint(n=7, label="left_arm"),
            Gripper(prefix="left_"),
            Joint(n=7, label="right_arm"),
            Gripper(prefix="right_"),
        )

    @classmethod
    def _stats_path(cls) -> Path:
        # Only consulted when action_normalization is not None (default is None:
        # raw absolute joint targets, like the DROID joint_pos recipe).
        return _NORMALIZERS_DIR / "behavior1k_lerobot_joint_pos.json"

    # ---- index helpers -----------------------------------------------------

    @staticmethod
    def _split_episode_ids(ep_ids: list[int], split: str, val_ratio: float, seed: int) -> set[int]:
        if split == "full":
            return set(int(v) for v in ep_ids)
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}.")
        n_val = max(1, int(round(len(ep_ids) * val_ratio)))
        rng = random.Random(seed)  # identical selection on every rank
        val = set(int(v) for v in rng.sample(list(ep_ids), n_val))
        if split == "train":
            return set(int(v) for v in ep_ids) - val
        return val  # val/valid/validation/eval/test

    def __len__(self) -> int:
        return int(self._valid_cum[-1]) if self._valid_cum.size else 0

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        """Per-episode ``(start, length)`` flat-index blocks for
        ``ActionIterableShuffleDataset``."""
        blocks: list[tuple[int, int]] = []
        prev = 0
        for c in np.asarray(self._valid_cum).tolist():
            c = int(c)
            if c > prev:
                blocks.append((prev, c - prev))
            prev = c
        return blocks

    # ---- sample build ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Resample a different valid window if a frame fails to decode (bounded retries).
        n = len(self)
        last_err: Exception | None = None
        for _attempt in range(8):
            try:
                return self._build_item(idx)
            except Exception as e:  # noqa: BLE001 — skip past undecodable frames
                last_err = e
                log.warning(f"BEHAVIOR-1K: sample idx={idx} failed to load ({type(e).__name__}: {e}); resampling")
                if n > 0:
                    idx = random.randint(0, n - 1)
        raise RuntimeError(f"BEHAVIOR-1K: failed to load a sample after 8 resamples; last error: {last_err}")

    def _build_item(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = int(idx)
        ep = int(np.searchsorted(self._valid_cum, idx, side="right"))
        prev = int(self._valid_cum[ep - 1]) if ep > 0 else 0
        start = int(self._ep_starts[ep]) + (idx - prev)
        episode_index = int(self._ep_vals[ep])
        episode = self._episodes[episode_index]

        stop = start + self._chunk_length + 1
        timestamps = [float(self._row_timestamp[j]) for j in range(start, stop)]
        video = self._load_video(episode, timestamps)

        # joint_pos: the chunk of raw absolute joint targets is the stored action directly.
        raw = self._row_action[start : start + self._chunk_length]  # [chunk, 23]
        action = torch.from_numpy(np.ascontiguousarray(raw)).float()

        task = self._tasks[int(self._row_task[start])]
        ai_caption = random.choice([p.strip() for p in task.split(" | ") if p.strip()] or [task])

        extras: dict[str, Any] = {"additional_view_description": CONCAT_VIEW_LAYOUT_DESCRIPTION}
        return self._build_result(mode=mode, video=video, action=action, ai_caption=ai_caption, **extras)

    def _load_video(self, episode: dict[str, Any], timestamps: list[float]) -> torch.Tensor:
        # lerobot is a heavy, optional ("train" extra) dependency; import lazily.
        from lerobot.datasets.video_utils import decode_video_frames

        frames_by_view = {}
        for key in self._video_keys:
            from_ts = float(episode.get(f"videos/{key}/from_timestamp", 0.0))
            frames = decode_video_frames(
                self._video_path(episode, key),
                [from_ts + ts for ts in timestamps],
                self._tolerance_s,
            )  # [T, C, H, W] in [0, 1]
            frames_by_view[key] = frames
        return self._compose_multi_view(frames_by_view)

    def _compose_multi_view(self, frames_by_view: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compose head, left-wrist, and right-wrist views into one frame.

        Layout (per frame)::

            ┌──────────────┐
            │     head     │   (H, W)        zed, 720x720 native
            ├───────┬──────┤
            │ left  │ right│   (H/2, W/2)    realsense wrists, downscaled
            └───────┴──────┘

        Output is ``(T, C, 3H/2, W)`` — h/w = 1.5, snapping to the exact-aspect
        "2,3" 512x768 canvas at the "480" resolution tier.
        """
        head = frames_by_view[_HEAD_FEATURE]  # [T,C,H,W]
        left = frames_by_view[_LEFT_WRIST_FEATURE]
        right = frames_by_view[_RIGHT_WRIST_FEATURE]

        _, _, h, w = head.shape
        half_h, half_w = h // 2, w // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)  # [T,C,H/2,W]
        return torch.cat([head, bottom], dim=-2)  # [T,C,3H/2,W]
