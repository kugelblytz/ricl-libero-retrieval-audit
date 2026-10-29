"""Audit LIBERO-90 RICL and normal fine-tuning comparability."""

from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np


SHARED_SNIPPETS = (
    'weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_libero/params")',
    "num_train_steps=10_000",
    "batch_size=16",
    "ema_decay=None",
    "log_interval=1",
    "save_interval=300",
    "keep_period=300",
    "warmup_steps=300",
    "peak_lr=2.5e-5",
    "decay_steps=3_000",
    "decay_lr=2.5e-6",
    "action_dim=7",
    "action_horizon=10",
    "max_token_len=250",
    ".get_freeze_filter_with_frozen_img_encoder()",
)


def _load_metadata(preprocessing_dir: pathlib.Path) -> dict:
    metadata_path = preprocessing_dir / "retrieval_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _episode_cache_frames(episodes_dir: pathlib.Path) -> tuple[set[tuple[int, int]], dict[int, int]]:
    frames = set()
    task_counts: dict[int, int] = {}
    for episode_path in sorted(episodes_dir.glob("episode_*.npz")):
        episode_index = int(episode_path.stem.removeprefix("episode_"))
        with np.load(episode_path, allow_pickle=False) as episode:
            task_index = int(episode["task_index"].item())
            frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
        task_counts[task_index] = task_counts.get(task_index, 0) + 1
        frames.update((episode_index, int(frame_index)) for frame_index in frame_indices)
    if not frames:
        raise FileNotFoundError(f"No cached episode frames found in {episodes_dir}")
    return frames, task_counts


def _extract_train_config_block(config_text: str, config_name: str) -> str:
    name_match = re.search(rf'name="{re.escape(config_name)}"', config_text)
    if name_match is None:
        raise ValueError(f"Could not find TrainConfig named {config_name}")

    start = config_text.rfind("TrainConfig(", 0, name_match.start())
    if start == -1:
        raise ValueError(f"Could not find TrainConfig block for {config_name}")

    depth = 0
    for idx in range(start, len(config_text)):
        char = config_text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return config_text[start : idx + 1]

    raise ValueError(f"Unterminated TrainConfig block for {config_name}")


def _assert_shared_config_settings(config_path: pathlib.Path) -> None:
    config_text = config_path.read_text()
    ricl_block = _extract_train_config_block(config_text, "pi0_fast_libero90_ricl")
    normal_block = _extract_train_config_block(config_text, "pi0_fast_libero90_normal")

    missing = []
    for snippet in SHARED_SNIPPETS:
        if snippet not in ricl_block:
            missing.append(f"RICL config missing {snippet}")
        if snippet not in normal_block:
            missing.append(f"normal config missing {snippet}")

    if "Pi0FASTRiclConfig" not in ricl_block:
        missing.append("RICL config must use Pi0FASTRiclConfig")
    if "Pi0FASTConfig" not in normal_block:
        missing.append("normal config must use Pi0FASTConfig")
    if "preprocessing_dir=pathlib.Path(\"ricl_libero_preprocessing/libero90_train20x20\")" not in ricl_block:
        missing.append("RICL config must use the shared libero90_train20x20 preprocessing dir")
    if "cached_episodes_dir=pathlib.Path(\"ricl_libero_preprocessing/libero90_train20x20/episodes\")" not in normal_block:
        missing.append("normal config must use the shared libero90_train20x20 episode cache")

    if missing:
        raise AssertionError("\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--expect-frames", type=int, default=60859)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    preprocessing_dir = repo_root / "ricl_libero_preprocessing/libero90_train20x20"
    episodes_dir = preprocessing_dir / "episodes"
    normal_norm_stats = repo_root / "assets/pi0_fast_libero90_normal/libero/norm_stats.json"

    metadata = _load_metadata(preprocessing_dir)
    index_file = metadata.get("index_file", "libero10_indices_and_distances.npz")
    index_path = preprocessing_dir / index_file
    if not index_path.exists():
        raise FileNotFoundError(f"Missing RICL index file: {index_path}")

    cache_frames, task_counts = _episode_cache_frames(episodes_dir)
    query_indices = np.asarray(np.load(index_path)["query_indices"], dtype=np.int32)
    query_frames = {(int(ep_idx), int(frame_idx)) for ep_idx, frame_idx in query_indices}

    expected_task_counts = {task_index: 20 for task_index in range(20)}
    if task_counts != expected_task_counts:
        raise AssertionError(f"Expected 20 demos for tasks 0..19, got {sorted(task_counts.items())}")
    if cache_frames != query_frames:
        raise AssertionError(
            f"RICL query frames and normal cache frames differ: "
            f"cache={len(cache_frames)} query={len(query_frames)}"
        )
    if len(cache_frames) != args.expect_frames:
        raise AssertionError(f"Expected {args.expect_frames} frames, got {len(cache_frames)}")
    if not normal_norm_stats.exists():
        raise FileNotFoundError(f"Missing normal norm stats: {normal_norm_stats}")

    _assert_shared_config_settings(repo_root / "src/openpi/training/config.py")

    print("LIBERO-90 comparability audit passed")
    print(f"episodes={sum(task_counts.values())} tasks={len(task_counts)} frames={len(cache_frames)} index={index_file}")


if __name__ == "__main__":
    main()
