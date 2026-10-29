#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from openpi.policies import libero_retrieval_benchmark as rb

BENCHMARK_VERSION = 1
METHODS = (
    "random",
    "state",
    "dino_top",
    "dino_top_wrist",
    "dino_top_state",
    "dino_top_wrist_state",
    "pi0_fast_first_fusion",
    "pi0_fast_first_fusion_no_prompt",
    "oracle_action",
)
RETRIEVAL_METHODS = METHODS[:-1]
METRIC_NAMES = (
    "top1_action_rmse",
    "best_of_4_action_rmse",
    "mean_top4_action_rmse",
    "oracle_action_rmse",
    "oracle_regret",
    "progress_error",
    "state_error",
    "unique_episodes",
    "retrieval_latency_ms",
)
SOLVABLE_TASKS = (
    20,
    21,
    24,
    25,
    28,
    29,
    30,
    31,
    34,
    35,
    43,
    44,
    46,
    49,
    50,
    51,
    52,
    55,
    56,
    57,
    59,
    60,
    61,
    68,
    69,
)
SMOKE_TASKS = (20, 30)
CALIBRATION_TASKS = tuple(range(20))
BETAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
EXPECTED_CANDIDATE_FRAMES = 69_990
EXPECTED_FULL_QUERIES = 7_500


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    repo_id: str
    retrieval_dir: Path


@dataclasses.dataclass
class CandidateData:
    task_id: int
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    progress: np.ndarray
    states: np.ndarray
    action_chunks: np.ndarray
    prompts: list[str]
    episode_paths: list[Path]
    top_embeddings: np.ndarray | None = None

    @property
    def order_hash(self) -> str:
        return order_hash(self.episode_indices, self.frame_indices)


@dataclasses.dataclass
class QueryData:
    task_id: int
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    progress: np.ndarray
    states: np.ndarray
    action_chunks: np.ndarray
    prompts: list[str]
    images: list[np.ndarray] | None = None
    wrist_images: list[np.ndarray] | None = None

    @property
    def order_hash(self) -> str:
        return order_hash(self.episode_indices, self.frame_indices)


class DatasetHandle:
    def __init__(self, repo_id: str):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.dataset = LeRobotDataset(repo_id, local_files_only=True)
        columns = set(self.dataset.hf_dataset.column_names)
        light_columns = [
            name
            for name in (
                "state",
                "actions",
                "episode_index",
                "frame_index",
                "task_index",
                "libero_global_task_id",
            )
            if name in columns
        ]
        self.light_dataset = self.dataset.hf_dataset.select_columns(light_columns)
        self.starts = np.asarray(self.dataset.episode_data_index["from"], dtype=np.int64)
        self.ends = np.asarray(self.dataset.episode_data_index["to"], dtype=np.int64)
        self._episodes_by_task: dict[int, list[int]] | None = None

    def episodes_by_task(self) -> dict[int, list[int]]:
        if self._episodes_by_task is not None:
            return self._episodes_by_task
        grouped: dict[int, list[int]] = {}
        columns = set(self.light_dataset.column_names)
        task_field = "libero_global_task_id" if "libero_global_task_id" in columns else "task_index"
        for episode_index, start in enumerate(self.starts):
            row = self.light_dataset[int(start)]
            task_id = to_int(row[task_field])
            grouped.setdefault(task_id, []).append(episode_index)
        self._episodes_by_task = grouped
        return grouped

    def load_episode_core(self, episode_index: int) -> tuple[np.ndarray, np.ndarray, str]:
        start, end = int(self.starts[episode_index]), int(self.ends[episode_index])
        rows = self.light_dataset[start:end]
        states = to_numpy(rows["state"]).astype(np.float32)
        actions = to_numpy(rows["actions"]).astype(np.float32)
        prompt = str(self.dataset.meta.episodes[episode_index]["tasks"][0])
        return states, actions, prompt

    def load_query_images(self, episode_index: int, steps: Sequence[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        start = int(self.starts[episode_index])
        images = []
        wrist_images = []
        for step in steps:
            row = self.dataset[start + int(step)]
            images.append(to_image_uint8(row["image"]))
            wrist_images.append(to_image_uint8(row["wrist_image"]))
        return images, wrist_images


def get_dataset_handle(handles: dict[str, DatasetHandle], repo_id: str) -> DatasetHandle:
    handle = handles.get(repo_id)
    if handle is None:
        handle = DatasetHandle(repo_id)
        handles[repo_id] = handle
    return handle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline same-task LIBERO retrieval benchmark.")
    parser.add_argument(
        "stage",
        choices=("validate", "extract-dino", "extract-pi0", "calibrate", "score"),
    )
    parser.add_argument("--split", choices=("smoke", "full", "calibration"), default="full")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/libero/retrieval_benchmark/libero90_solvable25_offline_v1"),
    )
    parser.add_argument("--config-name", default="pi0_fast_libero90_normal")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999"),
    )
    parser.add_argument("--dino-batch-size", type=int, default=64)
    parser.add_argument("--pi0-batch-size", type=int, default=4)
    parser.add_argument("--task-ids", default=None, help="Optional comma-separated subset for staged smoke work.")
    parser.add_argument("--calibration-output", type=Path, default=None)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        validate_environment(args)
    elif args.stage == "extract-dino":
        require_cuda()
        extract_dino(args)
    elif args.stage == "extract-pi0":
        require_cuda()
        extract_pi0(args)
    elif args.stage == "calibrate":
        calibrate(args)
    elif args.stage == "score":
        score(args)


def source_for_task(task_id: int, split: str) -> SourceSpec:
    if split == "calibration":
        return SourceSpec(
            repo_id="local/libero90_30tasks",
            retrieval_dir=Path(
                os.environ.get(
                    "LIBERO_CALIBRATION_RETRIEVAL_DIR",
                    "data/libero/retrieval_benchmark/libero90_solvable25_offline_v1/"
                    "calibration_source/retrieval",
                )
            ),
        )
    if 20 <= task_id <= 29:
        return SourceSpec(
            repo_id="local/libero90_official20to29",
            retrieval_dir=Path("ricl_libero_preprocessing/libero90_official20to29_spread20/retrieval"),
        )
    return SourceSpec(
        repo_id="local/libero90_official30to69",
        retrieval_dir=Path("ricl_libero_preprocessing/libero90_official30to69_leftout10_seeded20/retrieval"),
    )


def tasks_for_split(split: str) -> tuple[int, ...]:
    if split == "smoke":
        return SMOKE_TASKS
    if split == "calibration":
        return CALIBRATION_TASKS
    return SOLVABLE_TASKS


def selected_tasks(args: argparse.Namespace) -> tuple[int, ...]:
    available = tasks_for_split(args.split)
    if args.task_ids is None:
        return available
    requested = tuple(int(value.strip()) for value in args.task_ids.split(",") if value.strip())
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Task IDs {unknown} are not part of split {args.split}.")
    return requested


def validate_environment(args: argparse.Namespace) -> None:
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        raise RuntimeError("JAX_PLATFORMS must be set to cuda.")
    require_cuda()
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Normal Pi0-FAST checkpoint does not exist: {args.checkpoint_dir}")
    required_checkpoint_files = (
        args.checkpoint_dir / "_CHECKPOINT_METADATA",
        args.checkpoint_dir / "params" / "_METADATA",
        args.checkpoint_dir / "assets" / "libero" / "norm_stats.json",
    )
    for path in required_checkpoint_files:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint is incomplete; missing {path}")

    handles: dict[str, DatasetHandle] = {}
    candidate_frames = 0
    candidate_episodes = 0
    query_count = 0
    for task_id in SOLVABLE_TASKS:
        source = source_for_task(task_id, "full")
        validate_retrieval_task(source.retrieval_dir, task_id)
        handle = get_dataset_handle(handles, source.repo_id)
        candidate_ids = candidate_episode_ids(source.retrieval_dir, task_id)
        all_ids = handle.episodes_by_task().get(task_id, [])
        heldout_ids = sorted(set(all_ids) - set(candidate_ids))
        if len(heldout_ids) != 30:
            raise ValueError(f"Task {task_id}: expected 30 held-out episodes, found {len(heldout_ids)}.")
        candidate_episodes += len(candidate_ids)
        query_count += len(heldout_ids) * 10
        for path in candidate_episode_paths(source.retrieval_dir, task_id):
            with np.load(path, allow_pickle=True) as episode:
                candidate_frames += len(episode["state"])
    if candidate_episodes != 500:
        raise ValueError(f"Expected 500 candidate episodes, found {candidate_episodes}.")
    if candidate_frames != EXPECTED_CANDIDATE_FRAMES:
        raise ValueError(f"Expected {EXPECTED_CANDIDATE_FRAMES} candidate frames, found {candidate_frames}.")
    if query_count != EXPECTED_FULL_QUERIES:
        raise ValueError(f"Expected {EXPECTED_FULL_QUERIES} full queries, found {query_count}.")

    print(
        json.dumps(
            {
                "status": "ok",
                "tasks": len(SOLVABLE_TASKS),
                "candidate_episodes": candidate_episodes,
                "candidate_frames": candidate_frames,
                "queries": query_count,
                "checkpoint": str(args.checkpoint_dir),
            },
            indent=2,
        )
    )


def require_cuda() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"CUDA GPU is not visible through nvidia-smi: {result.stderr.strip()}")


def validate_retrieval_task(retrieval_dir: Path, task_id: int) -> None:
    paths = candidate_episode_paths(retrieval_dir, task_id)
    if len(paths) != 20:
        raise ValueError(f"Task {task_id}: expected 20 retrieval episodes in {retrieval_dir}, found {len(paths)}.")
    metadata_path = retrieval_dir / "retrieval_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing retrieval metadata: {metadata_path}")


def candidate_task_dir(retrieval_dir: Path, task_id: int) -> Path:
    padded = retrieval_dir / f"task_{task_id:02d}"
    return padded if padded.is_dir() else retrieval_dir / f"task_{task_id}"


def candidate_episode_paths(retrieval_dir: Path, task_id: int) -> list[Path]:
    return sorted(candidate_task_dir(retrieval_dir, task_id).glob("episode_*.npz"), key=episode_id_from_path)


def candidate_episode_ids(retrieval_dir: Path, task_id: int) -> list[int]:
    return [episode_id_from_path(path) for path in candidate_episode_paths(retrieval_dir, task_id)]


def episode_id_from_path(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def load_candidates(source: SourceSpec, task_id: int, *, include_top_embeddings: bool = False) -> CandidateData:
    paths = candidate_episode_paths(source.retrieval_dir, task_id)
    if len(paths) != 20:
        raise ValueError(f"Task {task_id}: expected 20 candidate episodes, found {len(paths)}.")
    episode_indices = []
    frame_indices = []
    progress = []
    states = []
    action_chunks = []
    prompts: list[str] = []
    top_embeddings = []
    for path in paths:
        with np.load(path, allow_pickle=True) as episode:
            episode_id = episode_id_from_path(path)
            episode_states = np.asarray(episode["state"], dtype=np.float32)
            episode_actions = np.asarray(episode["actions"], dtype=np.float32)
            episode_frames = np.asarray(
                episode["frame_indices"] if "frame_indices" in episode else np.arange(len(episode_states)),
                dtype=np.int32,
            )
            prompt = str(np.asarray(episode["prompt"]).item())
            if int(np.asarray(episode["task_index"]).item()) != task_id:
                raise ValueError(f"{path} does not belong to task {task_id}.")
            episode_indices.extend([episode_id] * len(episode_states))
            frame_indices.extend(episode_frames.tolist())
            progress.extend((episode_frames / max(int(episode_frames[-1]), 1)).tolist())
            states.append(episode_states)
            action_chunks.extend(rb.get_action_chunk(episode_actions, step) for step in range(len(episode_actions)))
            prompts.extend([prompt] * len(episode_states))
            if include_top_embeddings:
                if "image_embeddings" not in episode:
                    raise KeyError(f"{path} is missing image_embeddings.")
                top_embeddings.append(np.asarray(episode["image_embeddings"], dtype=np.float32))
    return CandidateData(
        task_id=task_id,
        episode_indices=np.asarray(episode_indices, dtype=np.int32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        progress=np.asarray(progress, dtype=np.float32),
        states=np.concatenate(states, axis=0),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        prompts=prompts,
        episode_paths=paths,
        top_embeddings=np.concatenate(top_embeddings, axis=0) if include_top_embeddings else None,
    )


def load_queries(
    handle: DatasetHandle,
    source: SourceSpec,
    task_id: int,
    *,
    split: str,
    include_images: bool,
) -> QueryData:
    candidate_ids = set(candidate_episode_ids(source.retrieval_dir, task_id))
    all_ids = handle.episodes_by_task().get(task_id, [])
    heldout_ids = sorted(set(all_ids) - candidate_ids)
    if len(heldout_ids) != 30:
        raise ValueError(f"Task {task_id}: expected 30 held-out episodes, found {len(heldout_ids)}.")
    if split == "smoke":
        heldout_ids = heldout_ids[:1]
        query_steps_per_episode = 3
    else:
        query_steps_per_episode = 10

    episode_indices = []
    frame_indices = []
    progress = []
    states = []
    action_chunks = []
    prompts = []
    images: list[np.ndarray] = []
    wrist_images: list[np.ndarray] = []
    for episode_id in heldout_ids:
        episode_states, episode_actions, prompt = handle.load_episode_core(episode_id)
        steps = rb.select_query_steps(len(episode_actions), count=query_steps_per_episode, action_horizon=10)
        episode_indices.extend([episode_id] * len(steps))
        frame_indices.extend(steps.tolist())
        progress.extend((steps / max(len(episode_actions) - 1, 1)).tolist())
        states.extend(episode_states[steps])
        action_chunks.extend(episode_actions[step : step + 10] for step in steps)
        prompts.extend([prompt] * len(steps))
        if include_images:
            episode_images, episode_wrist = handle.load_query_images(episode_id, steps)
            images.extend(episode_images)
            wrist_images.extend(episode_wrist)
    if candidate_ids & set(episode_indices):
        raise ValueError(f"Task {task_id}: query episodes overlap retrieval candidates.")
    return QueryData(
        task_id=task_id,
        episode_indices=np.asarray(episode_indices, dtype=np.int32),
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        progress=np.asarray(progress, dtype=np.float32),
        states=np.asarray(states, dtype=np.float32),
        action_chunks=np.asarray(action_chunks, dtype=np.float32),
        prompts=prompts,
        images=images if include_images else None,
        wrist_images=wrist_images if include_images else None,
    )


def extract_dino(args: argparse.Namespace) -> None:
    import torch

    from openpi.policies.utils import embed
    from openpi.policies.utils import get_embed_dim
    from openpi.policies.utils import load_dinov2

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; DINO extraction is CUDA-only.")
    model = load_dinov2("cuda")
    dim = get_embed_dim("64PATCHES")
    handles: dict[str, DatasetHandle] = {}
    for task_id in selected_tasks(args):
        source = source_for_task(task_id, args.split)
        handle = get_dataset_handle(handles, source.repo_id)
        candidates = load_candidates(source, task_id)
        queries = load_queries(handle, source, task_id, split=args.split, include_images=True)
        candidate_path = feature_path(args.output_root, "candidate", "dino_wrist", task_id)
        query_top_path = feature_path(args.output_root, f"query_{args.split}", "dino_top", task_id)
        query_wrist_path = feature_path(args.output_root, f"query_{args.split}", "dino_wrist", task_id)
        candidate_signature = feature_signature(candidates.order_hash, "dino_wrist", source, handle)
        query_top_signature = feature_signature(queries.order_hash, "dino_top", source, handle)
        query_wrist_signature = feature_signature(queries.order_hash, "dino_wrist", source, handle)

        if not cache_valid(candidate_path, candidate_signature, overwrite=args.overwrite_cache):
            print(f"[dino] task {task_id}: extracting candidate wrist features ({len(candidates.states)} frames)")
            memmap, temporary = create_memmap(candidate_path, (len(candidates.states), dim))
            offset = 0
            for episode_number, episode_path in enumerate(candidates.episode_paths, start=1):
                with np.load(episode_path, allow_pickle=True) as episode:
                    wrist_images = np.asarray(episode["wrist_image"], dtype=np.uint8)
                    for start in range(0, len(wrist_images), args.dino_batch_size):
                        batch = wrist_images[start : start + args.dino_batch_size]
                        features = embed(batch, model, embedding_type="64PATCHES").astype(np.float32)
                        memmap[offset : offset + len(features)] = features
                        offset += len(features)
                if episode_number % 5 == 0 or episode_number == len(candidates.episode_paths):
                    print(
                        f"[dino] task {task_id}: candidate episodes "
                        f"{episode_number}/{len(candidates.episode_paths)}, frames {offset}/{len(candidates.states)}"
                    )
            finish_memmap(memmap, temporary, candidate_path, candidate_signature)

        assert queries.images is not None
        assert queries.wrist_images is not None
        if not cache_valid(query_top_path, query_top_signature, overwrite=args.overwrite_cache):
            write_dino_image_cache(
                query_top_path,
                queries.images,
                query_top_signature,
                model,
                args.dino_batch_size,
                dim,
                embed,
            )
        if not cache_valid(query_wrist_path, query_wrist_signature, overwrite=args.overwrite_cache):
            write_dino_image_cache(
                query_wrist_path,
                queries.wrist_images,
                query_wrist_signature,
                model,
                args.dino_batch_size,
                dim,
                embed,
            )
        print(f"[dino] task {task_id}: caches ready")


def write_dino_image_cache(
    path: Path,
    images: Sequence[np.ndarray],
    expected_hash: str,
    model: Any,
    batch_size: int,
    dim: int,
    embed_fn: Any,
) -> None:
    memmap, temporary = create_memmap(path, (len(images), dim))
    for start in range(0, len(images), batch_size):
        batch = np.stack(images[start : start + batch_size])
        memmap[start : start + len(batch)] = embed_fn(batch, model, embedding_type="64PATCHES").astype(np.float32)
    finish_memmap(memmap, temporary, path, expected_hash)


class Pi0FirstFusionEmbedder:
    def __init__(self, config_name: str, checkpoint_dir: Path):
        import jax
        import jax.numpy as jnp

        from openpi import transforms
        from openpi.models import model as model_lib
        from openpi.shared import nnx_utils
        from openpi.training import checkpoints
        from openpi.training import config as train_config_lib

        if not jax.devices() or any(device.platform != "gpu" for device in jax.devices()):
            raise RuntimeError(f"Pi0 extraction requires CUDA-only JAX devices, got {jax.devices()}.")
        config = train_config_lib.get_config(config_name)
        model_config = dataclasses.replace(config.model, dtype="float32")
        params = model_lib.restore_params(checkpoint_dir / "params", dtype=jnp.float32)
        self.model = model_config.load(params)
        data_config = config.data.create(config.assets_dirs, model_config)
        if data_config.asset_id is None:
            raise ValueError("Normal Pi0-FAST config must define an asset_id.")
        norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        self.transform = transforms.compose(
            [
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self.model_lib = model_lib
        self.extract_pair = nnx_utils.module_jit(self.model.embed_first_fusion_feature_pair)
        self.jax = jax
        self.jnp = jnp

    def __call__(
        self,
        images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray],
        states: np.ndarray,
        prompts: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        full_items = []
        blank_items = []
        for image, wrist_image, state, prompt in zip(images, wrist_images, states, prompts, strict=True):
            raw = rb.make_pi0_observation_input(image, wrist_image, state, prompt)
            blank_raw = {**raw, "prompt": ""}
            if "actions" in raw or "actions" in blank_raw:
                raise AssertionError("Pi0 retrieval feature inputs must never contain actions.")
            full = self.transform(raw)
            blank = self.transform(blank_raw)
            if np.array_equal(full["tokenized_prompt"], blank["tokenized_prompt"]):
                raise AssertionError("Full-prompt and blank-prompt Pi0 token sequences unexpectedly match.")
            full_items.append(full)
            blank_items.append(blank)
        full_batch = stack_trees(full_items)
        blank_batch = stack_trees(blank_items)
        full_batch = self.jax.tree.map(self.jnp.asarray, full_batch)
        blank_batch = self.jax.tree.map(self.jnp.asarray, blank_batch)
        full_obs = self.model_lib.Observation.from_dict(full_batch)
        blank_obs = self.model_lib.Observation.from_dict(blank_batch)
        full_features, blank_features = self.extract_pair(full_obs, blank_obs)
        return np.asarray(self.jax.device_get(full_features)), np.asarray(self.jax.device_get(blank_features))


def extract_pi0(args: argparse.Namespace) -> None:
    embedder = Pi0FirstFusionEmbedder(args.config_name, args.checkpoint_dir)
    handles: dict[str, DatasetHandle] = {}
    for task_id in selected_tasks(args):
        source = source_for_task(task_id, args.split)
        handle = get_dataset_handle(handles, source.repo_id)
        candidates = load_candidates(source, task_id)
        queries = load_queries(handle, source, task_id, split=args.split, include_images=True)
        candidate_full_path = feature_path(args.output_root, "candidate", "pi0_fast_first_fusion", task_id)
        candidate_blank_path = feature_path(
            args.output_root, "candidate", "pi0_fast_first_fusion_no_prompt", task_id
        )
        query_full_path = feature_path(
            args.output_root, f"query_{args.split}", "pi0_fast_first_fusion", task_id
        )
        query_blank_path = feature_path(
            args.output_root, f"query_{args.split}", "pi0_fast_first_fusion_no_prompt", task_id
        )
        candidate_full_signature = feature_signature(
            candidates.order_hash, "pi0_fast_first_fusion", source, handle, checkpoint_dir=args.checkpoint_dir
        )
        candidate_blank_signature = feature_signature(
            candidates.order_hash,
            "pi0_fast_first_fusion_no_prompt",
            source,
            handle,
            checkpoint_dir=args.checkpoint_dir,
        )
        query_full_signature = feature_signature(
            queries.order_hash, "pi0_fast_first_fusion", source, handle, checkpoint_dir=args.checkpoint_dir
        )
        query_blank_signature = feature_signature(
            queries.order_hash,
            "pi0_fast_first_fusion_no_prompt",
            source,
            handle,
            checkpoint_dir=args.checkpoint_dir,
        )

        candidate_ready = cache_valid(
            candidate_full_path, candidate_full_signature, overwrite=args.overwrite_cache
        ) and cache_valid(
            candidate_blank_path, candidate_blank_signature, overwrite=args.overwrite_cache
        )
        if not candidate_ready:
            print(f"[pi0] task {task_id}: extracting candidate prefix features ({len(candidates.states)} frames)")
            full_parts = []
            blank_parts = []
            processed_frames = 0
            for episode_number, episode_path in enumerate(candidates.episode_paths, start=1):
                with np.load(episode_path, allow_pickle=True) as episode:
                    images = np.asarray(episode["image"], dtype=np.uint8)
                    wrist_images = np.asarray(episode["wrist_image"], dtype=np.uint8)
                    states = np.asarray(episode["state"], dtype=np.float32)
                    prompts = [str(np.asarray(episode["prompt"]).item())] * len(states)
                    full, blank = extract_pi0_batches(
                        embedder, images, wrist_images, states, prompts, args.pi0_batch_size
                    )
                    full_parts.append(full)
                    blank_parts.append(blank)
                    processed_frames += len(states)
                if episode_number % 5 == 0 or episode_number == len(candidates.episode_paths):
                    print(
                        f"[pi0] task {task_id}: candidate episodes "
                        f"{episode_number}/{len(candidates.episode_paths)}, "
                        f"frames {processed_frames}/{len(candidates.states)}"
                    )
            write_array_cache(candidate_full_path, np.concatenate(full_parts), candidate_full_signature)
            write_array_cache(candidate_blank_path, np.concatenate(blank_parts), candidate_blank_signature)

        assert queries.images is not None
        assert queries.wrist_images is not None
        query_ready = cache_valid(
            query_full_path, query_full_signature, overwrite=args.overwrite_cache
        ) and cache_valid(
            query_blank_path, query_blank_signature, overwrite=args.overwrite_cache
        )
        if not query_ready:
            print(f"[pi0] task {task_id}: extracting query prefix features ({len(queries.states)} frames)")
            full, blank = extract_pi0_batches(
                embedder,
                queries.images,
                queries.wrist_images,
                queries.states,
                queries.prompts,
                args.pi0_batch_size,
            )
            write_array_cache(query_full_path, full, query_full_signature)
            write_array_cache(query_blank_path, blank, query_blank_signature)
        print(f"[pi0] task {task_id}: caches ready")


def extract_pi0_batches(
    embedder: Pi0FirstFusionEmbedder,
    images: Sequence[np.ndarray],
    wrist_images: Sequence[np.ndarray],
    states: np.ndarray,
    prompts: Sequence[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    full_parts = []
    blank_parts = []
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        full, blank = embedder(images[start:end], wrist_images[start:end], states[start:end], prompts[start:end])
        if not np.all(np.isfinite(full)) or not np.all(np.isfinite(blank)):
            raise ValueError("Pi0 feature extraction produced NaN or infinity.")
        full_parts.append(full.astype(np.float32))
        blank_parts.append(blank.astype(np.float32))
    return np.concatenate(full_parts), np.concatenate(blank_parts)


def calibrate(args: argparse.Namespace) -> None:
    if args.split != "calibration":
        raise ValueError("The calibrate stage requires --split calibration.")
    rows = []
    handles: dict[str, DatasetHandle] = {}
    action_stats = load_action_stats(args.checkpoint_dir)
    state_stats = load_state_stats(args.checkpoint_dir)
    calibration_tasks = selected_tasks(args)
    for task_id in calibration_tasks:
        source = source_for_task(task_id, "calibration")
        handle = get_dataset_handle(handles, source.repo_id)
        candidates = load_candidates(source, task_id, include_top_embeddings=True)
        queries = load_queries(handle, source, task_id, split="calibration", include_images=False)
        wrist_candidates = load_feature(
            feature_path(args.output_root, "candidate", "dino_wrist", task_id),
            feature_signature(candidates.order_hash, "dino_wrist", source, handle),
        )
        top_queries = load_feature(
            feature_path(args.output_root, "query_calibration", "dino_top", task_id),
            feature_signature(queries.order_hash, "dino_top", source, handle),
        )
        wrist_queries = load_feature(
            feature_path(args.output_root, "query_calibration", "dino_wrist", task_id),
            feature_signature(queries.order_hash, "dino_wrist", source, handle),
        )
        normalized_candidate_states = rb.z_normalize(candidates.states, *state_stats)
        normalized_query_states = rb.z_normalize(queries.states, *state_stats)
        normalized_candidate_actions = rb.quantile_normalize(candidates.action_chunks, *action_stats)
        normalized_query_actions = rb.quantile_normalize(queries.action_chunks, *action_stats)
        family_scores = {family: {beta: [] for beta in BETAS} for family in ("dino_top_state", "dino_top_wrist_state")}
        assert candidates.top_embeddings is not None
        for query_index in range(len(queries.states)):
            top_sq = squared_l2_distances(candidates.top_embeddings, top_queries[query_index])
            wrist_sq = squared_l2_distances(wrist_candidates, wrist_queries[query_index])
            top_distances = np.sqrt(top_sq)
            both_distances = np.sqrt(top_sq + wrist_sq)
            state_distances = rb.l2_distances(normalized_candidate_states, normalized_query_states[query_index])
            action_errors = rb.action_rmse(normalized_candidate_actions, normalized_query_actions[query_index])
            for family, image_distances in (
                ("dino_top_state", top_distances),
                ("dino_top_wrist_state", both_distances),
            ):
                for beta in BETAS:
                    distances = rb.hybrid_distances(image_distances, state_distances, beta)
                    selected = rb.deterministic_top_k(
                        distances, candidates.episode_indices, candidates.frame_indices
                    )
                    family_scores[family][beta].append(float(np.min(action_errors[selected])))
        for family, beta_scores in family_scores.items():
            for beta, scores in beta_scores.items():
                rows.append(
                    {
                        "family": family,
                        "beta": beta,
                        "task_id": task_id,
                        "query_count": len(scores),
                        "best_of_4_action_rmse": float(np.mean(scores)),
                    }
                )

    selections = {}
    for family in ("dino_top_state", "dino_top_wrist_state"):
        beta_macro = {}
        for beta in BETAS:
            task_values = [row["best_of_4_action_rmse"] for row in rows if row["family"] == family and row["beta"] == beta]
            beta_macro[beta] = float(np.mean(task_values))
        selected_beta = min(BETAS, key=lambda beta: (beta_macro[beta], beta))
        selections[family] = {"beta": selected_beta, "macro_best_of_4_action_rmse": beta_macro[selected_beta]}
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "calibration_tasks": list(calibration_tasks),
        "betas": list(BETAS),
        "selection_metric": "macro_best_of_4_action_rmse",
        "selections": selections,
        "rows": rows,
    }
    path = args.calibration_output or args.output_root / "calibration.json"
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))
    write_csv(args.output_root / "calibration.csv", rows)
    print(json.dumps(selections, indent=2))


def score(args: argparse.Namespace) -> None:
    if args.split == "calibration":
        raise ValueError("Use the calibrate stage for the calibration split.")
    calibration_path = args.calibration_path or args.output_root / "calibration.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration must complete before scoring: {calibration_path}")
    calibration = json.loads(calibration_path.read_text())
    betas = {family: float(value["beta"]) for family, value in calibration["selections"].items()}
    action_stats = load_action_stats(args.checkpoint_dir)
    state_stats = load_state_stats(args.checkpoint_dir)
    handles: dict[str, DatasetHandle] = {}
    score_dir = args.output_root / "scores" / args.split
    score_dir.mkdir(parents=True, exist_ok=True)
    expected_queries_per_task = 3 if args.split == "smoke" else 300
    expected_rows_per_task = expected_queries_per_task * len(METHODS)

    benchmark_tasks = selected_tasks(args)
    for task_id in benchmark_tasks:
        task_output = score_dir / f"task_{task_id:02d}.jsonl"
        if args.resume and jsonl_row_count(task_output) == expected_rows_per_task:
            print(f"[score] task {task_id}: complete, skipping")
            continue
        source = source_for_task(task_id, args.split)
        handle = get_dataset_handle(handles, source.repo_id)
        candidates = load_candidates(source, task_id, include_top_embeddings=True)
        queries = load_queries(handle, source, task_id, split=args.split, include_images=False)
        features = load_scoring_features(
            args.output_root,
            args.split,
            task_id,
            candidates,
            queries,
            source,
            handle,
            args.checkpoint_dir,
        )
        task_rows = score_task(
            candidates,
            queries,
            features,
            betas,
            action_stats,
            state_stats,
            seed=args.seed,
        )
        if len(task_rows) != expected_rows_per_task:
            raise ValueError(f"Task {task_id}: expected {expected_rows_per_task} score rows, got {len(task_rows)}.")
        atomic_write_jsonl(task_output, task_rows)
        print(f"[score] task {task_id}: wrote {len(task_rows)} rows")

    combined_rows = []
    for task_id in benchmark_tasks:
        combined_rows.extend(read_jsonl(score_dir / f"task_{task_id:02d}.jsonl"))
    expected_total = len(benchmark_tasks) * expected_rows_per_task
    if len(combined_rows) != expected_total:
        raise ValueError(f"Expected {expected_total} combined score rows, got {len(combined_rows)}.")
    results_path = args.output_root / args.split / "queries.jsonl"
    atomic_write_jsonl(results_path, combined_rows)

    per_task, summary = rb.macro_average(combined_rows, metric_names=METRIC_NAMES)
    add_method_ranks(summary)
    pairwise = pairwise_rows(per_task)
    write_csv(args.output_root / args.split / "per_task.csv", per_task)
    write_csv(args.output_root / args.split / "summary.csv", summary)
    write_csv(args.output_root / args.split / "pairwise.csv", pairwise)
    summary_payload = {
        "split": args.split,
        "query_rows": len(combined_rows),
        "unique_queries": len(combined_rows) // len(METHODS),
        "methods": list(METHODS),
        "summary": summary,
        "pairwise": pairwise,
    }
    atomic_write_text(args.output_root / args.split / "summary.json", json.dumps(summary_payload, indent=2))

    if args.split == "smoke":
        verify_smoke_determinism(args.output_root, combined_rows, calibration)
    else:
        write_manifest(args, combined_rows, summary, pairwise, calibration)
    print(f"[score] {args.split}: {len(combined_rows) // len(METHODS)} queries scored")


def load_scoring_features(
    output_root: Path,
    split: str,
    task_id: int,
    candidates: CandidateData,
    queries: QueryData,
    source: SourceSpec,
    handle: DatasetHandle,
    checkpoint_dir: Path,
) -> dict[str, np.ndarray]:
    features = {
        "candidate_dino_wrist": load_feature(
            feature_path(output_root, "candidate", "dino_wrist", task_id),
            feature_signature(candidates.order_hash, "dino_wrist", source, handle),
        ),
        "query_dino_top": load_feature(
            feature_path(output_root, f"query_{split}", "dino_top", task_id),
            feature_signature(queries.order_hash, "dino_top", source, handle),
        ),
        "query_dino_wrist": load_feature(
            feature_path(output_root, f"query_{split}", "dino_wrist", task_id),
            feature_signature(queries.order_hash, "dino_wrist", source, handle),
        ),
        "candidate_pi0_fast_first_fusion": load_feature(
            feature_path(output_root, "candidate", "pi0_fast_first_fusion", task_id),
            feature_signature(
                candidates.order_hash,
                "pi0_fast_first_fusion",
                source,
                handle,
                checkpoint_dir=checkpoint_dir,
            ),
        ),
        "candidate_pi0_fast_first_fusion_no_prompt": load_feature(
            feature_path(output_root, "candidate", "pi0_fast_first_fusion_no_prompt", task_id),
            feature_signature(
                candidates.order_hash,
                "pi0_fast_first_fusion_no_prompt",
                source,
                handle,
                checkpoint_dir=checkpoint_dir,
            ),
        ),
        "query_pi0_fast_first_fusion": load_feature(
            feature_path(output_root, f"query_{split}", "pi0_fast_first_fusion", task_id),
            feature_signature(
                queries.order_hash,
                "pi0_fast_first_fusion",
                source,
                handle,
                checkpoint_dir=checkpoint_dir,
            ),
        ),
        "query_pi0_fast_first_fusion_no_prompt": load_feature(
            feature_path(output_root, f"query_{split}", "pi0_fast_first_fusion_no_prompt", task_id),
            feature_signature(
                queries.order_hash,
                "pi0_fast_first_fusion_no_prompt",
                source,
                handle,
                checkpoint_dir=checkpoint_dir,
            ),
        ),
    }
    expected_candidate_count = len(candidates.states)
    expected_query_count = len(queries.states)
    for name, values in features.items():
        expected = expected_candidate_count if name.startswith("candidate") else expected_query_count
        if len(values) != expected:
            raise ValueError(f"{name}: expected {expected} features, found {len(values)}.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity.")
    return features


def score_task(
    candidates: CandidateData,
    queries: QueryData,
    features: Mapping[str, np.ndarray],
    betas: Mapping[str, float],
    action_stats: tuple[np.ndarray, np.ndarray],
    state_stats: tuple[np.ndarray, np.ndarray],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    assert candidates.top_embeddings is not None
    normalized_candidate_states = rb.z_normalize(candidates.states, *state_stats)
    normalized_query_states = rb.z_normalize(queries.states, *state_stats)
    normalized_candidate_actions = rb.quantile_normalize(candidates.action_chunks, *action_stats)
    normalized_query_actions = rb.quantile_normalize(queries.action_chunks, *action_stats)
    rows = []
    for query_index in range(len(queries.states)):
        started = time.perf_counter()
        top_sq = squared_l2_distances(candidates.top_embeddings, features["query_dino_top"][query_index])
        top_distance_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        wrist_sq = squared_l2_distances(
            features["candidate_dino_wrist"], features["query_dino_wrist"][query_index]
        )
        wrist_distance_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        state_distances = rb.l2_distances(normalized_candidate_states, normalized_query_states[query_index])
        state_distance_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        dino_top_distances = np.sqrt(top_sq)
        dino_top_finalize_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        dino_top_wrist_distances = np.sqrt(top_sq + wrist_sq)
        dino_top_wrist_finalize_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        pi0_full_distances = np.sqrt(
            squared_l2_distances(
                features["candidate_pi0_fast_first_fusion"],
                features["query_pi0_fast_first_fusion"][query_index],
            )
        )
        pi0_full_distance_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        pi0_no_prompt_distances = np.sqrt(
            squared_l2_distances(
                features["candidate_pi0_fast_first_fusion_no_prompt"],
                features["query_pi0_fast_first_fusion_no_prompt"][query_index],
            )
        )
        pi0_no_prompt_distance_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        dino_top_state_distances = rb.hybrid_distances(
            dino_top_distances, state_distances, betas["dino_top_state"]
        )
        dino_top_state_finalize_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        dino_top_wrist_state_distances = rb.hybrid_distances(
            dino_top_wrist_distances, state_distances, betas["dino_top_wrist_state"]
        )
        dino_top_wrist_state_finalize_ms = (time.perf_counter() - started) * 1000.0

        distances_by_method = {
            "state": state_distances,
            "dino_top": dino_top_distances,
            "dino_top_wrist": dino_top_wrist_distances,
            "dino_top_state": dino_top_state_distances,
            "dino_top_wrist_state": dino_top_wrist_state_distances,
            "pi0_fast_first_fusion": pi0_full_distances,
            "pi0_fast_first_fusion_no_prompt": pi0_no_prompt_distances,
        }
        distance_latency_ms = {
            "state": state_distance_ms,
            "dino_top": top_distance_ms + dino_top_finalize_ms,
            "dino_top_wrist": top_distance_ms + wrist_distance_ms + dino_top_wrist_finalize_ms,
            "dino_top_state": (
                top_distance_ms + dino_top_finalize_ms + state_distance_ms + dino_top_state_finalize_ms
            ),
            "dino_top_wrist_state": (
                top_distance_ms
                + wrist_distance_ms
                + dino_top_wrist_finalize_ms
                + state_distance_ms
                + dino_top_wrist_state_finalize_ms
            ),
            "pi0_fast_first_fusion": pi0_full_distance_ms,
            "pi0_fast_first_fusion_no_prompt": pi0_no_prompt_distance_ms,
        }
        action_errors = rb.action_rmse(normalized_candidate_actions, normalized_query_actions[query_index])

        random_start = time.perf_counter()
        random_indices = rb.random_top_k(
            len(candidates.states),
            repeats=20,
            seed=seed + candidates.task_id * 1_000_003 + int(queries.episode_indices[query_index]) * 1009 + query_index,
        )
        repeated_random_indices = rb.random_top_k(
            len(candidates.states),
            repeats=20,
            seed=seed + candidates.task_id * 1_000_003 + int(queries.episode_indices[query_index]) * 1009 + query_index,
        )
        if not all(np.array_equal(first, second) for first, second in zip(random_indices, repeated_random_indices, strict=True)):
            raise AssertionError("Seeded random retrieval is not deterministic.")
        random_metric_rows = [
            metrics_for_selection(
                selected,
                candidates,
                queries,
                query_index,
                normalized_candidate_states,
                normalized_query_states,
                normalized_candidate_actions,
                normalized_query_actions,
                action_errors,
            )
            for selected in random_indices
        ]
        random_metrics = rb.average_metric_dicts(random_metric_rows)
        random_metrics["retrieval_latency_ms"] = (time.perf_counter() - random_start) * 1000.0 / 20.0
        rows.append(
            result_row(
                "random",
                candidates,
                queries,
                query_index,
                random_indices[0],
                random_metrics,
                extra={"random_repeats": 20},
            )
        )

        for method, distances in distances_by_method.items():
            started = time.perf_counter()
            selected = rb.deterministic_top_k(distances, candidates.episode_indices, candidates.frame_indices)
            repeated = rb.deterministic_top_k(distances, candidates.episode_indices, candidates.frame_indices)
            if not np.array_equal(selected, repeated):
                raise AssertionError(f"{method} retrieval is not deterministic.")
            latency = (time.perf_counter() - started) * 1000.0
            metrics = metrics_for_selection(
                selected,
                candidates,
                queries,
                query_index,
                normalized_candidate_states,
                normalized_query_states,
                normalized_candidate_actions,
                normalized_query_actions,
                action_errors,
            )
            metrics["retrieval_latency_ms"] = distance_latency_ms[method] + latency
            extra = {"beta": betas[method]} if method in betas else {}
            rows.append(result_row(method, candidates, queries, query_index, selected, metrics, extra=extra))

        oracle_started = time.perf_counter()
        oracle_selected = rb.deterministic_top_k(
            action_errors, candidates.episode_indices, candidates.frame_indices
        )
        oracle_latency = (time.perf_counter() - oracle_started) * 1000.0
        oracle_metrics = metrics_for_selection(
            oracle_selected,
            candidates,
            queries,
            query_index,
            normalized_candidate_states,
            normalized_query_states,
            normalized_candidate_actions,
            normalized_query_actions,
            action_errors,
        )
        oracle_metrics["retrieval_latency_ms"] = oracle_latency
        rows.append(
            result_row(
                "oracle_action",
                candidates,
                queries,
                query_index,
                oracle_selected,
                oracle_metrics,
                extra={"reporting_only": True},
            )
        )
    return rows


def metrics_for_selection(
    selected: np.ndarray,
    candidates: CandidateData,
    queries: QueryData,
    query_index: int,
    normalized_candidate_states: np.ndarray,
    normalized_query_states: np.ndarray,
    normalized_candidate_actions: np.ndarray,
    normalized_query_actions: np.ndarray,
    action_errors: np.ndarray,
) -> dict[str, float | int]:
    return rb.compute_retrieval_metrics(
        selected_indices=selected,
        candidate_action_chunks=normalized_candidate_actions,
        query_action_chunk=normalized_query_actions[query_index],
        candidate_states=normalized_candidate_states,
        query_state=normalized_query_states[query_index],
        candidate_progress=candidates.progress,
        query_progress=float(queries.progress[query_index]),
        candidate_episode_indices=candidates.episode_indices,
        all_action_errors=action_errors,
    ).as_dict()


def result_row(
    method: str,
    candidates: CandidateData,
    queries: QueryData,
    query_index: int,
    selected: np.ndarray,
    metrics: Mapping[str, float | int],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int64)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "method": method,
        "task_id": candidates.task_id,
        "prompt": queries.prompts[query_index],
        "query_episode_index": int(queries.episode_indices[query_index]),
        "query_frame_index": int(queries.frame_indices[query_index]),
        "query_progress": float(queries.progress[query_index]),
        "retrieved_episode_indices": [int(value) for value in candidates.episode_indices[selected]],
        "retrieved_frame_indices": [int(value) for value in candidates.frame_indices[selected]],
        **metrics,
        **(extra or {}),
    }


def add_method_ranks(summary: list[dict[str, object]]) -> None:
    ordered = sorted(summary, key=lambda row: (float(row["best_of_4_action_rmse"]), str(row["method"])))
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank


def pairwise_rows(per_task: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    values: dict[str, dict[int, float]] = {}
    for row in per_task:
        values.setdefault(str(row["method"]), {})[int(row["task_id"])] = float(row["best_of_4_action_rmse"])
    baseline = values["dino_top"]
    rows = []
    for method in METHODS:
        comparison = values[method]
        delta, low, high = rb.paired_task_bootstrap(baseline, comparison, samples=10_000, seed=0)
        wins = sum(comparison[task] < baseline[task] for task in baseline)
        ties = sum(comparison[task] == baseline[task] for task in baseline)
        losses = sum(comparison[task] > baseline[task] for task in baseline)
        rows.append(
            {
                "method": method,
                "baseline": "dino_top",
                "mean_delta_best_of_4_action_rmse": delta,
                "ci95_low": low,
                "ci95_high": high,
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        )
    return rows


def verify_smoke_determinism(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> None:
    calibration_tasks = [int(task) for task in calibration["calibration_tasks"]]
    if not calibration_tasks or not set(calibration_tasks).issubset(CALIBRATION_TASKS):
        raise AssertionError(f"Smoke calibration escaped tasks 0-19: {calibration_tasks}")
    rankings = [
        (
            row["method"],
            row["task_id"],
            row["query_episode_index"],
            row["query_frame_index"],
            tuple(row["retrieved_episode_indices"]),
            tuple(row["retrieved_frame_indices"]),
        )
        for row in rows
        if row["method"] != "random"
    ]
    first_hash = hashlib.sha256(repr(rankings).encode()).hexdigest()
    second_hash = hashlib.sha256(repr(list(rankings)).encode()).hexdigest()
    if first_hash != second_hash:
        raise AssertionError("Smoke ranking hashes are not deterministic.")
    payload = {
        "status": "passed",
        "tasks": list(SMOKE_TASKS),
        "queries": len(rows) // len(METHODS),
        "methods": list(METHODS),
        "deterministic_ranking_hash": first_hash,
        "candidate_query_overlap": False,
        "pi0_actions_supplied": False,
        "calibration_tasks": calibration_tasks,
    }
    atomic_write_text(output_root / "smoke" / "validation.json", json.dumps(payload, indent=2))


def write_manifest(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
) -> None:
    unique_queries = {
        (int(row["task_id"]), int(row["query_episode_index"]), int(row["query_frame_index"])) for row in rows
    }
    candidate_frames = 0
    for task_id in SOLVABLE_TASKS:
        source = source_for_task(task_id, "full")
        candidate_frames += len(load_candidates(source, task_id).states)
    sources = {
        (source_for_task(task_id, "full").repo_id, source_for_task(task_id, "full").retrieval_dir)
        for task_id in SOLVABLE_TASKS
    }
    source_identities = [
        source_metadata_identity(SourceSpec(repo_id=repo_id, retrieval_dir=retrieval_dir))
        for repo_id, retrieval_dir in sorted(sources, key=lambda item: item[0])
    ]
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "task_ids": list(SOLVABLE_TASKS),
        "same_task_retrieval_only": True,
        "candidate_episodes_per_task": 20,
        "candidate_episodes": len(SOLVABLE_TASKS) * 20,
        "candidate_frames": candidate_frames,
        "heldout_episodes_per_task": 30,
        "query_frames_per_episode": 10,
        "queries": len(unique_queries),
        "action_horizon": 10,
        "top_k": 4,
        "methods": list(METHODS),
        "normal_config": args.config_name,
        "normal_checkpoint": str(args.checkpoint_dir),
        "checkpoint_identity": checkpoint_identity(args.checkpoint_dir),
        "pi0_representation": {
            "description": (
                "Output projection of block 1 self-attention at the last valid prefix token, "
                "before residual addition, feed-forward processing, and later transformer blocks."
            ),
            "token_position": "last_valid_prefix_token",
            "layer": 1,
            "sublayer": "self_attention_output_projection",
            "l2_normalized": True,
            "expert_actions_supplied": False,
            "ricl_inputs_supplied": False,
        },
        "retrieval_dirs": sorted(
            {str(source_for_task(task_id, "full").retrieval_dir.resolve()) for task_id in SOLVABLE_TASKS}
        ),
        "datasets": sorted({source_for_task(task_id, "full").repo_id for task_id in SOLVABLE_TASKS}),
        "source_metadata_identities": source_identities,
        "calibration": calibration,
        "outputs": {
            "root": str(args.output_root),
            "queries_jsonl": str(args.output_root / "full" / "queries.jsonl"),
            "summary_csv": str(args.output_root / "full" / "summary.csv"),
            "per_task_csv": str(args.output_root / "full" / "per_task.csv"),
            "pairwise_csv": str(args.output_root / "full" / "pairwise.csv"),
            "xlsx": str(Path("data/libero/reports/libero90_solvable25_offline_v1.xlsx").resolve()),
            "log": str(Path("data/libero/logs/libero90_solvable25_offline_v1.log").resolve()),
        },
        "feature_cache_bytes": benchmark_cache_bytes(args.output_root),
        "summary": list(summary),
        "pairwise": list(pairwise),
        "cuda": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "packages": package_versions(("jax", "numpy", "torch", "transformers", "lerobot")),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "git_diff": command_output(["git", "diff", "--", "src/openpi", "scripts"]),
        "command": (
            "JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 MODE=all "
            "scripts/run_libero90_retrieval_benchmark.sh"
        ),
    }
    if manifest["candidate_frames"] != EXPECTED_CANDIDATE_FRAMES or manifest["queries"] != EXPECTED_FULL_QUERIES:
        raise ValueError(
            f"Full acceptance counts failed: candidates={manifest['candidate_frames']}, queries={manifest['queries']}."
        )
    manifest_path = Path("data/libero/reports/libero90_solvable25_offline_v1.manifest.json").resolve()
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    atomic_write_text(args.output_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def feature_path(output_root: Path, population: str, method: str, task_id: int) -> Path:
    return output_root / "cache" / population / method / f"task_{task_id:02d}.npy"


def cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def cache_valid(path: Path, expected_signature: str, *, overwrite: bool) -> bool:
    metadata_path = cache_metadata_path(path)
    if overwrite:
        return False
    if not path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError(f"Stale cache version at {path}; use --overwrite-cache.")
    if metadata.get("signature") != expected_signature:
        raise ValueError(f"Cache source/checkpoint signature mismatch at {path}; use --overwrite-cache.")
    return True


def create_memmap(path: Path, shape: tuple[int, int]) -> tuple[np.memmap, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    memmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    return memmap, temporary


def finish_memmap(
    memmap: np.memmap,
    temporary: Path,
    path: Path,
    expected_signature: str,
) -> None:
    memmap.flush()
    del memmap
    temporary.replace(path)
    write_cache_metadata(path, expected_signature)


def write_array_cache(path: Path, values: np.ndarray, expected_signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.save(output, np.asarray(values, dtype=np.float32))
    temporary.replace(path)
    write_cache_metadata(path, expected_signature)


def write_cache_metadata(path: Path, expected_signature: str) -> None:
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "signature": expected_signature,
        "shape": list(np.load(path, mmap_mode="r").shape),
        "dtype": str(np.load(path, mmap_mode="r").dtype),
        "bytes": path.stat().st_size,
    }
    atomic_write_text(cache_metadata_path(path), json.dumps(metadata, indent=2, sort_keys=True))


def load_feature(path: Path, expected_signature: str) -> np.ndarray:
    if not cache_valid(path, expected_signature, overwrite=False):
        raise FileNotFoundError(f"Required feature cache is missing: {path}")
    return np.load(path, mmap_mode="r")


def squared_l2_distances(candidate_features: np.ndarray, query_feature: np.ndarray) -> np.ndarray:
    candidates = np.asarray(candidate_features, dtype=np.float32)
    query = np.asarray(query_feature, dtype=np.float32)
    distances = np.einsum("ij,ij->i", candidates, candidates)
    distances += float(np.dot(query, query))
    distances -= 2.0 * np.asarray(candidates @ query)
    return np.maximum(distances, 0.0)


def load_action_stats(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads((checkpoint_dir / "assets" / "libero" / "norm_stats.json").read_text())["norm_stats"]["actions"]
    return np.asarray(stats["q01"], dtype=np.float32), np.asarray(stats["q99"], dtype=np.float32)


def load_state_stats(checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = json.loads((checkpoint_dir / "assets" / "libero" / "norm_stats.json").read_text())["norm_stats"]["state"]
    return np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)


def order_hash(episode_indices: np.ndarray, frame_indices: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(episode_indices, dtype=np.int32).tobytes())
    digest.update(np.asarray(frame_indices, dtype=np.int32).tobytes())
    return digest.hexdigest()


def feature_signature(
    frame_order_hash: str,
    method: str,
    source: SourceSpec,
    handle: DatasetHandle,
    *,
    checkpoint_dir: Path | None = None,
) -> str:
    dataset_metadata_paths = (
        handle.dataset.root / "meta" / "info.json",
        handle.dataset.root / "meta" / "episodes.jsonl",
        handle.dataset.root / "meta" / "tasks.jsonl",
    )
    retrieval_metadata = json.loads((source.retrieval_dir / "retrieval_metadata.json").read_text())
    retrieval_policy_keys = (
        "repo_id",
        "retrieval_key",
        "backend",
        "metric",
        "embedding_type",
        "max_episodes_per_task",
        "episode_selection",
        "seed",
    )
    payload: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "frame_order_hash": frame_order_hash,
        "method": method,
        "repo_id": source.repo_id,
        "retrieval_dir": str(source.retrieval_dir.resolve()),
        "retrieval_policy": {key: retrieval_metadata.get(key) for key in retrieval_policy_keys},
        "dataset_metadata_hashes": {str(path): sha256_file(path) for path in dataset_metadata_paths},
    }
    if checkpoint_dir is not None:
        payload["config_name"] = "pi0_fast_libero90_normal"
        payload["checkpoint_identity"] = checkpoint_identity(checkpoint_dir)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def checkpoint_identity(checkpoint_dir: Path) -> dict[str, str]:
    paths = (
        checkpoint_dir / "_CHECKPOINT_METADATA",
        checkpoint_dir / "params" / "_METADATA",
        checkpoint_dir / "assets" / "libero" / "norm_stats.json",
    )
    return {str(path.relative_to(checkpoint_dir)): sha256_file(path) for path in paths}


def source_metadata_identity(source: SourceSpec) -> dict[str, Any]:
    handle = DatasetHandle(source.repo_id)
    paths = (
        source.retrieval_dir / "retrieval_metadata.json",
        handle.dataset.root / "meta" / "info.json",
        handle.dataset.root / "meta" / "episodes.jsonl",
        handle.dataset.root / "meta" / "tasks.jsonl",
    )
    return {
        "repo_id": source.repo_id,
        "retrieval_dir": str(source.retrieval_dir.resolve()),
        "sha256": {str(path.resolve()): sha256_file(path) for path in paths},
    }


def benchmark_cache_bytes(output_root: Path) -> dict[str, int]:
    aliases = {
        "random": (),
        "state": (),
        "dino_top": ("dino_top",),
        "dino_top_wrist": ("dino_top", "dino_wrist"),
        "dino_top_state": ("dino_top",),
        "dino_top_wrist_state": ("dino_top", "dino_wrist"),
        "pi0_fast_first_fusion": ("pi0_fast_first_fusion",),
        "pi0_fast_first_fusion_no_prompt": ("pi0_fast_first_fusion_no_prompt",),
        "oracle_action": (),
    }
    sizes = {}
    for method, feature_names in aliases.items():
        sizes[method] = sum(
            path.stat().st_size
            for feature_name in feature_names
            for path in (output_root / "cache").glob(f"**/{feature_name}/task_*.npy")
        )
    return sizes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stack_trees(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import jax

    return jax.tree.map(lambda *values: np.stack(values), *items)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def to_int(value: Any) -> int:
    return int(np.asarray(to_numpy(value)).reshape(-1)[0])


def to_image_uint8(value: Any) -> np.ndarray:
    image = to_numpy(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open() as source:
        return sum(bool(line.strip()) for line in source)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    keys = list(rows[0])
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_output(command: Sequence[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout if result.returncode == 0 else result.stderr).strip()


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
