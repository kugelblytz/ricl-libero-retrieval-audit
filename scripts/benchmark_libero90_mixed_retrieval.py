#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import benchmark_libero90_retrieval as base
import numpy as np

from openpi.policies import libero_retrieval_benchmark as rb

BENCHMARK_VERSION = 1
DEFAULT_BASE_ROOT = Path("data/libero/retrieval_benchmark/libero90_solvable25_offline_v1")
DEFAULT_OUTPUT_ROOT = Path("data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1")
METHODS = (
    "random_global",
    "state_global",
    "dino_top_global",
    "dino_top_wrist_global",
    "dino_top_state_global",
    "dino_top_wrist_state_global",
    "pi0_fast_first_fusion_global",
    "pi0_fast_first_fusion_no_prompt_global",
    "pi0_fast_first_fusion_shuffled_prompt_global",
    "prompt_gated_dino_top",
    "task_oracle_gate_dino_top",
    "oracle_action_task",
)
GLOBAL_UNGATED_METHODS = (
    "random_global",
    "state_global",
    "dino_top_global",
    "dino_top_wrist_global",
    "dino_top_state_global",
    "dino_top_wrist_state_global",
    "pi0_fast_first_fusion_global",
    "pi0_fast_first_fusion_no_prompt_global",
    "pi0_fast_first_fusion_shuffled_prompt_global",
)
METRIC_NAMES = (
    "top1_task_accuracy",
    "task_recall_at4",
    "correct_task_fraction_at4",
    "task_aware_ndcg_at4",
    "useful_retrieval_at4",
    "conditional_best_action_rmse",
    "conditional_oracle_regret",
    "exact_oracle_top1",
    "any_oracle_overlap_at4",
    "oracle_overlap_count_at4",
    "retrieval_latency_ms",
)
SMOKE_QUERY_TASKS = (20, 30)
SMOKE_CANDIDATE_TASKS = (20, 21, 24, 30, 31)
FULL_TASKS = base.SOLVABLE_TASKS
EXPECTED_FULL_CANDIDATES = base.EXPECTED_CANDIDATE_FRAMES
EXPECTED_FULL_QUERIES = base.EXPECTED_FULL_QUERIES


@dataclasses.dataclass
class GlobalCandidates:
    task_ids: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    progress: np.ndarray
    states: np.ndarray
    action_chunks: np.ndarray
    prompts: list[str]
    task_slices: dict[int, slice]

    def task_indices(self, task_id: int) -> np.ndarray:
        section = self.task_slices[task_id]
        return np.arange(section.start, section.stop, dtype=np.int64)


@dataclasses.dataclass
class GlobalQueries:
    task_ids: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray
    progress: np.ndarray
    states: np.ndarray
    action_chunks: np.ndarray
    prompts: list[str]
    task_slices: dict[int, slice]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CUDA mixed-task LIBERO retrieval benchmark.")
    parser.add_argument(
        "stage",
        choices=("validate", "extract-shuffled-pi0", "build-distances", "score"),
    )
    parser.add_argument("--split", choices=("smoke", "full"), default="full")
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-name", default="pi0_fast_libero90_normal")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999"),
    )
    parser.add_argument("--pi0-batch-size", type=int, default=4)
    parser.add_argument("--cuda-query-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.base_root = args.base_root.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "validate":
        validate(args)
    elif args.stage == "extract-shuffled-pi0":
        require_working_cuda()
        extract_shuffled_pi0(args)
    elif args.stage == "build-distances":
        require_working_cuda()
        build_distances(args)
    elif args.stage == "score":
        score(args)


def tasks_for_split(split: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if split == "smoke":
        return SMOKE_QUERY_TASKS, SMOKE_CANDIDATE_TASKS
    return FULL_TASKS, FULL_TASKS


def validate(args: argparse.Namespace) -> None:
    if os.environ.get("JAX_PLATFORMS") != "cuda":
        raise RuntimeError("JAX_PLATFORMS must be set to cuda.")
    require_working_cuda()
    query_tasks, candidate_tasks = tasks_for_split(args.split)
    original_manifest = args.base_root / "manifest.json"
    if not original_manifest.is_file():
        raise FileNotFoundError(f"Completed task-specific manifest is missing: {original_manifest}")
    manifest = json.loads(original_manifest.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"Task-specific benchmark is not complete: {original_manifest}")
    if not args.checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Normal checkpoint is missing: {args.checkpoint_dir}")

    handles: dict[str, base.DatasetHandle] = {}
    candidates = load_global_candidates(candidate_tasks)
    queries = load_global_queries(query_tasks, args.split, handles=handles)
    validate_base_feature_caches(args, candidates, queries, candidate_tasks, query_tasks)
    expected_queries = 6 if args.split == "smoke" else EXPECTED_FULL_QUERIES
    expected_candidates = len(candidates.states)
    if args.split == "full" and expected_candidates != EXPECTED_FULL_CANDIDATES:
        raise ValueError(f"Expected {EXPECTED_FULL_CANDIDATES} full candidates, found {expected_candidates}.")
    if len(queries.states) != expected_queries:
        raise ValueError(f"Expected {expected_queries} {args.split} queries, found {len(queries.states)}.")
    print(
        json.dumps(
            {
                "status": "ok",
                "split": args.split,
                "query_tasks": list(query_tasks),
                "candidate_tasks": list(candidate_tasks),
                "queries": len(queries.states),
                "candidate_frames": len(candidates.states),
                "global_search": True,
                "cuda": cuda_identity(),
            },
            indent=2,
        )
    )


def require_working_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable.")
    try:
        left = torch.ones((8, 8), device="cuda")
        right = torch.ones((8, 8), device="cuda")
        value = float((left @ right)[0, 0].cpu())
    except Exception as exc:
        raise RuntimeError(f"CUDA is visible but a real matrix multiplication failed: {exc}") from exc
    if value != 8.0:
        raise RuntimeError(f"CUDA validation returned an unexpected value: {value}")


def cuda_identity() -> dict[str, Any]:
    import torch

    identity: dict[str, Any] = {
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        identity["device"] = torch.cuda.get_device_name(0)
        identity["capability"] = list(torch.cuda.get_device_capability(0))
        identity["memory_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    identity["nvidia_smi"] = base.command_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    driver_path = Path("/proc/driver/nvidia/version")
    identity["loaded_driver"] = driver_path.read_text().strip() if driver_path.is_file() else ""
    return identity


def load_global_candidates(task_ids: Sequence[int]) -> GlobalCandidates:
    task_arrays = []
    episode_arrays = []
    frame_arrays = []
    progress_arrays = []
    state_arrays = []
    action_arrays = []
    prompts: list[str] = []
    task_slices: dict[int, slice] = {}
    offset = 0
    for task_id in task_ids:
        source = base.source_for_task(task_id, "full")
        data = base.load_candidates(source, task_id)
        count = len(data.states)
        task_slices[task_id] = slice(offset, offset + count)
        task_arrays.append(np.full(count, task_id, dtype=np.int32))
        episode_arrays.append(data.episode_indices)
        frame_arrays.append(data.frame_indices)
        progress_arrays.append(data.progress)
        state_arrays.append(data.states)
        action_arrays.append(data.action_chunks)
        prompts.extend(data.prompts)
        offset += count
    return GlobalCandidates(
        task_ids=np.concatenate(task_arrays),
        episode_indices=np.concatenate(episode_arrays),
        frame_indices=np.concatenate(frame_arrays),
        progress=np.concatenate(progress_arrays),
        states=np.concatenate(state_arrays),
        action_chunks=np.concatenate(action_arrays),
        prompts=prompts,
        task_slices=task_slices,
    )


def load_global_queries(
    task_ids: Sequence[int],
    split: str,
    *,
    handles: dict[str, base.DatasetHandle] | None = None,
    include_images: bool = False,
) -> GlobalQueries:
    handles = {} if handles is None else handles
    task_arrays = []
    episode_arrays = []
    frame_arrays = []
    progress_arrays = []
    state_arrays = []
    action_arrays = []
    prompts: list[str] = []
    task_slices: dict[int, slice] = {}
    offset = 0
    for task_id in task_ids:
        source = base.source_for_task(task_id, split)
        handle = base.get_dataset_handle(handles, source.repo_id)
        data = base.load_queries(
            handle,
            source,
            task_id,
            split=split,
            include_images=include_images,
        )
        count = len(data.states)
        task_slices[task_id] = slice(offset, offset + count)
        task_arrays.append(np.full(count, task_id, dtype=np.int32))
        episode_arrays.append(data.episode_indices)
        frame_arrays.append(data.frame_indices)
        progress_arrays.append(data.progress)
        state_arrays.append(data.states)
        action_arrays.append(data.action_chunks)
        prompts.extend(data.prompts)
        offset += count
    return GlobalQueries(
        task_ids=np.concatenate(task_arrays),
        episode_indices=np.concatenate(episode_arrays),
        frame_indices=np.concatenate(frame_arrays),
        progress=np.concatenate(progress_arrays),
        states=np.concatenate(state_arrays),
        action_chunks=np.concatenate(action_arrays),
        prompts=prompts,
        task_slices=task_slices,
    )


def canonical_prompts(task_ids: Sequence[int]) -> dict[int, str]:
    result = {}
    for task_id in task_ids:
        prompts = set(base.load_candidates(base.source_for_task(task_id, "full"), task_id).prompts)
        if len(prompts) != 1:
            raise ValueError(f"Task {task_id} has {len(prompts)} candidate prompts; expected exactly one.")
        result[task_id] = prompts.pop()
    return result


def shuffled_prompt_map(task_ids: Sequence[int], prompts: Mapping[int, str]) -> dict[int, int]:
    ordered = list(task_ids)
    mapping = {}
    for index, task_id in enumerate(ordered):
        for offset in range(1, len(ordered) + 1):
            candidate = ordered[(index + offset) % len(ordered)]
            if prompts[candidate] != prompts[task_id]:
                mapping[task_id] = candidate
                break
        else:
            raise ValueError("Cannot create shuffled prompts because every task has the same prompt.")
    return mapping


def extract_shuffled_pi0(args: argparse.Namespace) -> None:
    query_tasks, candidate_tasks = tasks_for_split(args.split)
    prompts = canonical_prompts(candidate_tasks)
    query_prompts = canonical_prompts(query_tasks)
    mapping = shuffled_prompt_map(query_tasks, {**query_prompts, **prompts})
    embedder = base.Pi0FirstFusionEmbedder(args.config_name, args.checkpoint_dir)
    handles: dict[str, base.DatasetHandle] = {}
    for task_id in query_tasks:
        source = base.source_for_task(task_id, args.split)
        handle = base.get_dataset_handle(handles, source.repo_id)
        queries = base.load_queries(handle, source, task_id, split=args.split, include_images=True)
        wrong_task = mapping[task_id]
        wrong_prompt = prompts[wrong_task]
        wrong_prompts = [wrong_prompt] * len(queries.states)
        output = shuffled_feature_path(args.output_root, args.split, task_id)
        signature = shuffled_feature_signature(
            queries,
            source,
            handle,
            args.checkpoint_dir,
            task_id,
            wrong_task,
            wrong_prompt,
        )
        if args.resume and base.cache_valid(output, signature, overwrite=False):
            print(f"[mixed-pi0] task {task_id}: shuffled-prompt cache complete, skipping")
            continue
        assert queries.images is not None
        assert queries.wrist_images is not None
        if any(prompt == original for prompt, original in zip(wrong_prompts, queries.prompts, strict=True)):
            raise AssertionError(f"Task {task_id}: shuffled prompt unexpectedly matches the correct prompt.")
        print(f"[mixed-pi0] task {task_id}: extracting {len(queries.states)} queries with " f"task {wrong_task} prompt")
        full, _ = base.extract_pi0_batches(
            embedder,
            queries.images,
            queries.wrist_images,
            queries.states,
            wrong_prompts,
            args.pi0_batch_size,
        )
        base.write_array_cache(output, full, signature)
    payload = {
        "query_tasks": list(query_tasks),
        "mapping": {
            str(task): {
                "shuffled_prompt_task": mapping[task],
                "correct_prompt": query_prompts[task],
                "shuffled_prompt": prompts[mapping[task]],
            }
            for task in query_tasks
        },
    }
    base.atomic_write_text(
        args.output_root / f"shuffled_prompt_map_{args.split}.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )


def shuffled_feature_path(output_root: Path, split: str, task_id: int) -> Path:
    return (
        output_root / "cache" / f"query_{split}" / "pi0_fast_first_fusion_shuffled_prompt" / f"task_{task_id:02d}.npy"
    )


def shuffled_feature_signature(
    queries: base.QueryData,
    source: base.SourceSpec,
    handle: base.DatasetHandle,
    checkpoint_dir: Path,
    task_id: int,
    shuffled_task_id: int,
    shuffled_prompt: str,
) -> str:
    base_signature = base.feature_signature(
        queries.order_hash,
        "pi0_fast_first_fusion",
        source,
        handle,
        checkpoint_dir=checkpoint_dir,
    )
    payload = {
        "mixed_benchmark_version": BENCHMARK_VERSION,
        "base_signature": base_signature,
        "task_id": task_id,
        "shuffled_task_id": shuffled_task_id,
        "shuffled_prompt": shuffled_prompt,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_base_feature_caches(
    args: argparse.Namespace,
    candidates: GlobalCandidates,
    queries: GlobalQueries,
    candidate_tasks: Sequence[int],
    query_tasks: Sequence[int],
) -> None:
    handles: dict[str, base.DatasetHandle] = {}
    for task_id in candidate_tasks:
        source = base.source_for_task(task_id, "full")
        handle = base.get_dataset_handle(handles, source.repo_id)
        section = candidates.task_slices[task_id]
        task_candidates = base.load_candidates(source, task_id)
        if section.stop - section.start != len(task_candidates.states):
            raise AssertionError("Global candidate ordering does not match task cache ordering.")
        for method in (
            "dino_wrist",
            "pi0_fast_first_fusion",
            "pi0_fast_first_fusion_no_prompt",
        ):
            path = base.feature_path(args.base_root, "candidate", method, task_id)
            signature = base.feature_signature(
                task_candidates.order_hash,
                method,
                source,
                handle,
                checkpoint_dir=args.checkpoint_dir if method.startswith("pi0") else None,
            )
            base.load_feature(path, signature)
    for task_id in query_tasks:
        source = base.source_for_task(task_id, args.split)
        handle = base.get_dataset_handle(handles, source.repo_id)
        task_queries = base.load_queries(handle, source, task_id, split=args.split, include_images=False)
        section = queries.task_slices[task_id]
        if section.stop - section.start != len(task_queries.states):
            raise AssertionError("Global query ordering does not match task cache ordering.")
        for method in (
            "dino_top",
            "dino_wrist",
            "pi0_fast_first_fusion",
            "pi0_fast_first_fusion_no_prompt",
        ):
            path = base.feature_path(args.base_root, f"query_{args.split}", method, task_id)
            signature = base.feature_signature(
                task_queries.order_hash,
                method,
                source,
                handle,
                checkpoint_dir=args.checkpoint_dir if method.startswith("pi0") else None,
            )
            base.load_feature(path, signature)


def load_concatenated_query_feature(
    args: argparse.Namespace,
    method: str,
    task_ids: Sequence[int],
) -> np.ndarray:
    handles: dict[str, base.DatasetHandle] = {}
    parts = []
    for task_id in task_ids:
        source = base.source_for_task(task_id, args.split)
        handle = base.get_dataset_handle(handles, source.repo_id)
        queries = base.load_queries(handle, source, task_id, split=args.split, include_images=False)
        if method == "pi0_fast_first_fusion_shuffled_prompt":
            prompt_map = json.loads((args.output_root / f"shuffled_prompt_map_{args.split}.json").read_text())[
                "mapping"
            ]
            item = prompt_map[str(task_id)]
            path = shuffled_feature_path(args.output_root, args.split, task_id)
            signature = shuffled_feature_signature(
                queries,
                source,
                handle,
                args.checkpoint_dir,
                task_id,
                int(item["shuffled_prompt_task"]),
                str(item["shuffled_prompt"]),
            )
        else:
            path = base.feature_path(args.base_root, f"query_{args.split}", method, task_id)
            signature = base.feature_signature(
                queries.order_hash,
                method,
                source,
                handle,
                checkpoint_dir=args.checkpoint_dir if method.startswith("pi0") else None,
            )
        parts.append(np.asarray(base.load_feature(path, signature), dtype=np.float32))
    values = np.concatenate(parts)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Query feature {method} contains NaN or infinity.")
    return values


def load_concatenated_candidate_feature(
    args: argparse.Namespace,
    method: str,
    task_ids: Sequence[int],
) -> np.ndarray:
    handles: dict[str, base.DatasetHandle] = {}
    parts = []
    for task_id in task_ids:
        source = base.source_for_task(task_id, "full")
        handle = base.get_dataset_handle(handles, source.repo_id)
        candidates = base.load_candidates(source, task_id)
        path = base.feature_path(args.base_root, "candidate", method, task_id)
        signature = base.feature_signature(
            candidates.order_hash,
            method,
            source,
            handle,
            checkpoint_dir=args.checkpoint_dir if method.startswith("pi0") else None,
        )
        parts.append(np.asarray(base.load_feature(path, signature), dtype=np.float32))
    values = np.concatenate(parts)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Candidate feature {method} contains NaN or infinity.")
    return values


def build_distances(args: argparse.Namespace) -> None:
    query_tasks, candidate_tasks = tasks_for_split(args.split)
    candidates = load_global_candidates(candidate_tasks)
    queries = load_global_queries(query_tasks, args.split)
    shuffled_map_path = args.output_root / f"shuffled_prompt_map_{args.split}.json"
    if not shuffled_map_path.is_file():
        raise FileNotFoundError(f"Shuffled Pi0 extraction must run first: {shuffled_map_path}")
    build_dino_state_blocks(args, candidates, queries, query_tasks, candidate_tasks)
    build_pi0_selections(args, candidates, queries, query_tasks, candidate_tasks)


def build_dino_state_blocks(
    args: argparse.Namespace,
    candidates: GlobalCandidates,
    queries: GlobalQueries,
    query_tasks: Sequence[int],
    candidate_tasks: Sequence[int],
) -> None:
    import torch

    query_top_np = load_concatenated_query_feature(args, "dino_top", query_tasks)
    query_wrist_np = load_concatenated_query_feature(args, "dino_wrist", query_tasks)
    device = torch.device("cuda")
    query_top = torch.from_numpy(query_top_np).to(device)
    query_wrist = torch.from_numpy(query_wrist_np).to(device)
    del query_top_np, query_wrist_np
    state_mean, state_std = base.load_state_stats(args.checkpoint_dir)
    query_states_np = rb.z_normalize(queries.states, state_mean, state_std)
    query_states = torch.from_numpy(query_states_np).to(device)
    torch.cuda.synchronize()

    for task_id in candidate_tasks:
        block_dir = distance_block_dir(args.output_root, args.split, task_id)
        signature = distance_block_signature(args, query_tasks, candidate_tasks, task_id)
        if args.resume and distance_block_valid(block_dir, signature, len(queries.states)):
            print(f"[mixed-distance] candidate task {task_id}: complete, skipping")
            continue
        source = base.source_for_task(task_id, "full")
        task_candidates = base.load_candidates(source, task_id, include_top_embeddings=True)
        assert task_candidates.top_embeddings is not None
        wrist = load_concatenated_candidate_feature(args, "dino_wrist", (task_id,))
        candidate_states_np = rb.z_normalize(task_candidates.states, state_mean, state_std)
        count = len(task_candidates.states)
        temporary_dir = block_dir.with_name(block_dir.name + f".tmp.{os.getpid()}")
        temporary_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            name: np.lib.format.open_memmap(
                temporary_dir / f"{name}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(queries.states), count),
            )
            for name in ("dino_top", "dino_top_wrist", "state")
        }
        candidate_top = torch.from_numpy(task_candidates.top_embeddings).to(device)
        candidate_wrist = torch.from_numpy(wrist).to(device)
        candidate_states = torch.from_numpy(candidate_states_np).to(device)
        started = time.perf_counter()
        for start in range(0, len(queries.states), args.cuda_query_batch_size):
            end = min(start + args.cuda_query_batch_size, len(queries.states))
            top_sq = torch_squared_l2(query_top[start:end], candidate_top)
            wrist_sq = torch_squared_l2(query_wrist[start:end], candidate_wrist)
            state_sq = torch_squared_l2(query_states[start:end], candidate_states)
            outputs["dino_top"][start:end] = torch.sqrt(top_sq).cpu().numpy()
            outputs["dino_top_wrist"][start:end] = torch.sqrt(top_sq + wrist_sq).cpu().numpy()
            outputs["state"][start:end] = torch.sqrt(state_sq).cpu().numpy()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        for values in outputs.values():
            values.flush()
        del outputs, candidate_top, candidate_wrist, candidate_states, wrist, task_candidates
        metadata = {
            "benchmark_version": BENCHMARK_VERSION,
            "signature": signature,
            "candidate_task": task_id,
            "queries": len(queries.states),
            "candidates": count,
            "elapsed_seconds": elapsed,
            "cuda": cuda_identity(),
        }
        base.atomic_write_text(temporary_dir / "metadata.json", json.dumps(metadata, indent=2))
        if block_dir.exists():
            raise FileExistsError(f"Incomplete distance block already exists: {block_dir}")
        temporary_dir.replace(block_dir)
        print(
            f"[mixed-distance] candidate task {task_id}: {len(queries.states)}x{count} " f"distances in {elapsed:.1f}s"
        )
    del query_top, query_wrist, query_states
    torch.cuda.empty_cache()


def torch_squared_l2(query: Any, candidates: Any) -> Any:
    import torch

    query_norm = torch.sum(query * query, dim=1, keepdim=True)
    candidate_norm = torch.sum(candidates * candidates, dim=1).unsqueeze(0)
    return torch.clamp(query_norm + candidate_norm - 2.0 * (query @ candidates.T), min=0.0)


def distance_block_dir(output_root: Path, split: str, task_id: int) -> Path:
    return output_root / "distance_blocks" / split / f"candidate_task_{task_id:02d}"


def distance_block_signature(
    args: argparse.Namespace,
    query_tasks: Sequence[int],
    candidate_tasks: Sequence[int],
    candidate_task: int,
) -> str:
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "split": args.split,
        "query_tasks": list(query_tasks),
        "candidate_tasks": list(candidate_tasks),
        "candidate_task": candidate_task,
        "base_manifest_sha256": base.sha256_file(args.base_root / "manifest.json"),
        "checkpoint_identity": base.checkpoint_identity(args.checkpoint_dir),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def distance_block_valid(block_dir: Path, signature: str, query_count: int) -> bool:
    metadata_path = block_dir / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("signature") != signature or int(metadata.get("queries", -1)) != query_count:
        raise ValueError(f"Distance block identity mismatch: {block_dir}")
    count = int(metadata["candidates"])
    for name in ("dino_top", "dino_top_wrist", "state"):
        path = block_dir / f"{name}.npy"
        if not path.is_file() or np.load(path, mmap_mode="r").shape != (query_count, count):
            return False
    return True


def build_pi0_selections(
    args: argparse.Namespace,
    candidates: GlobalCandidates,
    queries: GlobalQueries,
    query_tasks: Sequence[int],
    candidate_tasks: Sequence[int],
) -> None:
    methods = {
        "pi0_fast_first_fusion_global": (
            "pi0_fast_first_fusion",
            "pi0_fast_first_fusion",
        ),
        "pi0_fast_first_fusion_no_prompt_global": (
            "pi0_fast_first_fusion_no_prompt",
            "pi0_fast_first_fusion_no_prompt",
        ),
        "pi0_fast_first_fusion_shuffled_prompt_global": (
            "pi0_fast_first_fusion",
            "pi0_fast_first_fusion_shuffled_prompt",
        ),
    }
    for output_method, (candidate_method, query_method) in methods.items():
        output = selection_path(args.output_root, args.split, output_method)
        signature = selection_signature(args, output_method, query_tasks, candidate_tasks)
        if args.resume and selection_valid(output, signature, len(queries.states)):
            print(f"[mixed-pi0-rank] {output_method}: complete, skipping")
            continue
        candidate_features = load_concatenated_candidate_feature(args, candidate_method, candidate_tasks)
        query_features = load_concatenated_query_feature(args, query_method, query_tasks)
        selected, selected_distances, elapsed = cuda_top_k(
            candidate_features,
            query_features,
            candidates,
            args.cuda_query_batch_size,
        )
        write_selection(
            output,
            selected,
            selected_distances,
            signature,
            elapsed,
        )
        print(f"[mixed-pi0-rank] {output_method}: {len(queries.states)} queries in {elapsed:.1f}s")


def cuda_top_k(
    candidate_features: np.ndarray,
    query_features: np.ndarray,
    candidates: GlobalCandidates,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch

    device = torch.device("cuda")
    candidate_tensor = torch.from_numpy(candidate_features).to(device)
    query_tensor = torch.from_numpy(query_features).to(device)
    selected = np.empty((len(query_features), rb.DEFAULT_TOP_K), dtype=np.int64)
    selected_distances = np.empty((len(query_features), rb.DEFAULT_TOP_K), dtype=np.float32)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(query_features), batch_size):
        end = min(start + batch_size, len(query_features))
        distances = torch.sqrt(torch_squared_l2(query_tensor[start:end], candidate_tensor)).cpu().numpy()
        for local_index, row in enumerate(distances):
            indices = rb.deterministic_global_top_k(
                row,
                candidates.task_ids,
                candidates.episode_indices,
                candidates.frame_indices,
            )
            selected[start + local_index] = indices
            selected_distances[start + local_index] = row[indices]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del candidate_tensor, query_tensor
    torch.cuda.empty_cache()
    return selected, selected_distances, elapsed


def selection_path(output_root: Path, split: str, method: str) -> Path:
    return output_root / "selections" / split / f"{method}.npz"


def selection_signature(
    args: argparse.Namespace,
    method: str,
    query_tasks: Sequence[int],
    candidate_tasks: Sequence[int],
) -> str:
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "method": method,
        "split": args.split,
        "query_tasks": list(query_tasks),
        "candidate_tasks": list(candidate_tasks),
        "base_manifest_sha256": base.sha256_file(args.base_root / "manifest.json"),
        "checkpoint_identity": base.checkpoint_identity(args.checkpoint_dir),
    }
    if "shuffled" in method:
        payload["shuffled_prompt_map"] = json.loads(
            (args.output_root / f"shuffled_prompt_map_{args.split}.json").read_text()
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_selection(
    path: Path,
    selected: np.ndarray,
    distances: np.ndarray,
    signature: str,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        np.savez(
            output,
            selected_indices=np.asarray(selected, dtype=np.int64),
            selected_distances=np.asarray(distances, dtype=np.float32),
        )
    temporary.replace(path)
    metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "signature": signature,
        "queries": len(selected),
        "elapsed_seconds": elapsed_seconds,
    }
    base.atomic_write_text(path.with_suffix(".json"), json.dumps(metadata, indent=2))


def selection_valid(path: Path, signature: str, query_count: int) -> bool:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("signature") != signature:
        raise ValueError(f"Selection identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as values:
        return values["selected_indices"].shape == (query_count, rb.DEFAULT_TOP_K)


def score(args: argparse.Namespace) -> None:
    query_tasks, candidate_tasks = tasks_for_split(args.split)
    candidates = load_global_candidates(candidate_tasks)
    queries = load_global_queries(query_tasks, args.split)
    calibration = json.loads((args.base_root / "calibration.json").read_text())
    betas = {family: float(item["beta"]) for family, item in calibration["selections"].items()}
    action_stats = base.load_action_stats(args.checkpoint_dir)
    normalized_candidate_actions = rb.quantile_normalize(candidates.action_chunks, *action_stats)
    normalized_query_actions = rb.quantile_normalize(queries.action_chunks, *action_stats)
    prompt_map = json.loads((args.output_root / f"shuffled_prompt_map_{args.split}.json").read_text())["mapping"]
    prompt_indices_by_prompt = {
        prompt: np.asarray(indices, dtype=np.int64) for prompt, indices in _group_indices(candidates.prompts).items()
    }
    task_specific_dino_selections = load_task_specific_dino_selections(
        args.base_root,
        args.split,
        candidates,
    )
    block_arrays = {
        name: [
            np.load(
                distance_block_dir(args.output_root, args.split, task_id) / f"{name}.npy",
                mmap_mode="r",
            )
            for task_id in candidate_tasks
        ]
        for name in ("dino_top", "dino_top_wrist", "state")
    }
    pi0_selections = {}
    for method in (
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
    ):
        path = selection_path(args.output_root, args.split, method)
        signature = selection_signature(args, method, query_tasks, candidate_tasks)
        if not selection_valid(path, signature, len(queries.states)):
            raise FileNotFoundError(f"Pi0 global selections are incomplete: {path}")
        values = np.load(path, allow_pickle=False)
        pi0_selections[method] = (
            values["selected_indices"],
            values["selected_distances"],
            float(json.loads(path.with_suffix(".json").read_text())["elapsed_seconds"]) / len(queries.states) * 1000.0,
        )

    score_dir = args.output_root / "scores" / args.split
    score_dir.mkdir(parents=True, exist_ok=True)
    expected_rows_per_task = (3 if args.split == "smoke" else 300) * len(METHODS)
    for task_id in query_tasks:
        output = score_dir / f"task_{task_id:02d}.jsonl"
        if args.resume and base.jsonl_row_count(output) == expected_rows_per_task:
            print(f"[mixed-score] task {task_id}: complete, skipping")
            continue
        task_rows = []
        section = queries.task_slices[task_id]
        for query_index in range(section.start, section.stop):
            top = np.concatenate([values[query_index] for values in block_arrays["dino_top"]])
            top_wrist = np.concatenate([values[query_index] for values in block_arrays["dino_top_wrist"]])
            state = np.concatenate([values[query_index] for values in block_arrays["state"]])
            distances = {
                "state_global": state,
                "dino_top_global": top,
                "dino_top_wrist_global": top_wrist,
                "dino_top_state_global": rb.hybrid_distances(top, state, betas["dino_top_state"]),
                "dino_top_wrist_state_global": rb.hybrid_distances(top_wrist, state, betas["dino_top_wrist_state"]),
            }
            same_task_indices = candidates.task_indices(task_id)
            action_errors = rb.action_rmse(
                normalized_candidate_actions[same_task_indices],
                normalized_query_actions[query_index],
            )
            sorted_action_errors = np.sort(action_errors)
            useful_action_error_threshold = float(np.quantile(sorted_action_errors, 0.1))
            oracle_local = rb.deterministic_top_k(
                action_errors,
                candidates.episode_indices[same_task_indices],
                candidates.frame_indices[same_task_indices],
            )
            oracle = same_task_indices[oracle_local]

            random_started = time.perf_counter()
            random_indices = rb.random_top_k(
                len(candidates.states),
                repeats=20,
                seed=(args.seed + task_id * 1_000_003 + int(queries.episode_indices[query_index]) * 1009 + query_index),
            )
            random_metrics = average_optional_metrics(
                [
                    mixed_metrics(
                        selected,
                        candidates,
                        queries,
                        query_index,
                        normalized_candidate_actions,
                        normalized_query_actions,
                        oracle,
                        same_task_indices,
                        action_errors,
                        sorted_action_errors,
                        useful_action_error_threshold,
                    )
                    for selected in random_indices
                ]
            )
            random_metrics["retrieval_latency_ms"] = (time.perf_counter() - random_started) * 1000.0 / 20.0
            task_rows.append(
                mixed_result_row(
                    "random_global",
                    random_indices[0],
                    candidates,
                    queries,
                    query_index,
                    random_metrics,
                    extra={"random_repeats": 20},
                )
            )

            for method, values in distances.items():
                started = time.perf_counter()
                selected = rb.deterministic_global_top_k(
                    values,
                    candidates.task_ids,
                    candidates.episode_indices,
                    candidates.frame_indices,
                )
                repeated = rb.deterministic_global_top_k(
                    values,
                    candidates.task_ids,
                    candidates.episode_indices,
                    candidates.frame_indices,
                )
                if not np.array_equal(selected, repeated):
                    raise AssertionError(f"{method} produced nondeterministic rankings.")
                metrics = mixed_metrics(
                    selected,
                    candidates,
                    queries,
                    query_index,
                    normalized_candidate_actions,
                    normalized_query_actions,
                    oracle,
                    same_task_indices,
                    action_errors,
                    sorted_action_errors,
                    useful_action_error_threshold,
                )
                metrics["retrieval_latency_ms"] = (time.perf_counter() - started) * 1000.0
                extra = {}
                if method == "dino_top_state_global":
                    extra["beta"] = betas["dino_top_state"]
                if method == "dino_top_wrist_state_global":
                    extra["beta"] = betas["dino_top_wrist_state"]
                task_rows.append(
                    mixed_result_row(
                        method,
                        selected,
                        candidates,
                        queries,
                        query_index,
                        metrics,
                        extra=extra,
                    )
                )

            for method, (indices, selected_distances, latency_ms) in pi0_selections.items():
                selected = np.asarray(indices[query_index], dtype=np.int64)
                metrics = mixed_metrics(
                    selected,
                    candidates,
                    queries,
                    query_index,
                    normalized_candidate_actions,
                    normalized_query_actions,
                    oracle,
                    same_task_indices,
                    action_errors,
                    sorted_action_errors,
                    useful_action_error_threshold,
                )
                metrics["retrieval_latency_ms"] = latency_ms
                extra: dict[str, Any] = {"top1_feature_distance": float(selected_distances[query_index][0])}
                if "shuffled_prompt" in method:
                    extra.update(prompt_map[str(task_id)])
                task_rows.append(
                    mixed_result_row(
                        method,
                        selected,
                        candidates,
                        queries,
                        query_index,
                        metrics,
                        extra=extra,
                    )
                )

            prompt_indices = prompt_indices_by_prompt[queries.prompts[query_index]]
            prompt_local = rb.deterministic_global_top_k(
                top[prompt_indices],
                candidates.task_ids[prompt_indices],
                candidates.episode_indices[prompt_indices],
                candidates.frame_indices[prompt_indices],
            )
            prompt_selected = prompt_indices[prompt_local]
            prompt_metrics = mixed_metrics(
                prompt_selected,
                candidates,
                queries,
                query_index,
                normalized_candidate_actions,
                normalized_query_actions,
                oracle,
                same_task_indices,
                action_errors,
                sorted_action_errors,
                useful_action_error_threshold,
            )
            prompt_metrics["retrieval_latency_ms"] = 0.0
            task_rows.append(
                mixed_result_row(
                    "prompt_gated_dino_top",
                    prompt_selected,
                    candidates,
                    queries,
                    query_index,
                    prompt_metrics,
                    extra={"eligible_candidate_count": len(prompt_indices)},
                )
            )

            query_key = (
                task_id,
                int(queries.episode_indices[query_index]),
                int(queries.frame_indices[query_index]),
            )
            task_selected = task_specific_dino_selections[query_key]
            task_metrics = mixed_metrics(
                task_selected,
                candidates,
                queries,
                query_index,
                normalized_candidate_actions,
                normalized_query_actions,
                oracle,
                same_task_indices,
                action_errors,
                sorted_action_errors,
                useful_action_error_threshold,
            )
            task_metrics["retrieval_latency_ms"] = 0.0
            task_rows.append(
                mixed_result_row(
                    "task_oracle_gate_dino_top",
                    task_selected,
                    candidates,
                    queries,
                    query_index,
                    task_metrics,
                    extra={"reporting_only": True},
                )
            )

            oracle_metrics = mixed_metrics(
                oracle,
                candidates,
                queries,
                query_index,
                normalized_candidate_actions,
                normalized_query_actions,
                oracle,
                same_task_indices,
                action_errors,
                sorted_action_errors,
                useful_action_error_threshold,
            )
            oracle_metrics["retrieval_latency_ms"] = 0.0
            task_rows.append(
                mixed_result_row(
                    "oracle_action_task",
                    oracle,
                    candidates,
                    queries,
                    query_index,
                    oracle_metrics,
                    extra={"reporting_only": True},
                )
            )
        if len(task_rows) != expected_rows_per_task:
            raise ValueError(f"Task {task_id}: expected {expected_rows_per_task} rows, got {len(task_rows)}.")
        base.atomic_write_jsonl(output, task_rows)
        print(f"[mixed-score] task {task_id}: wrote {len(task_rows)} rows")

    rows = []
    for task_id in query_tasks:
        rows.extend(base.read_jsonl(score_dir / f"task_{task_id:02d}.jsonl"))
    expected_queries = 6 if args.split == "smoke" else EXPECTED_FULL_QUERIES
    if len(rows) != expected_queries * len(METHODS):
        raise ValueError(f"Expected {expected_queries * len(METHODS)} rows, found {len(rows)}.")
    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_write_jsonl(output_dir / "queries.jsonl", rows)
    per_task, summary = aggregate_rows(rows)
    pairwise = mixed_pairwise_rows(per_task)
    confusion = confusion_rows(rows)
    base.write_csv(output_dir / "per_task.csv", per_task)
    base.write_csv(output_dir / "summary.csv", summary)
    base.write_csv(output_dir / "pairwise.csv", pairwise)
    base.write_csv(output_dir / "confusion.csv", confusion)
    if args.split == "smoke":
        verify_smoke(args, rows, candidates)
    else:
        write_manifest(args, rows, summary, pairwise, calibration, candidates, queries)
    print(f"[mixed-score] {args.split}: {expected_queries} queries scored")


def mixed_metrics(
    selected: np.ndarray,
    candidates: GlobalCandidates,
    queries: GlobalQueries,
    query_index: int,
    normalized_candidate_actions: np.ndarray,
    normalized_query_actions: np.ndarray,
    oracle: np.ndarray,
    same_task_indices: np.ndarray,
    same_task_action_errors: np.ndarray,
    sorted_same_task_action_errors: np.ndarray,
    useful_action_error_threshold: float,
) -> dict[str, float | None]:
    return rb.compute_mixed_task_metrics(
        selected_indices=selected,
        candidate_task_indices=candidates.task_ids,
        query_task_index=int(queries.task_ids[query_index]),
        candidate_action_chunks=normalized_candidate_actions,
        query_action_chunk=normalized_query_actions[query_index],
        oracle_indices=oracle,
        same_task_indices=same_task_indices,
        same_task_action_errors=same_task_action_errors,
        sorted_same_task_action_errors=sorted_same_task_action_errors,
        useful_action_error_threshold=useful_action_error_threshold,
    ).as_dict()


def _group_indices(values: Sequence[str]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        grouped[value].append(index)
    return grouped


def load_task_specific_dino_selections(
    base_root: Path,
    split: str,
    candidates: GlobalCandidates,
) -> dict[tuple[int, int, int], np.ndarray]:
    candidate_lookup = {
        (int(task), int(episode), int(frame)): index
        for index, (task, episode, frame) in enumerate(
            zip(
                candidates.task_ids,
                candidates.episode_indices,
                candidates.frame_indices,
                strict=True,
            )
        )
    }
    selections = {}
    for row in base.read_jsonl(base_root / split / "queries.jsonl"):
        if row["method"] != "dino_top":
            continue
        query_key = (
            int(row["task_id"]),
            int(row["query_episode_index"]),
            int(row["query_frame_index"]),
        )
        selections[query_key] = np.asarray(
            [
                candidate_lookup[(query_key[0], int(episode), int(frame))]
                for episode, frame in zip(
                    row["retrieved_episode_indices"],
                    row["retrieved_frame_indices"],
                    strict=True,
                )
            ],
            dtype=np.int64,
        )
    return selections


def average_optional_metrics(rows: Sequence[Mapping[str, float | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def mixed_result_row(
    method: str,
    selected: np.ndarray,
    candidates: GlobalCandidates,
    queries: GlobalQueries,
    query_index: int,
    metrics: Mapping[str, float | None],
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=np.int64)
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "method": method,
        "task_id": int(queries.task_ids[query_index]),
        "prompt": queries.prompts[query_index],
        "query_episode_index": int(queries.episode_indices[query_index]),
        "query_frame_index": int(queries.frame_indices[query_index]),
        "query_progress": float(queries.progress[query_index]),
        "retrieved_task_ids": [int(value) for value in candidates.task_ids[selected]],
        "retrieved_episode_indices": [int(value) for value in candidates.episode_indices[selected]],
        "retrieved_frame_indices": [int(value) for value in candidates.frame_indices[selected]],
        **metrics,
        **(extra or {}),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["task_id"]))].append(row)
    per_task = []
    for (method, task_id), task_rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "query_count": len(task_rows),
            "conditional_query_count": sum(row["conditional_best_action_rmse"] is not None for row in task_rows),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in task_rows if row.get(metric) is not None]
            item[metric] = float(np.mean(values)) if values else None
        per_task.append(item)

    method_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in per_task:
        method_rows[str(row["method"])].append(row)
    summary = []
    for method, task_rows in sorted(method_rows.items()):
        item = {
            "method": method,
            "task_count": len(task_rows),
            "query_count": sum(int(row["query_count"]) for row in task_rows),
            "conditional_query_count": sum(int(row["conditional_query_count"]) for row in task_rows),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in task_rows if row.get(metric) is not None]
            item[metric] = float(np.mean(values)) if values else None
        summary.append(item)
    ranked = sorted(
        summary,
        key=lambda row: (-float(row["task_aware_ndcg_at4"]), str(row["method"])),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return per_task, summary


def mixed_pairwise_rows(per_task: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for row in per_task:
        values[str(row["method"])][int(row["task_id"])] = float(row["task_aware_ndcg_at4"])
    baseline = values["dino_top_global"]
    output = []
    for method in METHODS:
        comparison = values[method]
        delta, low, high = rb.paired_task_bootstrap(
            baseline,
            comparison,
            samples=10_000,
            seed=0,
        )
        output.append(
            {
                "method": method,
                "baseline": "dino_top_global",
                "mean_delta_task_aware_ndcg_at4": delta,
                "ci95_low": low,
                "ci95_high": high,
                "wins": sum(comparison[task] > baseline[task] for task in baseline),
                "ties": sum(comparison[task] == baseline[task] for task in baseline),
                "losses": sum(comparison[task] < baseline[task] for task in baseline),
            }
        )
    return output


def confusion_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for row in rows:
        counts[
            (
                str(row["method"]),
                int(row["task_id"]),
                int(row["retrieved_task_ids"][0]),
            )
        ] += 1
    return [
        {
            "method": method,
            "query_task_id": query_task,
            "retrieved_top1_task_id": retrieved_task,
            "count": count,
        }
        for (method, query_task, retrieved_task), count in sorted(counts.items())
    ]


def verify_smoke(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    candidates: GlobalCandidates,
) -> None:
    grouped = {
        (
            row["method"],
            int(row["task_id"]),
            int(row["query_episode_index"]),
            int(row["query_frame_index"]),
        ): row
        for row in rows
    }
    if len(grouped) != len(rows):
        raise AssertionError("Smoke output contains duplicate method/query rows.")
    for row in rows:
        if len(row["retrieved_task_ids"]) != rb.DEFAULT_TOP_K:
            raise AssertionError("A smoke method did not return exactly four candidates.")
        if row["method"] == "task_oracle_gate_dino_top" and any(
            int(task) != int(row["task_id"]) for task in row["retrieved_task_ids"]
        ):
            raise AssertionError("Task-oracle gate returned a wrong-task candidate.")
        if row["method"] == "prompt_gated_dino_top":
            for task_id, episode_id, frame_id in zip(
                row["retrieved_task_ids"],
                row["retrieved_episode_indices"],
                row["retrieved_frame_indices"],
                strict=True,
            ):
                matches = np.flatnonzero(
                    (candidates.task_ids == int(task_id))
                    & (candidates.episode_indices == int(episode_id))
                    & (candidates.frame_indices == int(frame_id))
                )
                if len(matches) != 1 or candidates.prompts[int(matches[0])] != row["prompt"]:
                    raise AssertionError("Prompt gate returned a candidate with a different prompt.")
        if "shuffled_prompt" in row["method"] and row["shuffled_prompt"] == row["prompt"]:
            raise AssertionError("Shuffled-prompt Pi0 used the correct prompt.")

    original_rows = base.read_jsonl(args.base_root / "smoke" / "queries.jsonl")
    original = {
        (
            int(row["task_id"]),
            int(row["query_episode_index"]),
            int(row["query_frame_index"]),
        ): list(
            zip(
                row["retrieved_episode_indices"],
                row["retrieved_frame_indices"],
                strict=True,
            )
        )
        for row in original_rows
        if row["method"] == "dino_top"
    }
    for row in rows:
        if row["method"] != "task_oracle_gate_dino_top":
            continue
        key = (
            int(row["task_id"]),
            int(row["query_episode_index"]),
            int(row["query_frame_index"]),
        )
        selected = list(
            zip(
                row["retrieved_episode_indices"],
                row["retrieved_frame_indices"],
                strict=True,
            )
        )
        if selected != original[key]:
            raise AssertionError(f"Task-gated global DINO does not reproduce task-specific DINO for {key}.")

    rankings = [
        (
            row["method"],
            row["task_id"],
            row["query_episode_index"],
            row["query_frame_index"],
            tuple(row["retrieved_task_ids"]),
            tuple(row["retrieved_episode_indices"]),
            tuple(row["retrieved_frame_indices"]),
        )
        for row in rows
        if row["method"] != "random_global"
    ]
    payload = {
        "status": "passed",
        "query_tasks": list(SMOKE_QUERY_TASKS),
        "candidate_tasks": list(SMOKE_CANDIDATE_TASKS),
        "queries": len(rows) // len(METHODS),
        "methods": list(METHODS),
        "deterministic_ranking_hash": hashlib.sha256(repr(rankings).encode()).hexdigest(),
        "task_gate_matches_task_specific_dino": True,
        "shuffled_prompts_differ": True,
        "cuda": cuda_identity(),
    }
    base.atomic_write_text(
        args.output_root / "smoke" / "validation.json",
        json.dumps(payload, indent=2),
    )


def write_manifest(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    calibration: Mapping[str, Any],
    candidates: GlobalCandidates,
    queries: GlobalQueries,
) -> None:
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "complete",
        "task_ids": list(FULL_TASKS),
        "mixed_task_global_index": True,
        "task_id_available_to_global_retrievers": False,
        "candidate_tasks": len(candidates.task_slices),
        "candidate_episodes_per_task": 20,
        "candidate_episodes": len(candidates.task_slices) * 20,
        "candidate_frames": len(candidates.states),
        "query_tasks": len(queries.task_slices),
        "heldout_episodes_per_task": 30,
        "query_frames_per_episode": 10,
        "queries": len(queries.states),
        "top_k": rb.DEFAULT_TOP_K,
        "action_horizon": 10,
        "methods": list(METHODS),
        "primary_metric": "task_aware_ndcg_at4",
        "task_aware_relevance": (
            "Wrong-task candidates have zero relevance. Correct-task relevance is one minus "
            "the candidate's percentile rank among all same-task future-action RMSE values."
        ),
        "useful_retrieval_definition": (
            "At least one correct-task top-four candidate is in the best 10 percent of "
            "same-task future-action RMSE values."
        ),
        "normal_config": args.config_name,
        "normal_checkpoint": str(args.checkpoint_dir),
        "checkpoint_identity": base.checkpoint_identity(args.checkpoint_dir),
        "pi0_representation": json.loads((args.base_root / "manifest.json").read_text())["pi0_representation"],
        "shuffled_prompt_map": json.loads((args.output_root / "shuffled_prompt_map_full.json").read_text()),
        "hybrid_calibration": {
            "source": str(args.base_root / "calibration.json"),
            "tasks": calibration["calibration_tasks"],
            "selections": calibration["selections"],
            "note": "Weights calibrated only on tasks 0-19 and reused without benchmark-task tuning.",
        },
        "base_task_specific_benchmark": str(args.base_root),
        "base_manifest_sha256": base.sha256_file(args.base_root / "manifest.json"),
        "retrieval_dirs": sorted(
            {str(base.source_for_task(task_id, "full").retrieval_dir.resolve()) for task_id in FULL_TASKS}
        ),
        "datasets": sorted({base.source_for_task(task_id, "full").repo_id for task_id in FULL_TASKS}),
        "outputs": {
            "root": str(args.output_root),
            "queries_jsonl": str(args.output_root / "full" / "queries.jsonl"),
            "summary_csv": str(args.output_root / "full" / "summary.csv"),
            "per_task_csv": str(args.output_root / "full" / "per_task.csv"),
            "pairwise_csv": str(args.output_root / "full" / "pairwise.csv"),
            "confusion_csv": str(args.output_root / "full" / "confusion.csv"),
            "xlsx": str(Path("data/libero/reports/libero90_solvable25_mixed_global_v1.xlsx").resolve()),
            "plots": str(Path("data/libero/reports/libero90_solvable25_mixed_global_v1_plots").resolve()),
            "log": str(Path("data/libero/logs/libero90_solvable25_mixed_global_v1.log").resolve()),
        },
        "summary": list(summary),
        "pairwise": list(pairwise),
        "cuda": cuda_identity(),
        "packages": base.package_versions(("jax", "numpy", "torch", "matplotlib", "transformers", "lerobot")),
        "git_commit": base.command_output(["git", "rev-parse", "HEAD"]),
        "git_status": base.command_output(["git", "status", "--short"]),
        "git_diff": base.command_output(["git", "diff", "--", "src/openpi", "scripts"]),
        "command": (
            "JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 MODE=all "
            "scripts/run_libero90_mixed_retrieval_benchmark.sh"
        ),
    }
    if (
        manifest["candidate_frames"] != EXPECTED_FULL_CANDIDATES
        or manifest["queries"] != EXPECTED_FULL_QUERIES
        or len(rows) != EXPECTED_FULL_QUERIES * len(METHODS)
    ):
        raise ValueError("Full mixed-task acceptance counts failed.")
    report_manifest = Path("data/libero/reports/libero90_solvable25_mixed_global_v1.manifest.json").resolve()
    base.atomic_write_text(report_manifest, json.dumps(manifest, indent=2, sort_keys=True))
    base.atomic_write_text(
        args.output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )


if __name__ == "__main__":
    main()
