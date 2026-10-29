from __future__ import annotations

from collections.abc import Mapping
import json
import pathlib
import shutil
from typing import Any, Literal

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import tqdm
import tyro

from openpi.policies.utils import embed_with_batches
from openpi.policies.utils import get_embed_dim
from openpi.policies.utils import load_dinov2
import openpi.shared.normalize as normalize
from openpi.training.ricl_libero_dataset import DEFAULT_INDICES_FILE
from openpi.training.ricl_libero_dataset import DEFAULT_RETRIEVAL_METADATA_FILE
from openpi.training.ricl_libero_dataset import build_libero_embedding_retrieval_indices
from openpi.training.ricl_libero_dataset import build_libero_state_retrieval_indices

RetrievalKey = Literal["image", "wrist_image", "both", "state"]
EmbeddingType = Literal["CLS", "AVG", "16PATCHES", "64PATCHES"]
EpisodeSelection = Literal["seeded", "first"]
ArtifactRole = Literal["train", "retrieval"]


def _local_lerobot_dataset_exists(repo_id: str, dataset_root: pathlib.Path | None = None) -> bool:
    root = pathlib.Path(dataset_root) if dataset_root is not None else pathlib.Path(lerobot_dataset.LEROBOT_HOME)
    return (root / repo_id / "meta" / "info.json").exists()


def _get_field(row: MappingLike, *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"Could not find any of {names} in row keys: {list(row.keys())}")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _to_int(value: Any) -> int:
    return int(np.asarray(_to_numpy(value)).item())


def _to_image_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image


def _prompt_for_task(tasks: Any, task_index: int) -> str:
    if isinstance(tasks, Mapping):
        prompt = tasks.get(task_index)
        if prompt is None:
            prompt = tasks.get(str(task_index))
    else:
        try:
            prompt = tasks[task_index]
        except (IndexError, KeyError, TypeError):
            prompt = None

    if isinstance(prompt, Mapping):
        prompt = prompt.get("task") or prompt.get("name") or prompt.get("text")
    return f"task_{task_index}" if prompt is None else str(prompt)


def _flush_episode(
    episode_rows: list[dict[str, Any]],
    *,
    episodes_dir: pathlib.Path,
    state_stats: normalize.RunningStats,
    action_stats: normalize.RunningStats,
    retrieval_key: RetrievalKey,
    embedding_type: EmbeddingType,
    embedding_batch_size: int,
    overwrite_embeddings: bool,
    dinov2: Any | None,
) -> tuple[int, dict[str, Any]]:
    episode_rows = sorted(episode_rows, key=lambda row: row["frame_index"])
    episode_index = int(episode_rows[0]["episode_index"])
    task_index = int(episode_rows[0]["task_index"])
    prompt = str(episode_rows[0]["prompt"])

    images = np.stack([row["image"] for row in episode_rows], axis=0)
    wrist_images = np.stack([row["wrist_image"] for row in episode_rows], axis=0)
    states = np.stack([row["state"] for row in episode_rows], axis=0).astype(np.float32)
    actions = np.stack([row["actions"] for row in episode_rows], axis=0).astype(np.float32)
    frame_indices = np.asarray([row["frame_index"] for row in episode_rows], dtype=np.int32)

    if states.shape[-1] != 8:
        raise ValueError(f"Expected LIBERO state dim 8, got {states.shape}")
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected LIBERO action dim 7, got {actions.shape}")

    state_stats.update(states)
    action_stats.update(actions)

    episodes_dir.mkdir(parents=True, exist_ok=True)
    episode_path = episodes_dir / f"episode_{episode_index}.npz"
    episode_data = {
        "image": images,
        "wrist_image": wrist_images,
        "state": states,
        "actions": actions,
        "frame_indices": frame_indices,
        "task_index": np.asarray(task_index, dtype=np.int32),
        "prompt": np.asarray(prompt),
    }
    summary: dict[str, Any] = {
        "task_index": task_index,
        "state": states,
        "frame_indices": frame_indices,
    }

    if retrieval_key != "state":
        if dinov2 is None:
            raise ValueError("DINOv2 model must be loaded for image-embedding LIBERO retrieval.")
        for image_key in _image_keys_for_retrieval_key(retrieval_key):
            embedding_key = f"{image_key}_embeddings"
            episode_data[embedding_key] = _get_or_compute_embeddings(
                episode_path=episode_path,
                embedding_key=embedding_key,
                images=episode_data[image_key],
                dinov2=dinov2,
                embedding_type=embedding_type,
                embedding_batch_size=embedding_batch_size,
                overwrite_embeddings=overwrite_embeddings,
            )
        summary["embeddings"] = _combine_embeddings_for_retrieval_key(episode_data, retrieval_key)

    if not _episode_cache_ready(episode_path, episode_data):
        np.savez_compressed(
            episode_path,
            **episode_data,
        )

    return episode_index, summary


def _episode_cache_ready(episode_path: pathlib.Path, episode_data: Mapping[str, Any]) -> bool:
    identity_keys = {"frame_indices", "task_index", "prompt"}
    if not episode_path.exists():
        return False
    try:
        with np.load(episode_path, allow_pickle=True) as episode:
            for key, value in episode_data.items():
                if key not in episode:
                    return False
                if np.asarray(episode[key]).shape != np.asarray(value).shape:
                    return False
                if key in identity_keys and not np.array_equal(np.asarray(episode[key]), np.asarray(value)):
                    return False
    except Exception:
        return False
    return True


def _get_or_compute_embeddings(
    *,
    episode_path: pathlib.Path,
    embedding_key: str,
    images: np.ndarray,
    dinov2: Any,
    embedding_type: EmbeddingType,
    embedding_batch_size: int,
    overwrite_embeddings: bool,
) -> np.ndarray:
    expected_shape = (len(images), get_embed_dim(embedding_type))
    if episode_path.exists() and not overwrite_embeddings:
        with np.load(episode_path, allow_pickle=True) as episode:
            if embedding_key in episode and episode[embedding_key].shape == expected_shape:
                return episode[embedding_key].astype(np.float32)

    embeddings = embed_with_batches(
        images,
        dinov2,
        batch_size=embedding_batch_size,
        embedding_type=embedding_type,
    ).astype(np.float32)
    if embeddings.shape != expected_shape:
        raise ValueError(f"Expected {embedding_key} shape {expected_shape}, got {embeddings.shape}.")
    return embeddings


def _image_keys_for_retrieval_key(retrieval_key: RetrievalKey) -> tuple[str, ...]:
    if retrieval_key == "image":
        return ("image",)
    if retrieval_key == "wrist_image":
        return ("wrist_image",)
    if retrieval_key == "both":
        return ("image", "wrist_image")
    return ()


def _combine_embeddings_for_retrieval_key(episode_data: Mapping[str, Any], retrieval_key: RetrievalKey) -> np.ndarray:
    embedding_keys = [f"{image_key}_embeddings" for image_key in _image_keys_for_retrieval_key(retrieval_key)]
    embeddings = [np.asarray(episode_data[embedding_key], dtype=np.float32) for embedding_key in embedding_keys]
    if len(embeddings) == 1:
        return embeddings[0]
    return np.concatenate(embeddings, axis=1)


def _save_ricl_norm_stats(
    output_dir: pathlib.Path,
    *,
    state_stats: normalize.NormStats,
    action_stats: normalize.NormStats,
    num_retrieved_observations: int,
) -> None:
    norm_stats = {}
    for i in range(num_retrieved_observations):
        norm_stats[f"retrieved_{i}_state"] = state_stats
        norm_stats[f"retrieved_{i}_actions"] = action_stats
    norm_stats["query_state"] = state_stats
    norm_stats["query_actions"] = action_stats
    normalize.save(output_dir, norm_stats)


def _save_max_distance(path: pathlib.Path, max_distance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"distances": {"max": max(float(max_distance), 1e-6)}}, indent=2))


def _read_max_distance(path: pathlib.Path) -> float:
    with path.open() as f:
        return max(float(json.load(f)["distances"]["max"]), 1e-6)


def _save_retrieval_metadata(
    *,
    preprocessing_dir: pathlib.Path,
    assets_dir: pathlib.Path,
    retrieval_dir: pathlib.Path,
    retrieval_key: RetrievalKey,
    embedding_type: EmbeddingType,
    num_retrieved_observations: int,
    artifact_role: ArtifactRole = "train",
    repo_id: str | None = None,
    task_id_field: str | None = None,
    task_indices: tuple[int, ...] | None = None,
    max_episodes_per_task: int | None = None,
    episode_selection: EpisodeSelection | None = None,
    seed: int | None = None,
    max_distance_file: pathlib.Path | None = None,
    output_dirs: tuple[pathlib.Path, ...] | None = None,
) -> None:
    backend = "state" if retrieval_key == "state" else "dinov2"
    if max_distance_file is None:
        max_distance_file = assets_dir / "max_distance.json"
    metadata = {
        "retrieval_key": retrieval_key,
        "backend": backend,
        "metric": "l2",
        "embedding_type": None if backend == "state" else embedding_type,
        "embedding_keys": [f"{image_key}_embeddings" for image_key in _image_keys_for_retrieval_key(retrieval_key)],
        "num_retrieved_observations": num_retrieved_observations,
        "index_file": DEFAULT_INDICES_FILE,
        "max_distance_file": str(max_distance_file),
        "artifact_role": artifact_role,
    }
    if repo_id is not None:
        metadata["repo_id"] = repo_id
    if task_id_field is not None:
        metadata["task_id_field"] = task_id_field
    if task_indices is not None:
        metadata["task_indices"] = list(task_indices)
    if max_episodes_per_task is not None:
        metadata["max_episodes_per_task"] = max_episodes_per_task
    if episode_selection is not None:
        metadata["episode_selection"] = episode_selection
    if seed is not None:
        metadata["seed"] = seed

    if output_dirs is None:
        output_dirs = (preprocessing_dir, assets_dir, retrieval_dir)
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / DEFAULT_RETRIEVAL_METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def _stage_retrieval_episode(episode_path: pathlib.Path, retrieval_dir: pathlib.Path, task_index: int) -> None:
    task_dir = retrieval_dir / f"task_{task_index:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    target_path = task_dir / episode_path.name
    if target_path.exists():
        target_path.unlink()
    try:
        target_path.hardlink_to(episode_path)
    except OSError:
        shutil.copy2(episode_path, target_path)


def _require_cuda_dinov2(dinov2: Any) -> None:
    try:
        device = next(dinov2.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("DINOv2 has no parameters; cannot verify CUDA placement.") from exc

    if device.type != "cuda":
        raise RuntimeError(
            "DINOv2 image retrieval requires CUDA, but DINOv2 loaded on "
            f"{device}. Install a PyTorch CUDA 12.8+ build that supports RTX 5090 sm_120, "
            "then rerun this setup command."
        )


def _release_dinov2(dinov2: Any | None) -> None:
    if dinov2 is None:
        return
    del dinov2
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _select_episode_indices(
    rows: list[tuple[int, int, int, int, int]],
    *,
    max_episodes_per_task: int | None,
    episode_selection: EpisodeSelection,
    seed: int,
) -> set[int] | None:
    if max_episodes_per_task is None:
        return None
    if max_episodes_per_task <= 0:
        raise ValueError(f"max_episodes_per_task must be positive, got {max_episodes_per_task}.")

    first_seen_by_task: dict[int, dict[int, int]] = {}
    for episode_index, task_index, _frame_index, dataset_index, _prompt_task_index in rows:
        first_seen_by_task.setdefault(task_index, {}).setdefault(episode_index, dataset_index)

    selected_episode_indices: set[int] = set()
    for task_index in sorted(first_seen_by_task):
        first_seen = first_seen_by_task[task_index]
        episode_indices = sorted(first_seen, key=lambda ep_idx: (first_seen[ep_idx], ep_idx))
        if len(episode_indices) <= max_episodes_per_task:
            selected = episode_indices
        elif episode_selection == "first":
            selected = episode_indices[:max_episodes_per_task]
        elif episode_selection == "seeded":
            rng = np.random.default_rng(seed + task_index)
            selected = sorted(rng.choice(sorted(episode_indices), size=max_episodes_per_task, replace=False).tolist())
        else:
            raise ValueError(f"Unsupported episode_selection={episode_selection}.")
        selected_episode_indices.update(int(ep_idx) for ep_idx in selected)

    return selected_episode_indices


def main(
    *,
    repo_id: str = "physical-intelligence/libero",
    task_indices: tuple[int, ...] = tuple(range(10)),
    num_retrieved_observations: int = 4,
    retrieval_key: RetrievalKey = "image",
    embedding_type: EmbeddingType = "64PATCHES",
    embedding_batch_size: int = 256,
    overwrite_embeddings: bool = False,
    preprocessing_dir: pathlib.Path = pathlib.Path("ricl_libero_preprocessing"),
    retrieval_dir: pathlib.Path | None = None,
    assets_dir: pathlib.Path = pathlib.Path("assets/pi0_fast_libero_ricl/libero"),
    dataset_root: pathlib.Path | None = None,
    local_files_only: bool = False,
    task_id_field: str = "task_index",
    max_episodes_per_task: int | None = None,
    episode_selection: EpisodeSelection = "seeded",
    seed: int = 0,
    artifact_role: ArtifactRole = "train",
    max_distance_source: pathlib.Path | None = None,
) -> None:
    task_index_set = set(task_indices)
    if retrieval_dir is None:
        retrieval_dir = preprocessing_dir / "retrieval"
    local_files_only = local_files_only or _local_lerobot_dataset_exists(repo_id, dataset_root)
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
        repo_id,
        root=dataset_root,
        local_files_only=local_files_only,
    )
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=dataset_root,
        local_files_only=local_files_only,
    )
    tasks = dataset_meta.tasks
    if retrieval_key == "state":
        dinov2 = None
    else:
        dinov2 = load_dinov2()
        _require_cuda_dinov2(dinov2)

    scan_columns = set(dataset.hf_dataset.column_names)
    required_scan_columns = {task_id_field, "task_index", "episode_index", "frame_index"}
    missing_scan_columns = sorted(required_scan_columns - scan_columns)
    if missing_scan_columns:
        raise KeyError(f"LIBERO dataset is missing scan metadata columns: {missing_scan_columns}")
    scan_values = {name: dataset.hf_dataset[name] for name in sorted(required_scan_columns)}

    rows = []
    for dataset_index in tqdm.trange(len(dataset), desc="Scanning LIBERO rows"):
        row = {name: values[dataset_index] for name, values in scan_values.items()}
        task_index = _to_int(_get_field(row, task_id_field))
        if task_index not in task_index_set:
            continue
        prompt_task_index = _to_int(_get_field(row, "task_index"))
        rows.append(
            (
                _to_int(_get_field(row, "episode_index")),
                task_index,
                _to_int(_get_field(row, "frame_index")),
                dataset_index,
                prompt_task_index,
            )
        )

    if not rows:
        raise ValueError(f"No LIBERO rows found for task_indices={task_indices}.")
    selected_episode_indices = _select_episode_indices(
        rows,
        max_episodes_per_task=max_episodes_per_task,
        episode_selection=episode_selection,
        seed=seed,
    )
    if selected_episode_indices is not None:
        rows = [row for row in rows if row[0] in selected_episode_indices]
    rows.sort(key=lambda item: (item[0], item[1], item[2]))

    episodes_dir = preprocessing_dir / "episodes"
    episode_summaries = {}
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    current_episode_index = None
    episode_rows = []

    for episode_index, task_index, frame_index, dataset_index, prompt_task_index in tqdm.tqdm(
        rows, desc="Writing episode caches"
    ):
        if current_episode_index is not None and episode_index != current_episode_index:
            ep_idx, episode_summary = _flush_episode(
                episode_rows,
                episodes_dir=episodes_dir,
                state_stats=state_stats,
                action_stats=action_stats,
                retrieval_key=retrieval_key,
                embedding_type=embedding_type,
                embedding_batch_size=embedding_batch_size,
                overwrite_embeddings=overwrite_embeddings,
                dinov2=dinov2,
            )
            episode_summaries[ep_idx] = episode_summary
            episode_rows = []

        row = dataset[dataset_index]
        episode_rows.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "frame_index": frame_index,
                "image": _to_image_uint8(_get_field(row, "image", "observation/image", "observation.image")),
                "wrist_image": _to_image_uint8(
                    _get_field(row, "wrist_image", "observation/wrist_image", "observation.wrist_image")
                ),
                "state": _to_numpy(_get_field(row, "state", "observation/state", "observation.state")).astype(
                    np.float32
                ),
                "actions": _to_numpy(_get_field(row, "actions", "action")).astype(np.float32),
                "prompt": _prompt_for_task(tasks, prompt_task_index),
            }
        )
        current_episode_index = episode_index

    if episode_rows:
        ep_idx, episode_summary = _flush_episode(
            episode_rows,
            episodes_dir=episodes_dir,
            state_stats=state_stats,
            action_stats=action_stats,
            retrieval_key=retrieval_key,
            embedding_type=embedding_type,
            embedding_batch_size=embedding_batch_size,
            overwrite_embeddings=overwrite_embeddings,
            dinov2=dinov2,
        )
        episode_summaries[ep_idx] = episode_summary

    _release_dinov2(dinov2)
    dinov2 = None

    if artifact_role == "train":
        final_state_stats = state_stats.get_statistics()
        final_action_stats = action_stats.get_statistics()
        if retrieval_key == "state":
            query_indices, retrieved_indices, distances = build_libero_state_retrieval_indices(
                episode_summaries,
                num_retrieved_observations=num_retrieved_observations,
                state_mean=final_state_stats.mean,
                state_std=final_state_stats.std,
            )
        else:
            query_indices, retrieved_indices, distances = build_libero_embedding_retrieval_indices(
                episode_summaries,
                num_retrieved_observations=num_retrieved_observations,
                show_progress=True,
            )

        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            preprocessing_dir / DEFAULT_INDICES_FILE,
            query_indices=query_indices,
            retrieved_indices=retrieved_indices,
            distances=distances,
        )

        _save_ricl_norm_stats(
            assets_dir,
            state_stats=final_state_stats,
            action_stats=final_action_stats,
            num_retrieved_observations=num_retrieved_observations,
        )
        max_distance = float(np.max(distances))
        _save_max_distance(assets_dir / "max_distance.json", max_distance)
        _save_max_distance(preprocessing_dir / "max_distance.json", max_distance)
        _save_max_distance(retrieval_dir / "max_distance.json", max_distance)
        _save_retrieval_metadata(
            preprocessing_dir=preprocessing_dir,
            assets_dir=assets_dir,
            retrieval_dir=retrieval_dir,
            retrieval_key=retrieval_key,
            embedding_type=embedding_type,
            num_retrieved_observations=num_retrieved_observations,
            artifact_role=artifact_role,
            repo_id=repo_id,
            task_id_field=task_id_field,
            task_indices=task_indices,
            max_episodes_per_task=max_episodes_per_task,
            episode_selection=episode_selection,
            seed=seed,
        )
    else:
        if max_distance_source is None:
            raise ValueError("--max-distance-source is required when --artifact-role=retrieval.")
        max_distance = _read_max_distance(max_distance_source)
        _save_max_distance(retrieval_dir / "max_distance.json", max_distance)
        _save_retrieval_metadata(
            preprocessing_dir=preprocessing_dir,
            assets_dir=assets_dir,
            retrieval_dir=retrieval_dir,
            retrieval_key=retrieval_key,
            embedding_type=embedding_type,
            num_retrieved_observations=num_retrieved_observations,
            artifact_role=artifact_role,
            repo_id=repo_id,
            task_id_field=task_id_field,
            task_indices=task_indices,
            max_episodes_per_task=max_episodes_per_task,
            episode_selection=episode_selection,
            seed=seed,
            max_distance_file=retrieval_dir / "max_distance.json",
            output_dirs=(retrieval_dir,),
        )

    for episode_index, episode_summary in episode_summaries.items():
        _stage_retrieval_episode(
            episodes_dir / f"episode_{episode_index}.npz",
            retrieval_dir,
            int(episode_summary["task_index"]),
        )

    print(f"Wrote {len(episode_summaries)} episode caches to {episodes_dir}")
    if artifact_role == "train":
        print(f"Wrote {retrieval_key} retrieval indices to {preprocessing_dir / DEFAULT_INDICES_FILE}")
        print(f"Wrote norm stats and max distance to {assets_dir}")
    print(f"Wrote serving retrieval buffers to {retrieval_dir}")


MappingLike = Mapping[str, Any]


if __name__ == "__main__":
    tyro.cli(main)
