from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi.training.ricl_libero_dataset import DEFAULT_INDICES_FILE


def _parse_task_indices(value: str) -> tuple[int, ...]:
    if not value:
        raise argparse.ArgumentTypeError("Task index list cannot be empty.")
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Task indices must be comma-separated integers.") from exc


def _require_path(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _load_episode_metadata(episodes_dir: pathlib.Path) -> tuple[dict[int, int], dict[int, set[int]]]:
    episode_task = {}
    episode_frames = {}
    for path in sorted(episodes_dir.glob("episode_*.npz")):
        episode_index = int(path.stem.removeprefix("episode_"))
        with np.load(path, allow_pickle=True) as episode:
            episode_task[episode_index] = int(episode["task_index"])
            episode_frames[episode_index] = {int(frame_idx) for frame_idx in episode["frame_indices"]}

    if not episode_task:
        raise ValueError(f"No episode_*.npz files found in {episodes_dir}.")
    return episode_task, episode_frames


def _validate_index_frames(
    query_indices: np.ndarray,
    retrieved_indices: np.ndarray,
    *,
    episode_task: dict[int, int],
    episode_frames: dict[int, set[int]],
) -> None:
    for sample_idx, (query, retrieved) in enumerate(zip(query_indices, retrieved_indices, strict=True)):
        query_ep = int(query[0])
        query_frame = int(query[1])
        if query_ep not in episode_task:
            raise ValueError(f"Query sample {sample_idx} references missing episode {query_ep}.")
        if query_frame not in episode_frames[query_ep]:
            raise ValueError(f"Query sample {sample_idx} references missing frame {query_frame} in episode {query_ep}.")

        for retrieved_ep_raw, retrieved_frame_raw in retrieved:
            retrieved_ep = int(retrieved_ep_raw)
            retrieved_frame = int(retrieved_frame_raw)
            if retrieved_ep not in episode_task:
                raise ValueError(f"Sample {sample_idx} references missing retrieved episode {retrieved_ep}.")
            if retrieved_frame not in episode_frames[retrieved_ep]:
                raise ValueError(
                    f"Sample {sample_idx} references missing retrieved frame {retrieved_frame} "
                    f"in episode {retrieved_ep}."
                )
            if retrieved_ep == query_ep:
                raise ValueError(f"Sample {sample_idx} retrieves from its own query episode {query_ep}.")
            if episode_task[retrieved_ep] != episode_task[query_ep]:
                raise ValueError(
                    f"Sample {sample_idx} retrieved task {episode_task[retrieved_ep]} "
                    f"for query task {episode_task[query_ep]}."
                )


def _validate_norm_stats(norm_stats_path: pathlib.Path, num_retrieved_observations: int) -> None:
    with norm_stats_path.open() as f:
        stats = json.load(f)["norm_stats"]

    required_keys = ["query_state", "query_actions"]
    for retrieved_idx in range(num_retrieved_observations):
        required_keys.extend([f"retrieved_{retrieved_idx}_state", f"retrieved_{retrieved_idx}_actions"])

    missing_keys = [key for key in required_keys if key not in stats]
    if missing_keys:
        raise ValueError(f"Missing RICL-LIBERO norm stats keys: {missing_keys}")


def _validate_max_distance(path: pathlib.Path) -> float:
    with path.open() as f:
        max_distance = float(json.load(f)["distances"]["max"])
    if max_distance <= 0:
        raise ValueError(f"Expected positive max distance in {path}, got {max_distance}.")
    return max_distance


def validate(
    *,
    preprocessing_dir: pathlib.Path,
    assets_dir: pathlib.Path,
    num_retrieved_observations: int,
    task_indices: tuple[int, ...],
    artifact_role: str,
) -> None:
    episodes_dir = preprocessing_dir / "episodes"
    retrieval_dir = preprocessing_dir / "retrieval"
    preprocessing_metadata_path = preprocessing_dir / "retrieval_metadata.json"
    retrieval_metadata_path = retrieval_dir / "retrieval_metadata.json"
    if preprocessing_metadata_path.exists():
        metadata = json.loads(preprocessing_metadata_path.read_text())
    elif retrieval_metadata_path.exists():
        metadata = json.loads(retrieval_metadata_path.read_text())
    else:
        metadata = {}
    indices_path = preprocessing_dir / metadata.get("index_file", DEFAULT_INDICES_FILE)
    norm_stats_path = assets_dir / "norm_stats.json"
    asset_max_distance_path = assets_dir / "max_distance.json"
    preprocessing_max_distance_path = preprocessing_dir / "max_distance.json"
    retrieval_max_distance_path = retrieval_dir / "max_distance.json"
    asset_metadata_path = assets_dir / "retrieval_metadata.json"

    required_paths = [episodes_dir, retrieval_dir, retrieval_max_distance_path, retrieval_metadata_path]
    if artifact_role == "train":
        required_paths.extend(
            [
                indices_path,
                norm_stats_path,
                asset_max_distance_path,
                preprocessing_max_distance_path,
                preprocessing_metadata_path,
                asset_metadata_path,
            ]
        )

    for path in required_paths:
        _require_path(path)

    episode_task, episode_frames = _load_episode_metadata(episodes_dir)
    query_indices = retrieved_indices = distances = None
    asset_max_distance = None
    if artifact_role == "train":
        with np.load(indices_path) as indices:
            query_indices = np.asarray(indices["query_indices"])
            retrieved_indices = np.asarray(indices["retrieved_indices"])
            distances = np.asarray(indices["distances"])

        if query_indices.ndim != 2 or query_indices.shape[1] != 2:
            raise ValueError(f"Expected query_indices shape (N, 2), got {query_indices.shape}.")
        if retrieved_indices.ndim != 3 or retrieved_indices.shape[1:] != (num_retrieved_observations, 2):
            raise ValueError(
                f"Expected retrieved_indices shape (N, {num_retrieved_observations}, 2), "
                f"got {retrieved_indices.shape}."
            )
        if distances.shape != (len(query_indices), num_retrieved_observations + 1):
            raise ValueError(
                f"Expected distances shape ({len(query_indices)}, {num_retrieved_observations + 1}), "
                f"got {distances.shape}."
            )

        _validate_index_frames(
            query_indices,
            retrieved_indices,
            episode_task=episode_task,
            episode_frames=episode_frames,
        )
        _validate_norm_stats(norm_stats_path, num_retrieved_observations)
        asset_max_distance = _validate_max_distance(asset_max_distance_path)
        _validate_max_distance(preprocessing_max_distance_path)
    retrieval_max_distance = _validate_max_distance(retrieval_max_distance_path)
    metadata = json.loads(retrieval_metadata_path.read_text())
    if artifact_role == "train" and metadata.get("index_file") != indices_path.name:
        raise ValueError(f"Expected metadata index_file={indices_path.name}, got {metadata.get('index_file')}.")
    if metadata.get("num_retrieved_observations") != num_retrieved_observations:
        raise ValueError(
            "Expected metadata num_retrieved_observations="
            f"{num_retrieved_observations}, got {metadata.get('num_retrieved_observations')}."
        )

    for task_index in task_indices:
        task_dir = retrieval_dir / f"task_{task_index:02d}"
        _require_path(task_dir)
        if not any(task_dir.glob("*.npz")):
            raise ValueError(f"No .npz retrieval episodes found in {task_dir}.")

    print(
        "RICL-LIBERO artifacts OK",
        f"role={artifact_role}",
        f"query_indices={None if query_indices is None else query_indices.shape}",
        f"retrieved_indices={None if retrieved_indices is None else retrieved_indices.shape}",
        f"distances={None if distances is None else distances.shape}",
        f"max_distance={(asset_max_distance or retrieval_max_distance):.6f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated RICL-LIBERO preprocessing artifacts.")
    parser.add_argument("--preprocessing-dir", type=pathlib.Path, default=pathlib.Path("ricl_libero_preprocessing"))
    parser.add_argument(
        "--assets-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets/pi0_fast_libero_ricl/libero"),
    )
    parser.add_argument("--num-retrieved-observations", type=int, default=4)
    parser.add_argument("--task-indices", type=_parse_task_indices, default=tuple(range(10)))
    parser.add_argument("--artifact-role", choices=("train", "retrieval"), default="train")
    args = parser.parse_args()

    validate(
        preprocessing_dir=args.preprocessing_dir,
        assets_dir=args.assets_dir,
        num_retrieved_observations=args.num_retrieved_observations,
        task_indices=args.task_indices,
        artifact_role=args.artifact_role,
    )


if __name__ == "__main__":
    main()
