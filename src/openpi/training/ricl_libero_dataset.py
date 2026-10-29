from collections.abc import Mapping
import json
import pathlib
from typing import Any, SupportsIndex

import numpy as np

DEFAULT_PREPROCESSING_DIR = pathlib.Path("ricl_libero_preprocessing")
DEFAULT_INDICES_FILE = "libero10_indices_and_distances.npz"
DEFAULT_MAX_DISTANCE_FILE = pathlib.Path("assets/pi0_fast_libero_ricl/libero/max_distance.json")
DEFAULT_RETRIEVAL_METADATA_FILE = "retrieval_metadata.json"


def get_libero_action_chunk(actions: np.ndarray, step_idx: int, action_horizon: int) -> np.ndarray:
    """Return a fixed-horizon LIBERO action chunk, padding by repeating the last valid action."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected LIBERO actions with shape (T, 7), got {actions.shape}")
    if len(actions) == 0:
        raise ValueError("Cannot build an action chunk from an empty action array.")

    chunk = []
    for offset in range(action_horizon):
        action_idx = step_idx + offset
        if action_idx < len(actions):
            chunk.append(actions[action_idx])
        else:
            chunk.append(actions[-1])

    return np.stack(chunk, axis=0).astype(np.float32)


def build_libero_state_retrieval_indices(
    episodes: Mapping[int, Mapping[str, Any]],
    *,
    num_retrieved_observations: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build same-task, cross-episode retrieval indices from normalized state L2 distance."""
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)
    state_scale = state_std + 1e-6

    frames = []
    for ep_idx in sorted(episodes):
        ep = episodes[ep_idx]
        states = np.asarray(ep["state"], dtype=np.float32)
        frame_indices = np.asarray(ep.get("frame_indices", np.arange(len(states))), dtype=np.int32)
        task_index = int(ep["task_index"])

        if len(states) != len(frame_indices):
            raise ValueError(f"Episode {ep_idx} has {len(states)} states but {len(frame_indices)} frame indices.")

        normalized_states = (states - state_mean) / state_scale
        for step_pos, frame_idx in enumerate(frame_indices):
            frames.append(
                {
                    "episode_index": int(ep_idx),
                    "frame_index": int(frame_idx),
                    "task_index": task_index,
                    "state": normalized_states[step_pos],
                }
            )

    if not frames:
        raise ValueError("Cannot build LIBERO RICL retrieval indices from zero frames.")

    query_indices = []
    retrieved_indices = []
    distances = []

    for query_frame in frames:
        candidates = [
            frame
            for frame in frames
            if frame["task_index"] == query_frame["task_index"]
            and frame["episode_index"] != query_frame["episode_index"]
        ]
        if not candidates:
            raise ValueError(
                "Each LIBERO RICL query needs at least one same-task frame from a different episode. "
                f"No candidates found for episode={query_frame['episode_index']} task={query_frame['task_index']}."
            )

        candidate_states = np.stack([frame["state"] for frame in candidates], axis=0)
        query_to_candidate_distances = np.linalg.norm(candidate_states - query_frame["state"], axis=-1)
        order = np.argsort(query_to_candidate_distances, kind="stable")
        if len(order) < num_retrieved_observations:
            repeats = int(np.ceil(num_retrieved_observations / len(order)))
            order = np.tile(order, repeats)
        order = order[:num_retrieved_observations]
        selected = [candidates[int(candidate_idx)] for candidate_idx in order]

        first_retrieved_state = selected[0]["state"]
        retrieved_distances = [
            float(np.linalg.norm(retrieved_frame["state"] - first_retrieved_state)) for retrieved_frame in selected
        ]
        query_distance = float(np.linalg.norm(query_frame["state"] - first_retrieved_state))

        query_indices.append([query_frame["episode_index"], query_frame["frame_index"]])
        retrieved_indices.append(
            [[retrieved_frame["episode_index"], retrieved_frame["frame_index"]] for retrieved_frame in selected]
        )
        distances.append([*retrieved_distances, query_distance])

    return (
        np.asarray(query_indices, dtype=np.int32),
        np.asarray(retrieved_indices, dtype=np.int32),
        np.asarray(distances, dtype=np.float32),
    )


def build_libero_embedding_retrieval_indices(
    episodes: Mapping[int, Mapping[str, Any]],
    *,
    num_retrieved_observations: int,
    embedding_key: str = "embeddings",
    use_faiss: bool | None = None,
    exact_search_max_candidates: int = 20000,
    query_batch_size: int = 64,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build same-task, cross-episode retrieval indices from image embedding L2 distance."""
    del use_faiss, exact_search_max_candidates

    task_groups: dict[int, dict[str, Any]] = {}
    global_queries = []
    for ep_idx in sorted(episodes):
        ep = episodes[ep_idx]
        embeddings = np.asarray(ep[embedding_key], dtype=np.float32)
        frame_indices = np.asarray(ep.get("frame_indices", np.arange(len(embeddings))), dtype=np.int32)
        task_index = int(ep["task_index"])

        if embeddings.ndim != 2:
            raise ValueError(f"Episode {ep_idx} has embeddings with shape {embeddings.shape}; expected (T, D).")
        if len(embeddings) != len(frame_indices):
            raise ValueError(
                f"Episode {ep_idx} has {len(embeddings)} embeddings but {len(frame_indices)} frame indices."
            )

        group = task_groups.setdefault(
            task_index,
            {
                "embedding_chunks": [],
                "episode_indices": [],
                "frame_indices": [],
            },
        )
        task_position_start = len(group["episode_indices"])
        group["embedding_chunks"].append(embeddings)
        group["episode_indices"].extend([int(ep_idx)] * len(embeddings))
        group["frame_indices"].extend(frame_indices.astype(np.int32).tolist())
        global_queries.extend((task_index, task_position_start + step_pos) for step_pos in range(len(embeddings)))

    if not global_queries:
        raise ValueError("Cannot build LIBERO RICL retrieval indices from zero embedded frames.")

    task_results = {}
    task_items = sorted(task_groups.items())
    for task_index, group in _maybe_tqdm(task_items, enabled=show_progress, desc="Building embedding retrieval"):
        embeddings = np.concatenate(group["embedding_chunks"], axis=0).astype(np.float32, copy=False)
        episode_indices = np.asarray(group["episode_indices"], dtype=np.int32)
        frame_indices = np.asarray(group["frame_indices"], dtype=np.int32)

        if embeddings.shape[0] != episode_indices.shape[0] or embeddings.shape[0] != frame_indices.shape[0]:
            raise ValueError(f"Task {task_index} has inconsistent embedding/index counts.")
        if len(np.unique(episode_indices)) < 2:
            raise ValueError(
                "Each LIBERO RICL query needs at least one same-task embedded frame from a different episode. "
                f"No candidates found for task={task_index}."
            )

        selected_positions = _select_nearest_embedding_positions(
            embeddings,
            episode_indices,
            num_retrieved_observations=num_retrieved_observations,
            query_batch_size=query_batch_size,
        )
        distances = _compute_ricl_embedding_distances(
            embeddings,
            selected_positions,
            query_batch_size=query_batch_size,
        )

        task_results[task_index] = {
            "episode_indices": episode_indices,
            "frame_indices": frame_indices,
            "selected_positions": selected_positions,
            "distances": distances,
        }

    query_indices = []
    retrieved_indices = []
    all_distances = []
    for task_index, task_position in global_queries:
        result = task_results[task_index]
        selected_positions = result["selected_positions"][task_position]

        query_indices.append(
            [
                int(result["episode_indices"][task_position]),
                int(result["frame_indices"][task_position]),
            ]
        )
        retrieved_indices.append(
            [
                [
                    int(result["episode_indices"][selected_position]),
                    int(result["frame_indices"][selected_position]),
                ]
                for selected_position in selected_positions
            ]
        )
        all_distances.append(result["distances"][task_position])

    return (
        np.asarray(query_indices, dtype=np.int32),
        np.asarray(retrieved_indices, dtype=np.int32),
        np.asarray(all_distances, dtype=np.float32),
    )


def _maybe_tqdm(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        import tqdm

        return tqdm.tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def _select_nearest_embedding_positions(
    embeddings: np.ndarray,
    episode_indices: np.ndarray,
    *,
    num_retrieved_observations: int,
    query_batch_size: int,
) -> np.ndarray:
    if _torch_cuda_available():
        try:
            return _select_nearest_embedding_positions_torch(
                embeddings,
                episode_indices,
                num_retrieved_observations=num_retrieved_observations,
                query_batch_size=query_batch_size,
            )
        except RuntimeError:
            pass

    return _select_nearest_embedding_positions_numpy(
        embeddings,
        episode_indices,
        num_retrieved_observations=num_retrieved_observations,
        query_batch_size=query_batch_size,
    )


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _select_nearest_embedding_positions_torch(
    embeddings: np.ndarray,
    episode_indices: np.ndarray,
    *,
    num_retrieved_observations: int,
    query_batch_size: int,
) -> np.ndarray:
    import torch

    device = torch.device("cuda")
    embedding_tensor = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    all_norms = torch.sum(embedding_tensor * embedding_tensor, dim=1)
    all_episode_indices = torch.from_numpy(episode_indices).to(device=device)
    selected = np.empty((len(embeddings), num_retrieved_observations), dtype=np.int32)

    for start in range(0, len(embeddings), query_batch_size):
        end = min(start + query_batch_size, len(embeddings))
        queries = embedding_tensor[start:end]
        distances = (
            torch.sum(queries * queries, dim=1, keepdim=True)
            + all_norms.reshape(1, -1)
            - 2.0 * queries @ embedding_tensor.T
        )
        same_episode = all_episode_indices[start:end].reshape(-1, 1) == all_episode_indices.reshape(1, -1)
        distances = distances.masked_fill(same_episode, float("inf"))
        selected[start:end] = _repeat_valid_orders(
            torch.argsort(distances, dim=1).detach().cpu().numpy(),
            np.isfinite(distances.detach().cpu().numpy()),
            num_retrieved_observations,
        )

    return selected


def _select_nearest_embedding_positions_numpy(
    embeddings: np.ndarray,
    episode_indices: np.ndarray,
    *,
    num_retrieved_observations: int,
    query_batch_size: int,
) -> np.ndarray:
    all_norms = np.sum(embeddings * embeddings, axis=1)
    selected = np.empty((len(embeddings), num_retrieved_observations), dtype=np.int32)

    for start in range(0, len(embeddings), query_batch_size):
        end = min(start + query_batch_size, len(embeddings))
        query_embeddings = embeddings[start:end]
        distances = (
            np.sum(query_embeddings * query_embeddings, axis=1, keepdims=True)
            + all_norms.reshape(1, -1)
            - 2.0 * query_embeddings @ embeddings.T
        )
        distances[episode_indices[start:end, None] == episode_indices[None, :]] = np.inf
        selected[start:end] = _repeat_valid_orders(
            np.argsort(distances, axis=1, kind="stable"),
            np.isfinite(distances),
            num_retrieved_observations,
        )

    return selected


def _repeat_valid_orders(order: np.ndarray, finite_mask: np.ndarray, num_retrieved_observations: int) -> np.ndarray:
    selected = np.empty((order.shape[0], num_retrieved_observations), dtype=np.int32)
    for row_idx, row_order in enumerate(order):
        valid_order = row_order[finite_mask[row_idx, row_order]]
        if len(valid_order) == 0:
            raise ValueError("Each LIBERO RICL query needs at least one cross-episode retrieval candidate.")
        if len(valid_order) < num_retrieved_observations:
            repeats = int(np.ceil(num_retrieved_observations / len(valid_order)))
            valid_order = np.tile(valid_order, repeats)
        selected[row_idx] = valid_order[:num_retrieved_observations]
    return selected


def _compute_ricl_embedding_distances(
    embeddings: np.ndarray,
    selected_positions: np.ndarray,
    *,
    query_batch_size: int,
) -> np.ndarray:
    distances = np.empty((len(embeddings), selected_positions.shape[1] + 1), dtype=np.float32)
    for start in range(0, len(embeddings), query_batch_size):
        end = min(start + query_batch_size, len(embeddings))
        first_retrieved_embeddings = embeddings[selected_positions[start:end, 0]]
        for retrieved_idx in range(selected_positions.shape[1]):
            retrieved_embeddings = embeddings[selected_positions[start:end, retrieved_idx]]
            distances[start:end, retrieved_idx] = np.linalg.norm(
                retrieved_embeddings - first_retrieved_embeddings,
                axis=1,
            )
        distances[start:end, -1] = np.linalg.norm(embeddings[start:end] - first_retrieved_embeddings, axis=1)
    distances[:, 0] = 0.0
    return distances


class RiclLiberoDataset:
    def __init__(
        self,
        model_config,
        repo_id: str = "physical-intelligence/libero",
        task_indices=None,
        num_retrieved_observations: int = 4,
        preprocessing_dir: str | pathlib.Path = DEFAULT_PREPROCESSING_DIR,
        max_distance_path: str | pathlib.Path | None = None,
    ):
        if task_indices is None:
            task_indices = range(10)
        del repo_id, task_indices
        self.action_horizon = model_config.action_horizon
        self.action_dim = model_config.action_dim
        self.num_retrieved_observations = num_retrieved_observations
        self.use_action_interpolation = model_config.use_action_interpolation
        self.lamda = model_config.lamda
        self.preprocessing_dir = pathlib.Path(preprocessing_dir)
        self.episodes_dir = self.preprocessing_dir / "episodes"
        self.metadata_path = self.preprocessing_dir / DEFAULT_RETRIEVAL_METADATA_FILE
        self.metadata = self._load_metadata()
        self.indices_path = self._resolve_indices_path()

        if self.action_dim != 7:
            raise ValueError(f"RiclLiberoDataset expects action_dim=7, got {self.action_dim}.")
        if not self.episodes_dir.exists() or not self.indices_path.exists():
            raise FileNotFoundError(
                "RICL-LIBERO preprocessing artifacts are missing. Run "
                "`python scripts/setup_ricl_libero.py` first. Expected "
                f"{self.episodes_dir} and {self.indices_path}."
            )

        indices_and_distances = np.load(self.indices_path)
        self.all_query_indices = np.asarray(indices_and_distances["query_indices"], dtype=np.int32)
        self.all_retrieved_indices = np.asarray(indices_and_distances["retrieved_indices"], dtype=np.int32)
        self.all_distances = np.asarray(indices_and_distances["distances"], dtype=np.float32)

        if self.all_retrieved_indices.shape[1] < self.num_retrieved_observations:
            raise ValueError(
                f"Expected at least {self.num_retrieved_observations} retrieved observations, "
                f"got {self.all_retrieved_indices.shape[1]}."
            )
        self.all_retrieved_indices = self.all_retrieved_indices[:, : self.num_retrieved_observations, :]
        self.all_distances = np.concatenate(
            [
                self.all_distances[:, : self.num_retrieved_observations],
                self.all_distances[:, -1:],
            ],
            axis=1,
        )

        expected_distance_shape = (len(self.all_query_indices), self.num_retrieved_observations + 1)
        if self.all_distances.shape != expected_distance_shape:
            raise ValueError(f"Expected distances shape {expected_distance_shape}, got {self.all_distances.shape}.")

        self.max_distance = self._load_max_distance(max_distance_path)
        self.episode_paths = self._make_episode_paths()
        self._validate_metadata()
        self.frame_positions = self._load_frame_positions()

    def __len__(self) -> int:
        return len(self.all_query_indices)

    def __getitem__(self, index: SupportsIndex) -> dict:
        item_index = index.__index__()
        retrieved_indices = self.all_retrieved_indices[item_index]
        query_ep_idx, query_frame_idx = self.all_query_indices[item_index]
        ep_indices = {int(query_ep_idx), *[int(ep_idx) for ep_idx, _ in retrieved_indices]}
        episodes = {ep_idx: self._load_episode(ep_idx) for ep_idx in ep_indices}

        data = {}
        for retrieved_idx, (ep_idx, frame_idx) in enumerate(retrieved_indices):
            self._add_observation(
                data, f"retrieved_{retrieved_idx}_", episodes[int(ep_idx)], int(ep_idx), int(frame_idx)
            )

        self._add_observation(data, "query_", episodes[int(query_ep_idx)], int(query_ep_idx), int(query_frame_idx))

        if self.use_action_interpolation:
            normalized_distances = self.all_distances[item_index] / max(self.max_distance, 1e-6)
            data["exp_lamda_distances"] = np.exp(-self.lamda * normalized_distances).reshape(-1, 1).astype(np.float32)

        return data

    def _add_observation(self, data: dict, prefix: str, episode: dict, ep_idx: int, frame_idx: int) -> None:
        step_idx = self.frame_positions[ep_idx][frame_idx]
        data[f"{prefix}state"] = np.asarray(episode["state"][step_idx], dtype=np.float32)
        data[f"{prefix}image"] = episode["image"][step_idx]
        data[f"{prefix}wrist_image"] = episode["wrist_image"][step_idx]
        data[f"{prefix}actions"] = get_libero_action_chunk(episode["actions"], step_idx, self.action_horizon)
        data[f"{prefix}prompt"] = episode["prompt"]

    def _make_episode_paths(self) -> dict[int, pathlib.Path]:
        episode_indices = set(np.unique(self.all_query_indices[:, 0]).tolist())
        episode_indices.update(np.unique(self.all_retrieved_indices[:, :, 0]).tolist())
        episode_paths = {int(ep_idx): self.episodes_dir / f"episode_{int(ep_idx)}.npz" for ep_idx in episode_indices}

        missing_paths = [path for path in episode_paths.values() if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths[:5])
            raise FileNotFoundError(f"Missing LIBERO RICL episode cache files: {missing}")
        return episode_paths

    def _load_frame_positions(self) -> dict[int, dict[int, int]]:
        frame_positions = {}
        for ep_idx, episode_path in self.episode_paths.items():
            with np.load(episode_path, allow_pickle=True) as episode:
                frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
            frame_positions[ep_idx] = {int(frame_idx): pos for pos, frame_idx in enumerate(frame_indices)}
        return frame_positions

    def _load_episode(self, ep_idx: int) -> dict:
        with np.load(self.episode_paths[ep_idx], allow_pickle=True) as episode:
            return {
                "image": episode["image"],
                "wrist_image": episode["wrist_image"],
                "state": episode["state"].astype(np.float32),
                "actions": episode["actions"].astype(np.float32),
                "prompt": str(episode["prompt"].item()),
            }

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        return json.loads(self.metadata_path.read_text())

    def _resolve_indices_path(self) -> pathlib.Path:
        index_file = self.metadata.get("index_file", DEFAULT_INDICES_FILE)
        indices_path = self.preprocessing_dir / index_file
        if indices_path.exists():
            return indices_path

        fallback_path = self.preprocessing_dir / DEFAULT_INDICES_FILE
        if index_file != DEFAULT_INDICES_FILE and fallback_path.exists():
            return fallback_path

        return indices_path

    def _validate_metadata(self) -> None:
        metadata = self.metadata
        if metadata.get("backend") != "dinov2":
            return
        retrieval_key = metadata.get("retrieval_key")
        required_embedding_keys = _embedding_keys_for_retrieval_key(retrieval_key)
        for episode_path in self.episode_paths.values():
            with np.load(episode_path, allow_pickle=True) as episode:
                missing = [key for key in required_embedding_keys if key not in episode]
            if missing:
                raise ValueError(f"{episode_path} is missing embeddings required by {self.metadata_path}: {missing}")

    def _load_max_distance(self, max_distance_path: str | pathlib.Path | None) -> float:
        candidate_paths = [
            pathlib.Path(max_distance_path) if max_distance_path is not None else None,
            self.preprocessing_dir / "max_distance.json",
            DEFAULT_MAX_DISTANCE_FILE,
        ]
        for candidate_path in candidate_paths:
            if candidate_path is not None and candidate_path.exists():
                with candidate_path.open() as f:
                    max_distance = float(json.load(f)["distances"]["max"])
                return max(max_distance, 1e-6)

        raise FileNotFoundError(
            "Missing LIBERO RICL max distance. Run `python scripts/setup_ricl_libero.py` first. "
            f"Expected {DEFAULT_MAX_DISTANCE_FILE} or {self.preprocessing_dir / 'max_distance.json'}."
        )


def _embedding_keys_for_retrieval_key(retrieval_key: str | None) -> list[str]:
    if retrieval_key == "image":
        return ["image_embeddings"]
    if retrieval_key == "wrist_image":
        return ["wrist_image_embeddings"]
    if retrieval_key == "both":
        return ["image_embeddings", "wrist_image_embeddings"]
    return []
