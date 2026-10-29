from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import time
from typing import Any

import numpy as np

from openpi.policies import libero_dino_feature_cache
from openpi.policies import libero_retrieval_benchmark as rb
from openpi.policies.utils import embed
from openpi.policies.utils import load_dinov2

SUPPORTED_METHODS = ("dino", "dino_state", "pi0_first_fusion", "random")
DINO_METHODS = ("dino", "dino_state")
SUPPORTED_SCOPES = ("task", "global")
SUPPORTED_SEARCH_DEVICES = ("auto", "cpu", "cuda")
DEFAULT_DISTANCE_CHUNK_SIZE = 256


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CandidateFrame:
    task_id: int
    episode_id: int
    frame_index: int
    episode_path: pathlib.Path
    step_index: int


@dataclasses.dataclass(frozen=True)
class RetrievalResult:
    frames: tuple[dict[str, Any], ...]
    trace: tuple[dict[str, Any], ...]
    interpolation_distances: np.ndarray
    timings_ms: dict[str, float]


class Pi0FirstFusionQueryEmbedder:
    """Online query encoder matching the completed Pi0 first-fusion benchmark."""

    def __init__(self, config_name: str, checkpoint_dir: pathlib.Path, dtype: str):
        import jax
        import jax.numpy as jnp

        from openpi import transforms
        from openpi.models import model as model_lib
        from openpi.models import pi0_first_fusion
        from openpi.training import checkpoints
        from openpi.training import config as train_config_lib

        config = train_config_lib.get_config(config_name)
        if dtype != "float32":
            raise ValueError(
                "Pi first-fusion retrieval must use float32 to match its candidate cache; " f"got {dtype!r}."
            )
        model_config = dataclasses.replace(config.model, dtype=dtype)
        self._encoder = pi0_first_fusion.FirstFusionFeatureEncoder(model_config, checkpoint_dir)
        data_config = config.data.create(config.assets_dirs, model_config)
        if data_config.asset_id is None:
            raise ValueError("Pi retrieval config must define an asset_id.")
        norm_stats = checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
        self._transform = transforms.compose(
            [
                *data_config.data_transforms.inputs,
                transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._model_lib = model_lib
        self._jax = jax
        self._jnp = jnp

    def __call__(self, image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, prompt: str) -> np.ndarray:
        raw = rb.make_pi0_observation_input(image, wrist_image, state, prompt)
        transformed = self._transform(raw)
        batch = self._jax.tree.map(lambda value: self._jnp.asarray(value)[None, ...], transformed)
        observation = self._model_lib.Observation.from_dict(batch)
        feature = np.asarray(self._jax.device_get(self._encoder(observation))[0], dtype=np.float32)
        if not np.all(np.isfinite(feature)):
            raise ValueError("Pi0 query feature contains NaN or infinity.")
        return feature


class LiberoRuntimeRetriever:
    """Search a fixed LIBERO episode pool with a benchmarked retrieval method."""

    def __init__(
        self,
        demos_dir: pathlib.Path,
        *,
        method: str,
        scope: str,
        pi_feature_dir: pathlib.Path | None = None,
        pi_config_name: str = "pi0_fast_libero90_normal",
        pi_checkpoint_dir: pathlib.Path | None = None,
        pi_retriever_dtype: str = "float32",
        state_norm_checkpoint_dir: pathlib.Path | None = None,
        dino_state_beta: float = 1.0,
        embedding_type: str = "64PATCHES",
        distance_chunk_size: int = DEFAULT_DISTANCE_CHUNK_SIZE,
        search_device: str = "auto",
    ):
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported comparative retrieval method {method!r}; expected {SUPPORTED_METHODS}.")
        if scope not in SUPPORTED_SCOPES:
            raise ValueError(f"Unsupported retrieval scope {scope!r}; expected {SUPPORTED_SCOPES}.")
        self.method = method
        self.scope = scope
        self._beta = float(dino_state_beta)
        self._embedding_type = embedding_type
        if distance_chunk_size <= 0:
            raise ValueError(f"distance_chunk_size must be positive, got {distance_chunk_size}.")
        if search_device not in SUPPORTED_SEARCH_DEVICES:
            raise ValueError(
                f"Unsupported retrieval search_device={search_device!r}; expected {SUPPORTED_SEARCH_DEVICES}."
            )
        self._distance_chunk_size = int(distance_chunk_size)
        self._requested_search_device = search_device
        self._frames, self._states, self._features = self._load_candidates(
            demos_dir, method=method, pi_feature_dir=pi_feature_dir
        )
        self._task_ids = np.asarray([frame.task_id for frame in self._frames], dtype=np.int32)
        self._episode_ids = np.asarray([frame.episode_id for frame in self._frames], dtype=np.int32)
        self._frame_indices = np.asarray([frame.frame_index for frame in self._frames], dtype=np.int32)
        self._task_candidate_indices = {
            int(task_id): np.flatnonzero(self._task_ids == task_id) for task_id in np.unique(self._task_ids)
        }

        self._dinov2 = None
        self._pi_embedder = None
        self._state_mean = None
        self._state_std = None
        self._search_device = "cpu"
        if method in DINO_METHODS:
            if method == "dino_state":
                if state_norm_checkpoint_dir is None:
                    raise ValueError("DINO+state retrieval requires state_norm_checkpoint_dir.")
                self._state_mean, self._state_std = self._load_state_stats(state_norm_checkpoint_dir)
                self._normalized_states = rb.z_normalize(self._states, self._state_mean, self._state_std)
            else:
                self._normalized_states = None
            self._dinov2 = load_dinov2()
            self._search_device = self._resolve_search_device(search_device)
            logger.info(
                "DINO candidate search uses %s with chunk_size=%d",
                self._search_device,
                self._distance_chunk_size,
            )
        elif method == "pi0_first_fusion":
            self._normalized_states = None
            if pi_checkpoint_dir is None:
                raise ValueError("Pi retrieval requires pi_checkpoint_dir.")
            self._pi_embedder = Pi0FirstFusionQueryEmbedder(pi_config_name, pi_checkpoint_dir, pi_retriever_dtype)
        else:
            self._normalized_states = None

    @staticmethod
    def _load_candidates(
        demos_dir: pathlib.Path,
        *,
        method: str,
        pi_feature_dir: pathlib.Path | None,
    ) -> tuple[tuple[CandidateFrame, ...], np.ndarray, np.ndarray]:
        task_dirs = sorted(
            (path for path in demos_dir.glob("task_*") if path.is_dir()),
            key=lambda path: int(path.name.split("_")[-1]),
        )
        if not task_dirs:
            raise FileNotFoundError(f"No task_XX candidate folders found in {demos_dir}.")
        dino_feature_path, dino_manifest_path = libero_dino_feature_cache.cache_paths(demos_dir)
        if dino_feature_path.exists() != dino_manifest_path.exists():
            raise FileNotFoundError(
                f"Incomplete DINO feature cache under {dino_feature_path.parent}; expected both "
                f"{dino_feature_path.name} and {dino_manifest_path.name}."
            )
        use_dino_memmap = method in DINO_METHODS and dino_feature_path.is_file() and dino_manifest_path.is_file()
        frames: list[CandidateFrame] = []
        state_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        for task_dir in task_dirs:
            task_id = int(task_dir.name.split("_")[-1])
            task_frame_count = 0
            for episode_path in sorted(
                task_dir.glob("episode_*.npz"), key=lambda path: int(path.stem.split("_")[-1])
            ):
                episode_id = int(episode_path.stem.split("_")[-1])
                with np.load(episode_path, allow_pickle=True) as episode:
                    states = np.asarray(episode["state"], dtype=np.float32)
                    frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
                    if len(states) != len(frame_indices):
                        raise ValueError(f"State/frame count mismatch in {episode_path}.")
                    state_parts.append(states)
                    if method in DINO_METHODS and not use_dino_memmap:
                        features = np.asarray(episode["image_embeddings"], dtype=np.float32)
                        if len(features) != len(states):
                            raise ValueError(f"DINO feature/frame count mismatch in {episode_path}.")
                        feature_parts.append(features)
                    frames.extend(
                        CandidateFrame(task_id, episode_id, int(frame_index), episode_path, step_index)
                        for step_index, frame_index in enumerate(frame_indices)
                    )
                    task_frame_count += len(states)
            if method == "pi0_first_fusion":
                if pi_feature_dir is None:
                    raise ValueError("Pi retrieval requires pi_feature_dir.")
                feature_path = pi_feature_dir / f"task_{task_id:02d}.npy"
                if not feature_path.is_file():
                    raise FileNotFoundError(f"Pi candidate feature cache not found: {feature_path}")
                features = np.load(feature_path, mmap_mode="r")
                if len(features) != task_frame_count:
                    raise ValueError(
                        f"Pi cache/frame count mismatch for task {task_id}: {len(features)} != {task_frame_count}."
                    )
                feature_parts.append(np.asarray(features, dtype=np.float32))
        states = np.concatenate(state_parts)
        if use_dino_memmap:
            manifest = json.loads(dino_manifest_path.read_text())
            if manifest.get("version") != libero_dino_feature_cache.CACHE_VERSION:
                raise ValueError(f"Unsupported DINO feature cache version in {dino_manifest_path}.")
            # Copy-on-write mode exposes writable views to torch without ever
            # modifying the cache file or materializing the full matrix.
            features = np.load(dino_feature_path, mmap_mode="c")
            expected_order_hash = libero_dino_feature_cache.frame_order_sha256(
                (frame.task_id, frame.episode_id, frame.frame_index) for frame in frames
            )
            if manifest.get("frame_order_sha256") != expected_order_hash:
                raise ValueError(f"DINO feature cache frame order differs from episodes in {demos_dir}.")
            if list(features.shape) != manifest.get("shape"):
                raise ValueError(f"DINO feature cache shape differs from {dino_manifest_path}.")
            if features.dtype != np.float32:
                raise ValueError(f"DINO feature cache must use float32, got {features.dtype}.")
        elif method in DINO_METHODS or method == "pi0_first_fusion":
            features = np.concatenate(feature_parts)
        else:
            # Random retrieval needs only stable frame metadata. Keep the first
            # dimension aligned without materializing an unused feature matrix.
            features = np.empty((len(frames), 0), dtype=np.float32)
        if not (len(frames) == len(states) == len(features)):
            raise ValueError("Runtime candidate frame, state, and feature counts differ.")
        return tuple(frames), states, features

    @staticmethod
    def _load_state_stats(checkpoint_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
        path = checkpoint_dir / "assets" / "libero" / "norm_stats.json"
        stats = json.loads(path.read_text())["norm_stats"]["state"]
        return np.asarray(stats["mean"], dtype=np.float32), np.asarray(stats["std"], dtype=np.float32)

    def retrieve(
        self,
        *,
        task_id: int,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
        k: int,
    ) -> RetrievalResult:
        total_started = time.perf_counter()
        search_indices = (
            self._task_candidate_indices.get(task_id, np.asarray([], dtype=np.int64))
            if self.scope == "task"
            else np.arange(len(self._frames), dtype=np.int64)
        )
        if len(search_indices) < k:
            raise ValueError(f"Only {len(search_indices)} candidates available for task={task_id}, scope={self.scope}.")

        random_seed = None
        if self.method in DINO_METHODS:
            assert self._dinov2 is not None
            query_started = time.perf_counter()
            query_feature = (
                embed(self._to_image_uint8(image), self._dinov2, embedding_type=self._embedding_type)
                .reshape(-1)
                .astype(np.float32)
            )
            query_ms = (time.perf_counter() - query_started) * 1000
            search_started = time.perf_counter()
            image_distances = self._chunked_feature_distances(search_indices, query_feature)
            image_scale = max(float(np.median(image_distances)), rb.EPS)
            if self.method == "dino_state":
                assert self._state_mean is not None
                assert self._state_std is not None
                assert self._normalized_states is not None
                query_state = rb.z_normalize(np.asarray(state, dtype=np.float32), self._state_mean, self._state_std)
                candidate_states = self._normalized_states[search_indices]
                state_distances = rb.l2_distances(candidate_states, query_state)
                state_scale = max(float(np.median(state_distances)), rb.EPS)
                distances = image_distances / image_scale + self._beta * state_distances / state_scale
                query_parts = (query_feature, query_state)
                scales = (image_scale, state_scale)
            else:
                distances = image_distances
                query_parts = (query_feature,)
                scales = (image_scale,)
            search_ms = (time.perf_counter() - search_started) * 1000
        elif self.method == "pi0_first_fusion":
            assert self._pi_embedder is not None
            query_started = time.perf_counter()
            query_feature = self._pi_embedder(image, wrist_image, state, prompt)
            query_ms = (time.perf_counter() - query_started) * 1000
            candidate_features = self._features[search_indices]
            search_started = time.perf_counter()
            distances = rb.l2_distances(candidate_features, query_feature)
            search_ms = (time.perf_counter() - search_started) * 1000
            query_parts = (query_feature,)
            scales = (1.0,)
        else:
            query_started = time.perf_counter()
            random_seed = self._query_content_seed(task_id, image, wrist_image, state, prompt)
            query_ms = (time.perf_counter() - query_started) * 1000
            search_started = time.perf_counter()
            selected_indices = np.random.default_rng(random_seed).choice(search_indices, size=k, replace=False)
            search_ms = (time.perf_counter() - search_started) * 1000
            query_parts = ()
            scales = ()

        if self.method != "random":
            local_selected = rb.deterministic_global_top_k(
                distances,
                self._task_ids[search_indices],
                self._episode_ids[search_indices],
                self._frame_indices[search_indices],
                k=k,
            )
            selected_indices = search_indices[local_selected]
            selected_distances = distances[local_selected]
        else:
            selected_distances = None
        load_started = time.perf_counter()
        loaded_frames = tuple(
            self._load_frame(self._frames[int(candidate_index)]) for candidate_index in selected_indices
        )
        frame_load_ms = (time.perf_counter() - load_started) * 1000
        candidate_pool_num_tasks = len(np.unique(self._task_ids[search_indices]))
        trace = tuple(
            {
                "rank": rank,
                "retrieved_task_id": int(self._task_ids[candidate_index]),
                "episode_id": int(self._episode_ids[candidate_index]),
                "frame_index": int(self._frame_indices[candidate_index]),
                "distance": None if selected_distances is None else float(selected_distances[rank]),
                "retrieval_method": self.method,
                "retrieval_scope": self.scope,
                "search_device": self._search_device,
                "candidate_pool_num_tasks": candidate_pool_num_tasks,
                "candidate_pool_num_frames": len(search_indices),
                **({"random_query_seed": random_seed} if random_seed is not None else {}),
            }
            for rank, candidate_index in enumerate(selected_indices)
        )
        interpolation_distances = (
            np.zeros(len(selected_indices) + 1, dtype=np.float32)
            if self.method == "random"
            else self._interpolation_distances(selected_indices, query_parts=query_parts, scales=scales)
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        return RetrievalResult(
            loaded_frames,
            trace,
            interpolation_distances,
            {
                "retrieval_query_encode": query_ms,
                "retrieval_search": search_ms,
                "retrieval_frame_load": frame_load_ms,
                "retrieval_total": total_ms,
            },
        )

    def _chunked_feature_distances(self, candidate_indices: np.ndarray, query_feature: np.ndarray) -> np.ndarray:
        """Compute a flat float32 L2 scan without materializing the candidate matrix."""
        if self._search_device == "cuda":
            return self._chunked_feature_distances_cuda(candidate_indices, query_feature)

        distances = np.empty(len(candidate_indices), dtype=np.float32)
        for start in range(0, len(candidate_indices), self._distance_chunk_size):
            stop = min(start + self._distance_chunk_size, len(candidate_indices))
            feature_chunk = self._feature_chunk(candidate_indices[start:stop])
            distances[start:stop] = rb.l2_distances(feature_chunk, query_feature)
        return distances

    def _chunked_feature_distances_cuda(self, candidate_indices: np.ndarray, query_feature: np.ndarray) -> np.ndarray:
        import torch

        distances = np.empty(len(candidate_indices), dtype=np.float32)
        device = next(self._dinov2.parameters()).device
        if device.type != "cuda":
            raise RuntimeError(f"CUDA DINO search requested, but the DINO model is on {device}.")
        with torch.inference_mode():
            query = torch.as_tensor(query_feature, dtype=torch.float32, device=device)
            for start in range(0, len(candidate_indices), self._distance_chunk_size):
                stop = min(start + self._distance_chunk_size, len(candidate_indices))
                feature_chunk = self._feature_chunk(candidate_indices[start:stop])
                candidate_tensor = torch.from_numpy(feature_chunk).to(device=device, non_blocking=False)
                chunk_distances = torch.linalg.vector_norm(candidate_tensor - query, dim=1)
                distances[start:stop] = chunk_distances.cpu().numpy()
        return distances

    def _feature_chunk(self, indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.empty((0, self._features.shape[1]), dtype=np.float32)
        first = int(indices[0])
        if len(indices) == 1 or np.all(np.diff(indices) == 1):
            return np.asarray(self._features[first : int(indices[-1]) + 1], dtype=np.float32)
        return np.asarray(self._features[indices], dtype=np.float32)

    def _resolve_search_device(self, requested: str) -> str:
        if requested == "cpu":
            return "cpu"
        try:
            import torch

            dino_device = next(self._dinov2.parameters()).device
            cuda_available = torch.cuda.is_available() and dino_device.type == "cuda"
        except (ImportError, StopIteration, AttributeError):
            cuda_available = False
        if requested == "cuda" and not cuda_available:
            raise RuntimeError("CUDA DINO search requested, but CUDA or a CUDA DINO model is unavailable.")
        return "cuda" if cuda_available else "cpu"

    def _interpolation_distances(
        self,
        selected_indices: np.ndarray,
        *,
        query_parts: tuple[np.ndarray, ...],
        scales: tuple[float, ...],
    ) -> np.ndarray:
        first = int(selected_indices[0])
        if self.method == "pi0_first_fusion":
            selected = self._features[selected_indices]
            values = [np.linalg.norm(vector - selected[0]) for vector in selected]
            values.append(np.linalg.norm(query_parts[0] - selected[0]))
            return np.asarray(values, dtype=np.float32)

        if self.method == "dino":
            image_scale = scales[0]
            first_image = self._features[first]
            values = [
                np.linalg.norm(self._features[candidate_index] - first_image) / image_scale
                for candidate_index in selected_indices
            ]
            values.append(np.linalg.norm(query_parts[0] - first_image) / image_scale)
            return np.asarray(values, dtype=np.float32)

        assert self._normalized_states is not None
        first_image = self._features[first]
        first_state = self._normalized_states[first]
        image_scale, state_scale = scales
        values = []
        for candidate_index in selected_indices:
            image_distance = np.linalg.norm(self._features[candidate_index] - first_image) / image_scale
            state_distance = np.linalg.norm(self._normalized_states[candidate_index] - first_state) / state_scale
            values.append(image_distance + self._beta * state_distance)
        query_image, query_state = query_parts
        values.append(
            np.linalg.norm(query_image - first_image) / image_scale
            + self._beta * np.linalg.norm(query_state - first_state) / state_scale
        )
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _query_content_seed(
        task_id: int,
        image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        prompt: str,
    ) -> int:
        """Return a process-independent seed derived from the complete query."""
        digest = hashlib.sha256(b"libero-runtime-random-v1\0")
        digest.update(int(task_id).to_bytes(8, byteorder="little", signed=True))
        for name, value in (("image", image), ("wrist_image", wrist_image), ("state", state)):
            array = np.ascontiguousarray(value)
            digest.update(name.encode() + b"\0")
            digest.update(array.dtype.str.encode() + b"\0")
            digest.update(json.dumps(array.shape).encode() + b"\0")
            digest.update(array.tobytes())
        digest.update(prompt.encode("utf-8"))
        return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False) & ((1 << 63) - 1)

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _load_episode(path: pathlib.Path) -> Mapping[str, Any]:
        with np.load(path, allow_pickle=True) as episode:
            return {
                "image": np.asarray(episode["image"]),
                "wrist_image": np.asarray(episode["wrist_image"]),
                "state": np.asarray(episode["state"], dtype=np.float32),
                "actions": np.asarray(episode["actions"], dtype=np.float32),
                "prompt": str(np.asarray(episode["prompt"]).item()),
            }

    def _load_frame(self, frame: CandidateFrame) -> dict[str, Any]:
        episode = self._load_episode(frame.episode_path)
        return {
            "episode": episode,
            "step_idx": frame.step_index,
        }

    @staticmethod
    def _to_image_uint8(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            return np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image.astype(np.uint8)
