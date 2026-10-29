from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
import gc
import json
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_fast_ricl as _pi0_fast_ricl
from openpi.policies import libero_runtime_retrieval
from openpi.policies.utils import EMBED_DIM
from openpi.policies.utils import embed
from openpi.policies.utils import load_dinov2
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.training.ricl_libero_dataset import DEFAULT_RETRIEVAL_METADATA_FILE
from openpi.training.ricl_libero_dataset import get_libero_action_chunk

logger = logging.getLogger()
BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        logger.debug("outputs: %s", outputs)
        final_outputs = self._output_transform(outputs)
        logger.info(f"final_outputs: {final_outputs}")
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def get_action_chunk_at_inference_time(actions, step_idx, action_horizon):
    num_steps = len(actions)
    action_chunk = []
    for i in range(action_horizon):
        if step_idx + i < num_steps:
            action_chunk.append(actions[step_idx + i])
        else:
            action_chunk.append(
                np.concatenate([np.zeros(actions.shape[-1] - 1, dtype=np.float32), actions[-1, -1:]], axis=0)
            )  # combines 0 joint vels with last gripper pos
    action_chunk = np.stack(action_chunk, axis=0)
    assert action_chunk.shape == (action_horizon, 8), f"{action_chunk.shape=}"
    return action_chunk


class RiclPolicy(BasePolicy):
    def __init__(
        self,
        model: _pi0_fast_ricl.Pi0FASTRicl,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        demos_dir: str | None = None,
        use_action_interpolation: bool | None = None,
        lamda: float | None = None,
        action_horizon: int | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model
        self._use_action_interpolation = use_action_interpolation
        self._lamda = lamda
        self._action_horizon = action_horizon or model.action_horizon
        # setup demos for retrieval
        logger.info(f"loading demos from {demos_dir}...")
        demos_path = pathlib.Path(demos_dir)
        self._demos = {
            demo_idx: np.load(folder / "processed_demo.npz")
            for demo_idx, folder in enumerate(demos_path.iterdir())
            if folder.is_dir()
        }
        self._all_indices = np.array(
            [
                (ep_idx, step_idx)
                for ep_idx in list(self._demos.keys())
                for step_idx in range(self._demos[ep_idx]["actions"].shape[0])
            ]
        )
        _all_embeddings = np.concatenate(
            [self._demos[ep_idx]["top_image_embeddings"] for ep_idx in list(self._demos.keys())]
        )
        assert _all_embeddings.shape == (len(self._all_indices), EMBED_DIM), f"{_all_embeddings.shape=}"
        self._knn_k = self._model.num_retrieved_observations
        logger.info("building retrieval index...")
        from autofaiss import build_index

        self._knn_index, knn_index_infos = build_index(
            embeddings=_all_embeddings,  # Note: embeddings have to be float to avoid errors in autofaiss / embedding_reader!
            save_on_disk=False,
            min_nearest_neighbors_to_retrieve=self._knn_k + 5,  # default: 20
            max_index_query_time_ms=10,  # default: 10
            max_index_memory_usage="25G",  # default: "16G"
            current_memory_available="50G",  # default: "32G"
            metric_type="l2",
            nb_cores=8,  # default: None # "The number of cores to use, by default will use all cores" as seen in https://criteo.github.io/autofaiss/getting_started/quantization.html#the-build-index-command
        )
        # setup the dinov2 model for embedding only
        logger.info("loading dinov2 for image embedding...")
        self._dinov2 = load_dinov2()
        with pathlib.Path("assets/max_distance.json").open() as f:
            self._max_dist = json.load(f)["distances"]["max"]
        logger.debug("self._max_dist: %s", self._max_dist)

    def retrieve(self, obs: dict) -> dict:
        more_obs = {"inference_time": True}
        # embed
        query_embedding = embed(obs["query_top_image"], self._dinov2)
        assert query_embedding.shape == (1, EMBED_DIM), f"{query_embedding.shape=}"
        # retrieve
        topk_distance, topk_indices = self._knn_index.search(query_embedding, self._knn_k)
        retrieved_indices = self._all_indices[topk_indices]
        assert retrieved_indices.shape == (1, self._knn_k, 2), f"{retrieved_indices.shape=}"
        # collect retrieved info
        for ct, (ep_idx, step_idx) in enumerate(retrieved_indices[0]):
            for key in ["state", "wrist_image", "top_image", "right_image"]:
                more_obs[f"retrieved_{ct}_{key}"] = self._demos[ep_idx][key][step_idx]
            more_obs[f"retrieved_{ct}_actions"] = get_action_chunk_at_inference_time(
                self._demos[ep_idx]["actions"], step_idx, self._action_horizon
            )
            more_obs[f"retrieved_{ct}_prompt"] = self._demos[ep_idx]["prompt"].item()
        # Compute exp_lamda_distances if use_action_interpolation
        if self._use_action_interpolation:
            first_embedding = self._demos[retrieved_indices[0, 0, 0]]["top_image_embeddings"][
                retrieved_indices[0, 0, 1]
            ]
            distances = [0.0] + [
                np.linalg.norm(self._demos[ep_idx]["top_image_embeddings"][step_idx : step_idx + 1] - first_embedding)
                for ep_idx, step_idx in retrieved_indices[0, 1:]
            ]
            distances.append(np.linalg.norm(query_embedding - first_embedding))
            distances = np.clip(np.array(distances), 0, self._max_dist) / self._max_dist
            logger.debug("distances: %s", distances)
            more_obs["exp_lamda_distances"] = np.exp(-self._lamda * distances).reshape(-1, 1)
            logger.debug("exp_lamda_distances: %s", more_obs["exp_lamda_distances"])
        return {**obs, **more_obs}

    def save_obs(self, obs: dict, date: str, prefix: str):
        fol = pathlib.Path("obs_logs") / date / prefix
        fol.mkdir(parents=True, exist_ok=True)
        current_datettime = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
        # save all images in one png
        big_top_image = []
        big_right_image = []
        big_wrist_image = []
        for ct in range(self._knn_k):
            big_top_image.append(obs[f"retrieved_{ct}_top_image"])
            big_right_image.append(obs[f"retrieved_{ct}_right_image"])
            big_wrist_image.append(obs[f"retrieved_{ct}_wrist_image"])
        big_top_image.append(obs["query_top_image"])
        big_right_image.append(obs["query_right_image"])
        big_wrist_image.append(obs["query_wrist_image"])
        final_image = np.concatenate(
            (
                np.concatenate(big_top_image, axis=1),
                np.concatenate(big_right_image, axis=1),
                np.concatenate(big_wrist_image, axis=1),
            ),
            axis=0,
        )
        Image.fromarray(final_image).save(fol / f"{current_datettime}.png")
        # save everything else to json
        with (fol / f"{current_datettime}.json").open("w") as f:
            everything_else = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obs.items() if "image" not in k
            }
            everything_else["final_image_shape"] = list(final_image.shape)
            json.dump(everything_else, f, indent=4)
        return current_datettime

    def save_tokenized_inputs(self, inputs: dict, current_datettime: str, date: str, prefix: str):
        fol = pathlib.Path("obs_logs") / date / prefix
        fol.mkdir(parents=True, exist_ok=True)
        every_tokenized_input = {}
        for k, v in inputs.items():
            if "token" in k:
                if v is None:
                    every_tokenized_input[k] = v
                    continue
                assert isinstance(v, np.ndarray), f"{k=}, {type(v)=}"
                assert v.dtype in [np.bool_, np.int64], f"{k=}, {v.dtype=}"
                every_tokenized_input[k] = v.astype(np.int32).tolist()
        with (fol / f"{current_datettime}_token_inputs.json").open("w") as f:
            json.dump(every_tokenized_input, f, indent=4)

    @override
    def infer(self, obs: dict, debug: bool = True) -> dict:  # type: ignore[misc]
        # Remove the prefix from the obs; get date; below for saving folder only
        prefix = obs.pop("prefix", "temp")
        date = datetime.now(tz=UTC).strftime("%m%d")
        # Retrieval
        logger.info("retrieving...")
        obs = self.retrieve(obs)
        # for debugging, save everything in obs
        if debug:
            logger.info("saving obs...")
            current_datettime = self.save_obs(obs, date, prefix)
        # Make a copy since transformations may modify the inputs in place.
        logger.info("transforming...")
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # for debugging, save tokenized inputs
        if debug:
            logger.info("saving tokenized inputs...")
            self.save_tokenized_inputs(inputs, current_datettime, date, prefix)
        # Make a batch and convert to jax.Array.
        logger.info("batching...")
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        logger.info("sampling...")
        outputs = {
            "query_state": inputs["query_state"],
            "query_actions": self._sample_actions(
                sample_rng,
                _model.RiclObservation.from_dict(inputs, num_retrieved_observations=self._knn_k),
                **self._sample_kwargs,
            ),
        }

        # Unbatch and convert to np.ndarray.
        logger.info("unbatching...")
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        final_outputs = self._output_transform(outputs)
        logger.debug("final_outputs: %s", final_outputs)
        return final_outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RiclLiberoPolicy(BasePolicy):
    def __init__(
        self,
        model: _pi0_fast_ricl.Pi0FASTRicl,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        demos_dir: str | None = None,
        use_action_interpolation: bool | None = None,
        lamda: float | None = None,
        action_horizon: int | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        max_distance_path: str | pathlib.Path | None = None,
        retrieval_mode: str = "nearest",
        retrieval_seed: int = 0,
        retrieval_method: str = "dino",
        retrieval_scope: str = "task",
        pi_feature_dir: str | pathlib.Path | None = None,
        pi_retriever_config: str = "pi0_fast_libero90_normal",
        pi_retriever_checkpoint: str | pathlib.Path | None = None,
        pi_retriever_dtype: str = "float32",
        state_norm_checkpoint: str | pathlib.Path | None = None,
        dino_state_beta: float = 1.0,
        retrieval_search_device: str = "auto",
        retrieval_distance_chunk_size: int = libero_runtime_retrieval.DEFAULT_DISTANCE_CHUNK_SIZE,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._model = model
        self._knn_k = model.num_retrieved_observations
        self._use_action_interpolation = (
            model.use_action_interpolation if use_action_interpolation is None else use_action_interpolation
        )
        self._lamda = model.lamda if lamda is None else lamda
        self._action_horizon = action_horizon or model.action_horizon

        if demos_dir is None:
            raise ValueError("RiclLiberoPolicy requires a demos_dir containing task_XX retrieval buffers.")
        self._demos_dir = pathlib.Path(demos_dir)
        self._retrieval_metadata = self._load_retrieval_metadata(self._demos_dir)
        self._retrieval_backend = self._retrieval_metadata["backend"]
        self._retrieval_key = self._retrieval_metadata["retrieval_key"]
        self._embedding_type = self._retrieval_metadata.get("embedding_type") or "64PATCHES"
        self._comparative_retriever = None
        if retrieval_method in libero_runtime_retrieval.SUPPORTED_METHODS:
            self._dinov2 = None
            self._state_mean = None
            self._state_std = None
            self._comparative_retriever = libero_runtime_retrieval.LiberoRuntimeRetriever(
                self._demos_dir,
                method=retrieval_method,
                scope=retrieval_scope,
                pi_feature_dir=None if pi_feature_dir is None else pathlib.Path(pi_feature_dir),
                pi_config_name=pi_retriever_config,
                pi_checkpoint_dir=(None if pi_retriever_checkpoint is None else pathlib.Path(pi_retriever_checkpoint)),
                pi_retriever_dtype=pi_retriever_dtype,
                state_norm_checkpoint_dir=(
                    None if state_norm_checkpoint is None else pathlib.Path(state_norm_checkpoint)
                ),
                dino_state_beta=dino_state_beta,
                embedding_type=self._embedding_type,
                search_device=retrieval_search_device,
                distance_chunk_size=retrieval_distance_chunk_size,
            )
        elif retrieval_method != "dino":
            raise ValueError(
                f"Unsupported retrieval_method={retrieval_method!r}; expected 'dino' or "
                f"{libero_runtime_retrieval.SUPPORTED_METHODS}."
            )
        elif self._retrieval_backend == "dinov2":
            self._dinov2 = load_dinov2()
            self._state_mean = None
            self._state_std = None
        else:
            self._dinov2 = None
            self._state_mean, self._state_std = self._get_state_norm(norm_stats)
        self._task_buffers = self._load_task_buffers(self._demos_dir)
        self._active_task_index: int | None = None
        self._single_task_index = next(iter(self._task_buffers)) if len(self._task_buffers) == 1 else None
        self._max_dist = self._load_max_distance(self._demos_dir, max_distance_path)
        if retrieval_mode not in {"nearest", "random"}:
            raise ValueError(f"Unsupported LIBERO RICL retrieval_mode={retrieval_mode!r}.")
        self._retrieval_mode = retrieval_mode
        self._retrieval_rng = np.random.default_rng(retrieval_seed)
        self._last_retrieval_trace: list[dict[str, Any]] = []
        self._last_retrieval_timings: dict[str, float] = {}

    @override
    def infer(self, obs: dict, debug: bool = False) -> dict:  # type: ignore[misc]
        del debug
        total_started = time.perf_counter()
        inputs = jax.tree.map(lambda x: x, self.retrieve(obs))
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        policy_started = time.perf_counter()
        outputs = {
            "query_state": inputs["query_state"],
            "query_actions": self._sample_actions(
                sample_rng,
                _model.RiclObservation.from_dict(inputs, num_retrieved_observations=self._knn_k),
                **self._sample_kwargs,
            ),
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        policy_ms = (time.perf_counter() - policy_started) * 1000
        final_outputs = self._output_transform(outputs)
        final_outputs["retrieval_trace"] = getattr(self, "_last_retrieval_trace", [])
        final_outputs["timings_ms"] = {
            **getattr(self, "_last_retrieval_timings", {}),
            "policy_inference": policy_ms,
            "request_total": (time.perf_counter() - total_started) * 1000,
        }
        return final_outputs

    def retrieve(self, obs: dict) -> dict:
        task_index = self._get_task_index(obs)
        if task_index not in self._task_buffers:
            raise ValueError(f"No LIBERO retrieval demos found for task_index={task_index}.")

        query_state = np.asarray(
            self._get_obs_value(obs, "query_state", "observation/state", "state"), dtype=np.float32
        )
        query_image = self._get_obs_value(obs, "query_image", "observation/image", "image")
        query_wrist_image = self._get_obs_value(obs, "query_wrist_image", "observation/wrist_image", "wrist_image")
        query_prompt = self._get_prompt(obs, task_index)

        interpolation_distances = None
        if getattr(self, "_comparative_retriever", None) is not None:
            result = self._comparative_retriever.retrieve(
                task_id=task_index,
                image=query_image,
                wrist_image=query_wrist_image,
                state=query_state,
                prompt=query_prompt,
                k=self._knn_k,
            )
            selected = result.frames
            interpolation_distances = result.interpolation_distances
            self._last_retrieval_trace = list(result.trace)
            self._last_retrieval_timings = dict(result.timings_ms)
        else:
            task_buffer = self._ensure_task_buffer_loaded(task_index)
            candidates = task_buffer["frames"]
            candidate_vectors = task_buffer["candidate_vectors"]
            query_vector = self._get_query_retrieval_vector(query_state, query_image, query_wrist_image)
            selected, selected_candidate_indices = self._select_retrieval_frames(
                candidates, candidate_vectors, query_vector
            )
            self._last_retrieval_trace = []
            self._last_retrieval_timings = {}

        data = {"inference_time": True}
        for retrieved_idx, retrieved_frame in enumerate(selected):
            prefix = f"retrieved_{retrieved_idx}_"
            episode = retrieved_frame["episode"]
            step_idx = retrieved_frame["step_idx"]
            data[f"{prefix}state"] = np.asarray(episode["state"][step_idx], dtype=np.float32)
            data[f"{prefix}image"] = episode["image"][step_idx]
            data[f"{prefix}wrist_image"] = episode["wrist_image"][step_idx]
            data[f"{prefix}actions"] = get_libero_action_chunk(episode["actions"], step_idx, self._action_horizon)
            data[f"{prefix}prompt"] = episode["prompt"]

        data["query_state"] = query_state
        data["query_image"] = query_image
        data["query_wrist_image"] = query_wrist_image
        data["query_prompt"] = query_prompt

        if self._use_action_interpolation:
            if interpolation_distances is None:
                selected_vectors = candidate_vectors[selected_candidate_indices]
                first_vector = selected_vectors[0]
                distances = [np.linalg.norm(vector - first_vector) for vector in selected_vectors]
                distances.append(np.linalg.norm(query_vector - first_vector))
                distances = np.asarray(distances, dtype=np.float32)
            else:
                distances = interpolation_distances
            distances = distances / max(self._max_dist, 1e-6)
            data["exp_lamda_distances"] = np.exp(-self._lamda * distances).reshape(-1, 1).astype(np.float32)

        return data

    def _select_retrieval_frames(
        self,
        candidates: Sequence[dict[str, Any]],
        candidate_vectors: np.ndarray,
        query_vector: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        if self._retrieval_mode == "nearest":
            query_distances = np.linalg.norm(candidate_vectors - query_vector, axis=-1)
            order = np.argsort(query_distances, kind="stable")
            if len(order) < self._knn_k:
                order = np.tile(order, int(np.ceil(self._knn_k / len(order))))
            selected_candidate_indices = order[: self._knn_k]
            return [
                candidates[int(candidate_idx)] for candidate_idx in selected_candidate_indices
            ], selected_candidate_indices

        replace = len(candidates) < self._knn_k
        indices = self._retrieval_rng.choice(len(candidates), size=self._knn_k, replace=replace)
        return [candidates[int(candidate_idx)] for candidate_idx in indices], indices

    def _load_task_buffers(self, demos_dir: pathlib.Path) -> dict[int, dict[str, Any]]:
        if not demos_dir.exists():
            raise FileNotFoundError(f"LIBERO retrieval demos_dir does not exist: {demos_dir}")

        task_buffers = {}
        for task_dir in sorted(path for path in demos_dir.iterdir() if path.is_dir() and path.name.startswith("task_")):
            task_index = int(task_dir.name.split("_")[-1])
            episode_paths = sorted(task_dir.glob("*.npz"))
            if not episode_paths:
                raise FileNotFoundError(f"No .npz retrieval episodes found in {task_dir}")
            task_buffers[task_index] = {
                "task_dir": task_dir,
                "episode_paths": episode_paths,
                "episodes": None,
                "frames": None,
                "candidate_vectors": None,
            }

        if not task_buffers:
            raise FileNotFoundError(f"No task_XX LIBERO retrieval folders found in {demos_dir}")
        return task_buffers

    def _ensure_task_buffer_loaded(self, task_index: int) -> dict[str, Any]:
        task_buffer = self._task_buffers[task_index]
        if task_buffer["frames"] is not None:
            return task_buffer

        active_task_index = getattr(self, "_active_task_index", None)
        if active_task_index is not None and active_task_index != task_index:
            active_buffer = self._task_buffers.get(active_task_index)
            if active_buffer is not None:
                active_buffer["episodes"] = None
                active_buffer["frames"] = None
                active_buffer["candidate_vectors"] = None
                gc.collect()

        episodes = [self._load_episode(path) for path in task_buffer["episode_paths"]]
        frames = []
        vector_chunks = []
        for episode in episodes:
            retrieval_vectors = self._episode_retrieval_vectors(episode).astype(np.float32, copy=False)
            vector_chunks.append(retrieval_vectors)
            frames.extend(
                [
                    {
                        "episode": episode,
                        "step_idx": step_idx,
                    }
                    for step_idx in range(len(episode["state"]))
                ]
            )
        candidate_vectors = np.concatenate(vector_chunks, axis=0)
        if len(frames) != len(candidate_vectors):
            raise ValueError(
                f"Task {task_index} has mismatched retrieval frames and vectors: "
                f"{len(frames)} frames vs {len(candidate_vectors)} vectors."
            )

        for episode in episodes:
            for embedding_key in ("image_embeddings", "wrist_image_embeddings"):
                episode.pop(embedding_key, None)

        task_buffer["episodes"] = episodes
        task_buffer["frames"] = frames
        task_buffer["candidate_vectors"] = candidate_vectors
        self._active_task_index = task_index
        return task_buffer

    def _load_episode(self, path: pathlib.Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=True) as episode:
            prompt = str(episode["prompt"].item()) if "prompt" in episode else ""
            data = {
                "image": episode["image"],
                "wrist_image": episode["wrist_image"],
                "state": episode["state"].astype(np.float32),
                "actions": episode["actions"].astype(np.float32),
                "prompt": prompt,
            }
            for embedding_key in ("image_embeddings", "wrist_image_embeddings"):
                if embedding_key in episode:
                    data[embedding_key] = episode[embedding_key].astype(np.float32)
            return data

    def _load_retrieval_metadata(self, demos_dir: pathlib.Path) -> dict[str, Any]:
        metadata_path = demos_dir / DEFAULT_RETRIEVAL_METADATA_FILE
        if not metadata_path.exists():
            return {"backend": "state", "retrieval_key": "state", "metric": "l2"}

        metadata = json.loads(metadata_path.read_text())
        retrieval_key = metadata.get("retrieval_key", "state")
        backend = metadata.get("backend", "state")
        if retrieval_key not in {"image", "wrist_image", "both", "state"}:
            raise ValueError(f"Unsupported LIBERO retrieval key in {metadata_path}: {retrieval_key}")
        if backend not in {"dinov2", "state"}:
            raise ValueError(f"Unsupported LIBERO retrieval backend in {metadata_path}: {backend}")
        if backend == "dinov2" and retrieval_key == "state":
            raise ValueError(f"{metadata_path} uses backend=dinov2 but retrieval_key=state.")
        return {**metadata, "retrieval_key": retrieval_key, "backend": backend}

    def _episode_retrieval_vectors(self, episode: dict[str, Any]) -> np.ndarray:
        if getattr(self, "_retrieval_backend", "state") == "state":
            return self._normalize_state(episode["state"])

        embedding_keys = self._embedding_keys_for_retrieval_key(self._retrieval_key)
        missing_keys = [embedding_key for embedding_key in embedding_keys if embedding_key not in episode]
        if missing_keys:
            raise ValueError(f"LIBERO retrieval episode is missing required embedding keys: {missing_keys}")

        embeddings = [episode[embedding_key].astype(np.float32) for embedding_key in embedding_keys]
        if len(embeddings) == 1:
            return embeddings[0]
        return np.concatenate(embeddings, axis=1)

    def _get_query_retrieval_vector(
        self,
        query_state: np.ndarray,
        query_image: np.ndarray,
        query_wrist_image: np.ndarray,
    ) -> np.ndarray:
        if self._retrieval_backend == "state":
            return self._normalize_state(query_state)
        if self._retrieval_key == "image":
            return self._embed_query_image(query_image)
        if self._retrieval_key == "wrist_image":
            return self._embed_query_image(query_wrist_image)
        if self._retrieval_key == "both":
            return np.concatenate(
                [
                    self._embed_query_image(query_image),
                    self._embed_query_image(query_wrist_image),
                ],
                axis=0,
            )
        raise ValueError(f"Unsupported LIBERO retrieval key: {self._retrieval_key}")

    def _embed_query_image(self, image: np.ndarray) -> np.ndarray:
        if self._dinov2 is None:
            raise ValueError("DINOv2 model is not loaded for image retrieval.")
        embedding = embed(self._to_image_uint8(image), self._dinov2, embedding_type=self._embedding_type)
        return embedding.reshape(1, -1)[0].astype(np.float32)

    def _embedding_keys_for_retrieval_key(self, retrieval_key: str) -> list[str]:
        if retrieval_key == "image":
            return ["image_embeddings"]
        if retrieval_key == "wrist_image":
            return ["wrist_image_embeddings"]
        if retrieval_key == "both":
            return ["image_embeddings", "wrist_image_embeddings"]
        return []

    def _load_max_distance(self, demos_dir: pathlib.Path, max_distance_path: str | pathlib.Path | None = None) -> float:
        candidate_paths = [
            pathlib.Path(max_distance_path) if max_distance_path is not None else None,
            demos_dir / "max_distance.json",
        ]
        for candidate_path in candidate_paths:
            if candidate_path is not None and candidate_path.exists():
                with candidate_path.open() as f:
                    return max(float(json.load(f)["distances"]["max"]), 1e-6)
        raise FileNotFoundError(
            "Missing LIBERO max distance. Expected checkpoint assets max_distance.json "
            f"or {demos_dir / 'max_distance.json'}."
        )

    def _get_state_norm(self, norm_stats: dict[str, _transforms.NormStats] | None) -> tuple[np.ndarray, np.ndarray]:
        if norm_stats is None:
            raise ValueError("RiclLiberoPolicy requires norm_stats for state-vector retrieval.")
        stats = norm_stats.get("query_state") or norm_stats.get("retrieved_0_state")
        if stats is None:
            raise ValueError("RiclLiberoPolicy norm_stats must contain query_state or retrieved_0_state.")
        return np.asarray(stats.mean, dtype=np.float32), np.asarray(stats.std, dtype=np.float32)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self._state_mean is None or self._state_std is None:
            raise ValueError("State normalization stats are required for state-vector LIBERO retrieval.")
        return (np.asarray(state, dtype=np.float32) - self._state_mean) / (self._state_std + 1e-6)

    def _get_task_index(self, obs: dict) -> int:
        if "task_index" in obs:
            return int(np.asarray(obs["task_index"]).item())
        if self._single_task_index is not None:
            return int(self._single_task_index)
        raise ValueError("LIBERO RICL inference requires task_index when demos_dir contains multiple task folders.")

    def _get_prompt(self, obs: dict, task_index: int) -> str:
        for key in ("query_prompt", "prompt"):
            if key in obs:
                prompt = obs[key]
                return prompt if isinstance(prompt, str) else str(np.asarray(prompt).item())
        first_episode = self._ensure_task_buffer_loaded(task_index)["episodes"][0]
        return first_episode["prompt"]

    def _get_obs_value(self, obs: dict, *keys: str):
        for key in keys:
            if key in obs:
                return obs[key]
        raise KeyError(f"Expected one of {keys} in LIBERO RICL inference observation.")

    def _to_image_uint8(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        return image

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
