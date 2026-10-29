from __future__ import annotations

import json

import numpy as np

from openpi.policies import libero_dino_feature_cache
from openpi.policies import libero_runtime_retrieval as runtime


class FakePiEmbedder:
    def __init__(self, config_name, checkpoint_dir, dtype):
        del config_name, checkpoint_dir, dtype

    def __call__(self, image, wrist_image, state, prompt):
        del image, wrist_image, state, prompt
        return np.asarray([0.0, 0.0], dtype=np.float32)


def write_episode(path, task_id: int, episode_id: int) -> None:
    del episode_id
    np.savez(
        path,
        image=np.full((2, 4, 4, 3), task_id, dtype=np.uint8),
        wrist_image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        state=np.full((2, 8), task_id, dtype=np.float32),
        actions=np.full((2, 7), task_id, dtype=np.float32),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        task_index=np.asarray(task_id, dtype=np.int32),
        prompt=np.asarray(f"task {task_id}"),
    )


def test_pi_retrieval_supports_task_and_global_memory(tmp_path, monkeypatch) -> None:
    demos = tmp_path / "demos"
    features = tmp_path / "features"
    features.mkdir()
    for task_id in (20, 21):
        task_dir = demos / f"task_{task_id}"
        task_dir.mkdir(parents=True)
        write_episode(task_dir / f"episode_{task_id}.npz", task_id, task_id)

    np.save(features / "task_20.npy", np.asarray([[5.0, 0.0], [6.0, 0.0]], dtype=np.float32))
    np.save(features / "task_21.npy", np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(runtime, "Pi0FirstFusionQueryEmbedder", FakePiEmbedder)

    common = {
        "demos_dir": demos,
        "method": "pi0_first_fusion",
        "pi_feature_dir": features,
        "pi_checkpoint_dir": tmp_path,
    }
    task_retriever = runtime.LiberoRuntimeRetriever(scope="task", **common)
    global_retriever = runtime.LiberoRuntimeRetriever(scope="global", **common)
    query = {
        "task_id": 20,
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "task 20",
        "k": 1,
    }

    task_result = task_retriever.retrieve(**query)
    global_result = global_retriever.retrieve(**query)

    assert task_result.trace[0]["retrieved_task_id"] == 20
    assert global_result.trace[0]["retrieved_task_id"] == 21
    assert task_result.frames[0]["episode"]["prompt"] == "task 20"
    assert global_result.interpolation_distances.shape == (2,)


def test_pi_cache_rows_follow_numeric_episode_order(tmp_path, monkeypatch) -> None:
    demos = tmp_path / "demos"
    features = tmp_path / "features"
    task_dir = demos / "task_20"
    task_dir.mkdir(parents=True)
    features.mkdir()
    write_episode(task_dir / "episode_10.npz", 20, 10)
    write_episode(task_dir / "episode_2.npz", 20, 2)
    np.save(
        features / "task_20.npy",
        np.asarray([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(runtime, "Pi0FirstFusionQueryEmbedder", FakePiEmbedder)

    result = runtime.LiberoRuntimeRetriever(
        demos,
        method="pi0_first_fusion",
        scope="global",
        pi_feature_dir=features,
        pi_checkpoint_dir=tmp_path,
    ).retrieve(
        task_id=20,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="task 20",
        k=1,
    )

    assert result.trace[0]["episode_id"] == 2


def test_dino_state_uses_calibrated_hybrid_distance(tmp_path, monkeypatch) -> None:
    demos = tmp_path / "demos"
    task_dir = demos / "task_20"
    task_dir.mkdir(parents=True)
    np.savez(
        task_dir / "episode_20.npz",
        image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        state=np.asarray([[0.0] * 8, [10.0] * 8], dtype=np.float32),
        actions=np.zeros((2, 7), dtype=np.float32),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        task_index=np.asarray(20, dtype=np.int32),
        prompt=np.asarray("task 20"),
        image_embeddings=np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    checkpoint = tmp_path / "checkpoint"
    stats_path = checkpoint / "assets" / "libero"
    stats_path.mkdir(parents=True)
    (stats_path / "norm_stats.json").write_text(
        json.dumps({"norm_stats": {"state": {"mean": [0.0] * 8, "std": [1.0] * 8}}})
    )
    monkeypatch.setattr(runtime, "load_dinov2", lambda: object())
    monkeypatch.setattr(
        runtime,
        "embed",
        lambda image, model, embedding_type: np.asarray([[0.0, 0.0]], dtype=np.float32),
    )
    common = {
        "demos_dir": demos,
        "method": "dino_state",
        "scope": "task",
        "state_norm_checkpoint_dir": checkpoint,
        "dino_state_beta": 2.0,
        "distance_chunk_size": 1,
    }
    fallback_retriever = runtime.LiberoRuntimeRetriever(**common)
    fallback_result = fallback_retriever.retrieve(
        task_id=20,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="task 20",
        k=1,
    )

    libero_dino_feature_cache.build_cache(demos)
    retriever = runtime.LiberoRuntimeRetriever(
        **common,
    )

    result = retriever.retrieve(
        task_id=20,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="task 20",
        k=1,
    )

    assert result.trace[0]["frame_index"] == 0
    assert result.trace[0]["retrieval_method"] == "dino_state"
    assert result.trace == fallback_result.trace
    assert isinstance(retriever._features, np.memmap)  # noqa: SLF001


def test_dino_uses_image_distance_only_and_keeps_distinct_label(tmp_path, monkeypatch) -> None:
    demos = tmp_path / "demos"
    task_dir = demos / "task_20"
    task_dir.mkdir(parents=True)
    np.savez(
        task_dir / "episode_20.npz",
        image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        state=np.asarray([[100.0] * 8, [0.0] * 8], dtype=np.float32),
        actions=np.zeros((2, 7), dtype=np.float32),
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        task_index=np.asarray(20, dtype=np.int32),
        prompt=np.asarray("task 20"),
        image_embeddings=np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )
    libero_dino_feature_cache.build_cache(demos)
    monkeypatch.setattr(runtime, "load_dinov2", lambda: object())
    monkeypatch.setattr(
        runtime,
        "embed",
        lambda image, model, embedding_type: np.asarray([[0.0, 0.0]], dtype=np.float32),
    )

    retriever = runtime.LiberoRuntimeRetriever(demos, method="dino", scope="global")
    result = retriever.retrieve(
        task_id=20,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        wrist_image=np.zeros((4, 4, 3), dtype=np.uint8),
        state=np.zeros(8, dtype=np.float32),
        prompt="task 20",
        k=1,
    )

    assert result.trace[0]["frame_index"] == 0
    assert result.trace[0]["retrieval_method"] == "dino"
    assert result.trace[0]["candidate_pool_num_tasks"] == 1
    assert isinstance(retriever._features, np.memmap)  # noqa: SLF001


def test_random_global_is_query_deterministic_distinct_and_feature_free(tmp_path) -> None:
    demos = tmp_path / "demos"
    for task_id in (20, 21):
        task_dir = demos / f"task_{task_id}"
        task_dir.mkdir(parents=True)
        write_episode(task_dir / f"episode_{task_id}.npz", task_id, task_id)

    retriever = runtime.LiberoRuntimeRetriever(demos, method="random", scope="global")
    query = {
        "task_id": 20,
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
        "prompt": "task 20",
        "k": 4,
    }

    first = retriever.retrieve(**query)
    second = retriever.retrieve(**query)
    restarted = runtime.LiberoRuntimeRetriever(demos, method="random", scope="global").retrieve(**query)
    first_ids = [(item["retrieved_task_id"], item["frame_index"]) for item in first.trace]

    assert first.trace == second.trace == restarted.trace
    assert len(set(first_ids)) == 4
    assert all(item["retrieval_method"] == "random" for item in first.trace)
    assert all(item["candidate_pool_num_tasks"] == 2 for item in first.trace)
    assert first.interpolation_distances.shape == (5,)
    assert retriever._features.shape == (4, 0)  # noqa: SLF001
    assert runtime.LiberoRuntimeRetriever._query_content_seed(  # noqa: SLF001
        **{key: value for key, value in query.items() if key != "k"}
    ) != runtime.LiberoRuntimeRetriever._query_content_seed(  # noqa: SLF001
        **{**{key: value for key, value in query.items() if key != "k"}, "prompt": "changed"}
    )
