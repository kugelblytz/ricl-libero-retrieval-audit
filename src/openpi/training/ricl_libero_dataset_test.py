# ruff: noqa: SLF001

import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training.ricl_libero_dataset import RiclLiberoDataset
from openpi.training.ricl_libero_dataset import build_libero_embedding_retrieval_indices
from openpi.training.ricl_libero_dataset import build_libero_state_retrieval_indices
from openpi.training.ricl_libero_dataset import get_libero_action_chunk


def _load_setup_ricl_libero_module():
    script_path = pathlib.Path(__file__).parents[3] / "scripts" / "setup_ricl_libero.py"
    spec = importlib.util.spec_from_file_location("setup_ricl_libero", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_ricl_libero = _load_setup_ricl_libero_module()


def _episode(task_index: int, state_offset: float) -> dict:
    return {
        "task_index": task_index,
        "state": np.asarray(
            [
                np.full((8,), state_offset, dtype=np.float32),
                np.full((8,), state_offset + 1, dtype=np.float32),
                np.full((8,), state_offset + 2, dtype=np.float32),
            ]
        ),
        "frame_indices": np.asarray([0, 1, 2], dtype=np.int32),
    }


def _embedding_episode(task_index: int, values: list[float]) -> dict:
    return {
        "task_index": task_index,
        "embeddings": np.asarray([[value, 0.0] for value in values], dtype=np.float32),
        "frame_indices": np.arange(len(values), dtype=np.int32),
    }


def _write_episode(path, ep_idx: int, task_index: int, state_offset: float) -> None:
    episode = _episode(task_index, state_offset)
    image = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    wrist_image = np.ones((3, 4, 4, 3), dtype=np.uint8)
    actions = np.arange(21, dtype=np.float32).reshape(3, 7) + state_offset
    np.savez(
        path / f"episode_{ep_idx}.npz",
        image=image,
        wrist_image=wrist_image,
        state=episode["state"],
        actions=actions,
        frame_indices=episode["frame_indices"],
        task_index=np.asarray(task_index, dtype=np.int32),
        prompt=np.asarray(f"task {task_index}"),
    )


def test_get_libero_action_chunk_repeats_last_action():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    full_chunk = get_libero_action_chunk(actions, step_idx=0, action_horizon=2)
    padded_chunk = get_libero_action_chunk(actions, step_idx=2, action_horizon=3)

    assert full_chunk.shape == (2, 7)
    np.testing.assert_allclose(full_chunk, actions[:2])
    assert padded_chunk.shape == (3, 7)
    np.testing.assert_allclose(padded_chunk[0], actions[-1])
    np.testing.assert_allclose(padded_chunk[1], actions[-1])
    np.testing.assert_allclose(padded_chunk[2], actions[-1])


def test_build_libero_state_retrieval_indices_same_task_and_cross_episode(tmp_path):
    episodes = {
        0: _episode(task_index=0, state_offset=0),
        1: _episode(task_index=0, state_offset=10),
        2: _episode(task_index=1, state_offset=20),
        3: _episode(task_index=1, state_offset=30),
    }
    query_indices, retrieved_indices, distances = build_libero_state_retrieval_indices(
        episodes,
        num_retrieved_observations=2,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
    )

    assert query_indices.shape == (12, 2)
    assert retrieved_indices.shape == (12, 2, 2)
    assert distances.shape == (12, 3)

    episode_tasks = {ep_idx: episode["task_index"] for ep_idx, episode in episodes.items()}
    for query, retrieved in zip(query_indices, retrieved_indices, strict=True):
        query_ep = int(query[0])
        for retrieved_ep, _ in retrieved:
            assert int(retrieved_ep) != query_ep
            assert episode_tasks[int(retrieved_ep)] == episode_tasks[query_ep]

    np.savez(
        tmp_path / "indices.npz", query_indices=query_indices, retrieved_indices=retrieved_indices, distances=distances
    )
    reloaded = np.load(tmp_path / "indices.npz")
    assert reloaded["query_indices"].shape == (12, 2)
    assert reloaded["retrieved_indices"].shape == (12, 2, 2)
    assert reloaded["distances"].shape == (12, 3)


def test_build_libero_embedding_retrieval_indices_same_task_cross_episode_and_repeat():
    episodes = {
        0: _embedding_episode(task_index=0, values=[0.0, 10.0]),
        1: _embedding_episode(task_index=0, values=[1.0, 20.0]),
        2: _embedding_episode(task_index=1, values=[100.0]),
        3: _embedding_episode(task_index=1, values=[101.0]),
    }
    query_indices, retrieved_indices, distances = build_libero_embedding_retrieval_indices(
        episodes,
        num_retrieved_observations=3,
        use_faiss=False,
    )

    assert query_indices.shape == (6, 2)
    assert retrieved_indices.shape == (6, 3, 2)
    assert distances.shape == (6, 4)
    np.testing.assert_allclose(distances[:, 0], 0.0)

    episode_tasks = {ep_idx: episode["task_index"] for ep_idx, episode in episodes.items()}
    for query, retrieved in zip(query_indices, retrieved_indices, strict=True):
        query_ep = int(query[0])
        for retrieved_ep, _ in retrieved:
            assert int(retrieved_ep) != query_ep
            assert episode_tasks[int(retrieved_ep)] == episode_tasks[query_ep]

    task_one_query = np.where(query_indices[:, 0] == 2)[0][0]
    assert np.all(retrieved_indices[task_one_query, :, 0] == 3)


@pytest.mark.parametrize(
    ("retrieval_key", "expected_keys", "expected_dim"),
    [
        ("image", ["image_embeddings"], 768),
        ("wrist_image", ["wrist_image_embeddings"], 768),
        ("both", ["image_embeddings", "wrist_image_embeddings"], 1536),
    ],
)
def test_setup_ricl_libero_flush_episode_writes_embedding_cache_and_metadata(
    tmp_path,
    monkeypatch,
    retrieval_key,
    expected_keys,
    expected_dim,
):
    def fake_embed_with_batches(images, dinov2, batch_size, embedding_type):
        del dinov2, batch_size, embedding_type
        embeddings = np.zeros((len(images), 768), dtype=np.float32)
        embeddings[:, 0] = images[:, 0, 0, 0]
        return embeddings

    monkeypatch.setattr(setup_ricl_libero, "embed_with_batches", fake_embed_with_batches)
    episode_rows = [
        {
            "episode_index": 0,
            "task_index": 0,
            "frame_index": frame_index,
            "image": np.full((4, 4, 3), frame_index + 1, dtype=np.uint8),
            "wrist_image": np.full((4, 4, 3), frame_index + 10, dtype=np.uint8),
            "state": np.full((8,), frame_index, dtype=np.float32),
            "actions": np.full((7,), frame_index, dtype=np.float32),
            "prompt": "pick up the object",
        }
        for frame_index in range(2)
    ]
    episode_index, summary = setup_ricl_libero._flush_episode(
        episode_rows,
        episodes_dir=tmp_path / "episodes",
        state_stats=setup_ricl_libero.normalize.RunningStats(),
        action_stats=setup_ricl_libero.normalize.RunningStats(),
        retrieval_key=retrieval_key,
        embedding_type="CLS",
        embedding_batch_size=1,
        overwrite_embeddings=True,
        dinov2=object(),
    )

    assert episode_index == 0
    assert summary["embeddings"].shape == (2, expected_dim)
    with np.load(tmp_path / "episodes" / "episode_0.npz") as episode:
        for key in expected_keys:
            assert episode[key].shape == (2, 768)

    setup_ricl_libero._save_retrieval_metadata(
        preprocessing_dir=tmp_path / "preprocessing",
        assets_dir=tmp_path / "assets",
        retrieval_dir=tmp_path / "retrieval",
        retrieval_key=retrieval_key,
        embedding_type="CLS",
        num_retrieved_observations=4,
    )
    for root in ("preprocessing", "assets", "retrieval"):
        assert (tmp_path / root / "retrieval_metadata.json").exists()


def test_setup_ricl_libero_episode_cache_rejects_identity_mismatch(tmp_path):
    episode_path = tmp_path / "episode_0.npz"
    cached = {
        "state": np.zeros((2, 8), dtype=np.float32),
        "frame_indices": np.asarray([0, 1], dtype=np.int32),
        "task_index": np.asarray(1, dtype=np.int32),
        "prompt": np.asarray("cached prompt"),
    }
    np.savez(episode_path, **cached)

    assert setup_ricl_libero._episode_cache_ready(episode_path, cached)
    assert not setup_ricl_libero._episode_cache_ready(
        episode_path,
        {**cached, "task_index": np.asarray(28, dtype=np.int32)},
    )
    assert not setup_ricl_libero._episode_cache_ready(
        episode_path,
        {**cached, "prompt": np.asarray("different prompt")},
    )


def test_setup_ricl_libero_select_episode_indices_caps_each_task_deterministically():
    rows = []
    for task_index in range(4):
        for episode_offset in range(5):
            episode_index = task_index * 100 + episode_offset
            for frame_index in range(2):
                rows.append((episode_index, task_index, frame_index, len(rows), task_index))

    train_selection = setup_ricl_libero._select_episode_indices(
        [row for row in rows if row[1] in {0, 1}],
        max_episodes_per_task=2,
        episode_selection="seeded",
        seed=0,
    )
    eval_selection = setup_ricl_libero._select_episode_indices(
        [row for row in rows if row[1] in {2, 3}],
        max_episodes_per_task=2,
        episode_selection="seeded",
        seed=1,
    )

    assert train_selection is not None
    assert eval_selection is not None
    assert len(train_selection) == 4
    assert len(eval_selection) == 4
    assert train_selection.isdisjoint(eval_selection)

    for task_index in range(2):
        assert sum(1 for episode_index in train_selection if episode_index // 100 == task_index) == 2
    for task_index in range(2, 4):
        assert sum(1 for episode_index in eval_selection if episode_index // 100 == task_index) == 2

    repeated_selection = setup_ricl_libero._select_episode_indices(
        [row for row in rows if row[1] in {0, 1}],
        max_episodes_per_task=2,
        episode_selection="seeded",
        seed=0,
    )
    assert repeated_selection == train_selection


def test_setup_ricl_libero_select_episode_indices_first_mode_uses_first_episodes():
    rows = [(episode_index, 0, 0, dataset_index, 0) for dataset_index, episode_index in enumerate([4, 3, 2, 1, 0])]

    selected = setup_ricl_libero._select_episode_indices(
        rows,
        max_episodes_per_task=2,
        episode_selection="first",
        seed=0,
    )

    assert selected == {4, 3}


def test_save_retrieval_metadata_can_write_retrieval_only_role(tmp_path):
    setup_ricl_libero._save_retrieval_metadata(
        preprocessing_dir=tmp_path / "preprocessing",
        assets_dir=tmp_path / "assets",
        retrieval_dir=tmp_path / "retrieval",
        retrieval_key="image",
        embedding_type="64PATCHES",
        num_retrieved_observations=4,
        artifact_role="retrieval",
        repo_id="local/libero90",
        task_indices=(20, 21),
        max_episodes_per_task=20,
        episode_selection="seeded",
        seed=1,
        max_distance_file=tmp_path / "retrieval" / "max_distance.json",
        output_dirs=(tmp_path / "retrieval",),
    )

    metadata_path = tmp_path / "retrieval" / "retrieval_metadata.json"
    assert metadata_path.exists()
    assert not (tmp_path / "assets" / "retrieval_metadata.json").exists()
    metadata = setup_ricl_libero.json.loads(metadata_path.read_text())
    assert metadata["artifact_role"] == "retrieval"
    assert metadata["task_indices"] == [20, 21]
    assert metadata["max_episodes_per_task"] == 20


def test_setup_ricl_libero_detects_local_lerobot_dataset_root(tmp_path):
    repo_id = "local/libero90"
    (tmp_path / repo_id / "meta").mkdir(parents=True)
    (tmp_path / repo_id / "meta" / "info.json").write_text("{}")

    assert setup_ricl_libero._local_lerobot_dataset_exists(repo_id, tmp_path)
    assert not setup_ricl_libero._local_lerobot_dataset_exists("local/missing", tmp_path)


def test_ricl_libero_dataset_returns_prefixed_sample(tmp_path):
    preprocessing_dir = tmp_path / "ricl_libero_preprocessing"
    episodes_dir = preprocessing_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    _write_episode(episodes_dir, ep_idx=0, task_index=0, state_offset=0)
    _write_episode(episodes_dir, ep_idx=1, task_index=0, state_offset=10)

    episodes = {
        0: _episode(task_index=0, state_offset=0),
        1: _episode(task_index=0, state_offset=10),
    }
    query_indices, retrieved_indices, distances = build_libero_state_retrieval_indices(
        episodes,
        num_retrieved_observations=4,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
    )
    np.savez(
        preprocessing_dir / "libero10_indices_and_distances.npz",
        query_indices=query_indices,
        retrieved_indices=retrieved_indices,
        distances=distances,
    )
    max_distance_path = tmp_path / "max_distance.json"
    max_distance_path.write_text('{"distances": {"max": 100.0}}')

    dataset = RiclLiberoDataset(
        SimpleNamespace(
            action_dim=7,
            action_horizon=10,
            num_retrieved_observations=4,
            use_action_interpolation=True,
            lamda=10.0,
        ),
        preprocessing_dir=preprocessing_dir,
        max_distance_path=max_distance_path,
    )
    sample = dataset[0]

    assert sample["query_state"].shape == (8,)
    assert sample["query_actions"].shape == (10, 7)
    assert sample["retrieved_0_state"].shape == (8,)
    assert sample["retrieved_0_actions"].shape == (10, 7)
    assert sample["exp_lamda_distances"].shape == (5, 1)


def test_ricl_libero_dataset_uses_metadata_index_file(tmp_path):
    preprocessing_dir = tmp_path / "ricl_libero_preprocessing"
    episodes_dir = preprocessing_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    _write_episode(episodes_dir, ep_idx=0, task_index=0, state_offset=0)
    _write_episode(episodes_dir, ep_idx=1, task_index=0, state_offset=10)

    episodes = {
        0: _episode(task_index=0, state_offset=0),
        1: _episode(task_index=0, state_offset=10),
    }
    query_indices, retrieved_indices, distances = build_libero_state_retrieval_indices(
        episodes,
        num_retrieved_observations=4,
        state_mean=np.zeros(8, dtype=np.float32),
        state_std=np.ones(8, dtype=np.float32),
    )
    np.savez(
        preprocessing_dir / "custom_indices.npz",
        query_indices=query_indices,
        retrieved_indices=retrieved_indices,
        distances=distances,
    )
    (preprocessing_dir / "retrieval_metadata.json").write_text('{"index_file": "custom_indices.npz"}')
    max_distance_path = tmp_path / "max_distance.json"
    max_distance_path.write_text('{"distances": {"max": 100.0}}')

    dataset = RiclLiberoDataset(
        SimpleNamespace(
            action_dim=7,
            action_horizon=10,
            num_retrieved_observations=4,
            use_action_interpolation=True,
            lamda=10.0,
        ),
        preprocessing_dir=preprocessing_dir,
        max_distance_path=max_distance_path,
    )

    assert dataset.indices_path == preprocessing_dir / "custom_indices.npz"
    assert int(dataset.all_query_indices[0, 0]) != int(dataset.all_retrieved_indices[0, 0, 0])


def test_build_libero_state_retrieval_indices_requires_cross_episode_candidate():
    with pytest.raises(ValueError, match="different episode"):
        build_libero_state_retrieval_indices(
            {0: _episode(task_index=0, state_offset=0)},
            num_retrieved_observations=4,
            state_mean=np.zeros(8, dtype=np.float32),
            state_std=np.ones(8, dtype=np.float32),
        )
