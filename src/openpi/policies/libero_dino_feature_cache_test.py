from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.policies import libero_dino_feature_cache as cache


def write_episode(path, task_id: int, episode_id: int, features: np.ndarray) -> None:
    np.savez(
        path,
        state=np.zeros((len(features), 8), dtype=np.float32),
        frame_indices=np.arange(3, 3 + len(features), dtype=np.int32),
        task_index=np.asarray(task_id, dtype=np.int32),
        episode_index=np.asarray(episode_id, dtype=np.int32),
        image_embeddings=np.asarray(features, dtype=np.float32),
    )


def test_build_cache_writes_ordered_memory_mappable_matrix(tmp_path) -> None:
    demos = tmp_path / "demos"
    task_20 = demos / "task_20"
    task_21 = demos / "task_21"
    task_20.mkdir(parents=True)
    task_21.mkdir(parents=True)
    write_episode(task_20 / "episode_2.npz", 20, 2, np.asarray([[2, 3], [4, 5]], dtype=np.float32))
    write_episode(task_21 / "episode_1.npz", 21, 1, np.asarray([[6, 7]], dtype=np.float32))

    manifest = cache.build_cache(demos)
    feature_path, manifest_path = cache.cache_paths(demos)
    features = np.load(feature_path, mmap_mode="r")

    assert isinstance(features, np.memmap)
    np.testing.assert_array_equal(features, [[2, 3], [4, 5], [6, 7]])
    assert manifest["shape"] == [3, 2]
    assert manifest["frame_order_sha256"] == cache.frame_order_sha256(
        [(20, 2, 3), (20, 2, 4), (21, 1, 3)]
    )
    assert json.loads(manifest_path.read_text()) == manifest
    with pytest.raises(FileExistsError, match="already exists"):
        cache.build_cache(demos)


def test_build_cache_rejects_inconsistent_feature_dimensions(tmp_path) -> None:
    demos = tmp_path / "demos"
    task_dir = demos / "task_20"
    task_dir.mkdir(parents=True)
    write_episode(task_dir / "episode_1.npz", 20, 1, np.zeros((1, 2), dtype=np.float32))
    write_episode(task_dir / "episode_2.npz", 20, 2, np.zeros((1, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="dimension changed"):
        cache.build_cache(demos)
