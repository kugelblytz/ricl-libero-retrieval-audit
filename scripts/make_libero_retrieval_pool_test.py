from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import make_libero_retrieval_pool as pool_builder


def make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "retrieval_metadata.json").write_text(
        json.dumps(
            {
                "backend": "dinov2",
                "task_indices": [20, 21, 22],
                "index_file": "stale_global_index.npz",
                "selected_episode_indices_by_task": {
                    "20": [0, 1],
                    "21": [2, 3],
                    "22": [4, 5],
                },
            }
        )
    )
    (source / "max_distance.json").write_text('{"distances": {"max": 1.0}}\n')
    for task_id in (20, 21, 22):
        task_dir = source / f"task_{task_id}"
        task_dir.mkdir()
        for episode_id in range(2):
            np.savez(task_dir / f"episode_{episode_id}.npz", task_index=np.asarray(task_id))
    return source


def test_build_pool_filters_tasks_and_metadata(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "output"

    manifest = pool_builder.build_pool(
        source,
        output,
        (20, 22),
        materialization="hardlink",
        overwrite=False,
        max_episodes_per_task=None,
    )

    assert manifest["task_ids"] == [20, 22]
    assert manifest["num_tasks"] == 2
    assert manifest["num_episodes"] == 4
    assert not (output / "task_21").exists()
    assert len(list((output / "task_20").glob("episode_*.npz"))) == 2
    assert (source / "task_20" / "episode_0.npz").stat().st_ino == (
        output / "task_20" / "episode_0.npz"
    ).stat().st_ino

    metadata = json.loads((output / "retrieval_metadata.json").read_text())
    assert metadata["task_indices"] == [20, 22]
    assert metadata["num_retrieval_tasks"] == 2
    assert metadata["num_retrieval_episodes"] == 4
    assert metadata["selected_episode_indices_by_task"] == {"20": [0, 1], "22": [4, 5]}
    assert metadata["max_distance_file"] == str(output / "max_distance.json")
    assert "index_file" not in metadata


def test_parse_task_ids_supports_ranges_and_rejects_duplicates() -> None:
    assert pool_builder.parse_task_ids("20-21,24,29") == (20, 21, 24, 29)
    with pytest.raises(ValueError, match="unique"):
        pool_builder.parse_task_ids("20,20")


def test_max_episodes_uses_numeric_episode_order(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    task_dir = source / "task_20"
    np.savez(task_dir / "episode_10.npz", task_index=np.asarray(20))
    output = tmp_path / "output"

    manifest = pool_builder.build_pool(
        source,
        output,
        (20,),
        materialization="hardlink",
        overwrite=False,
        max_episodes_per_task=2,
    )

    assert manifest["tasks"]["20"]["episode_files"] == ["episode_0.npz", "episode_1.npz"]
