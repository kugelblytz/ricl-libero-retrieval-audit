from __future__ import annotations

from collections.abc import Iterable
import datetime
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

CACHE_DIRNAME = "dino_feature_cache"
FEATURES_FILENAME = "features.npy"
MANIFEST_FILENAME = "manifest.json"
CACHE_VERSION = 1


def cache_paths(demos_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    cache_dir = demos_dir / CACHE_DIRNAME
    return cache_dir / FEATURES_FILENAME, cache_dir / MANIFEST_FILENAME


def frame_order_sha256(rows: Iterable[tuple[int, int, int]]) -> str:
    digest = hashlib.sha256()
    for task_id, episode_id, frame_index in rows:
        digest.update(f"{task_id}\0{episode_id}\0{frame_index}\n".encode())
    return digest.hexdigest()


def build_cache(demos_dir: pathlib.Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Consolidate per-episode DINO features into one exact, memory-mappable matrix."""
    demos_dir = demos_dir.resolve()
    feature_path, manifest_path = cache_paths(demos_dir)
    if (feature_path.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"DINO feature cache already exists under {feature_path.parent}; pass overwrite=True to replace it."
        )

    episode_paths = sorted(
        demos_dir.glob("task_*/episode_*.npz"),
        key=lambda path: (
            int(path.parent.name.split("_")[-1]),
            int(path.stem.split("_")[-1]),
        ),
    )
    if not episode_paths:
        raise FileNotFoundError(f"No task_*/episode_*.npz files found in {demos_dir}.")

    episode_metadata: list[tuple[pathlib.Path, int, int, np.ndarray, int]] = []
    feature_dim: int | None = None
    frame_count = 0
    order_rows: list[tuple[int, int, int]] = []
    for episode_path in episode_paths:
        task_id = int(episode_path.parent.name.split("_")[-1])
        episode_id = int(episode_path.stem.split("_")[-1])
        with np.load(episode_path, allow_pickle=True) as episode:
            if "image_embeddings" not in episode:
                raise KeyError(f"Missing image_embeddings in {episode_path}.")
            embeddings = episode["image_embeddings"]
            frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
            if embeddings.ndim != 2:
                raise ValueError(f"Expected 2D image_embeddings in {episode_path}, got {embeddings.shape}.")
            if len(embeddings) != len(frame_indices):
                raise ValueError(f"Embedding/frame count mismatch in {episode_path}.")
            if feature_dim is None:
                feature_dim = int(embeddings.shape[1])
            elif embeddings.shape[1] != feature_dim:
                raise ValueError(
                    f"DINO feature dimension changed in {episode_path}: {embeddings.shape[1]} != {feature_dim}."
                )
        episode_metadata.append((episode_path, task_id, episode_id, frame_indices, len(frame_indices)))
        order_rows.extend((task_id, episode_id, int(frame_index)) for frame_index in frame_indices)
        frame_count += len(frame_indices)

    assert feature_dim is not None
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_feature_path = feature_path.with_name(f".{feature_path.name}.tmp")
    temporary_manifest_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if temporary_feature_path.exists():
        temporary_feature_path.unlink()
    if temporary_manifest_path.exists():
        temporary_manifest_path.unlink()

    destination = np.lib.format.open_memmap(
        temporary_feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, feature_dim),
    )
    offset = 0
    try:
        for episode_path, _, _, _, episode_frame_count in episode_metadata:
            with np.load(episode_path, allow_pickle=True) as episode:
                embeddings = np.asarray(episode["image_embeddings"], dtype=np.float32)
            destination[offset : offset + episode_frame_count] = embeddings
            offset += episode_frame_count
        destination.flush()
    finally:
        del destination

    manifest: dict[str, Any] = {
        "version": CACHE_VERSION,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "source_dir": str(demos_dir),
        "features_file": FEATURES_FILENAME,
        "dtype": "float32",
        "shape": [frame_count, feature_dim],
        "frame_count": frame_count,
        "feature_dim": feature_dim,
        "episode_count": len(episode_paths),
        "frame_order_sha256": frame_order_sha256(order_rows),
    }
    temporary_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_feature_path.replace(feature_path)
    temporary_manifest_path.replace(manifest_path)
    return manifest
