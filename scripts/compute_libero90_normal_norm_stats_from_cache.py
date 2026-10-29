"""Compute normal LIBERO-90 norm stats from the shared RICL episode cache."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import tyro

from openpi.training.ricl_libero_dataset import get_libero_action_chunk


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main(
    episodes_dir: pathlib.Path = pathlib.Path("ricl_libero_preprocessing/libero90_train20x20/episodes"),
    output_dir: pathlib.Path = pathlib.Path("assets/pi0_fast_libero90_normal/libero"),
    action_horizon: int = 10,
) -> None:
    episode_paths = sorted(episodes_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {episodes_dir}")

    states = []
    action_chunks = []
    for episode_path in episode_paths:
        with np.load(episode_path, allow_pickle=False) as episode:
            episode_states = np.asarray(episode["state"], dtype=np.float32)
            episode_actions = np.asarray(episode["actions"], dtype=np.float32)
            frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
            states.append(episode_states[: len(frame_indices)])
            action_chunks.extend(
                get_libero_action_chunk(episode_actions, step_idx, action_horizon)
                for step_idx in range(len(frame_indices))
            )

    norm_stats = {
        "norm_stats": {
            "state": _stats(np.concatenate(states, axis=0)),
            "actions": _stats(np.concatenate(action_chunks, axis=0)),
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    output_path.write_text(json.dumps(norm_stats, indent=2) + "\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
