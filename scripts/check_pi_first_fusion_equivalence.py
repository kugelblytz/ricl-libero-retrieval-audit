from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies.libero_runtime_retrieval import Pi0FirstFusionQueryEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--config", default="pi0_fast_libero90_normal")
    parser.add_argument("--task-ids", default="20,21,24,25,28,29")
    parser.add_argument("--max-abs-tolerance", type=float, default=1e-5)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    task_ids = [int(value) for value in args.task_ids.split(",")]
    task_id = task_ids[0]
    episode_path = sorted(
        (args.pool_dir / f"task_{task_id}").glob("episode_*.npz"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )[0]
    with np.load(episode_path, allow_pickle=True) as episode:
        image = episode["image"][0]
        wrist_image = episode["wrist_image"][0]
        state = episode["state"][0]
        prompt = str(episode["prompt"].item())

    task_features = np.load(args.feature_dir / f"task_{task_id:02d}.npy", mmap_mode="r")
    expected = np.asarray(task_features[0], dtype=np.float32)
    encoder = Pi0FirstFusionQueryEmbedder(args.config, args.checkpoint_dir, "float32")
    actual = encoder(image, wrist_image, state, prompt)
    max_abs = float(np.max(np.abs(actual - expected)))
    cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))

    task_expected_top4 = np.argsort(np.linalg.norm(task_features - expected, axis=1), kind="stable")[:4]
    task_actual_top4 = np.argsort(np.linalg.norm(task_features - actual, axis=1), kind="stable")[:4]
    all_features = np.concatenate(
        [np.load(args.feature_dir / f"task_{candidate_task:02d}.npy", mmap_mode="r") for candidate_task in task_ids]
    )
    global_expected_top4 = np.argsort(np.linalg.norm(all_features - expected, axis=1), kind="stable")[:4]
    global_actual_top4 = np.argsort(np.linalg.norm(all_features - actual, axis=1), kind="stable")[:4]
    payload = {
        "task_id": task_id,
        "episode_path": str(episode_path.resolve()),
        "max_abs": max_abs,
        "cosine": cosine,
        "task_top4_equal": bool(np.array_equal(task_expected_top4, task_actual_top4)),
        "global_top4_equal": bool(np.array_equal(global_expected_top4, global_actual_top4)),
        "task_top4": task_actual_top4.tolist(),
        "global_top4": global_actual_top4.tolist(),
    }
    if max_abs > args.max_abs_tolerance:
        raise SystemExit(f"Pi feature max_abs {max_abs} exceeds {args.max_abs_tolerance}")
    if cosine < args.min_cosine:
        raise SystemExit(f"Pi feature cosine {cosine} is below {args.min_cosine}")
    if not payload["task_top4_equal"] or not payload["global_top4_equal"]:
        raise SystemExit(f"Pi pruned/full-cache top-4 mismatch: {payload}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
