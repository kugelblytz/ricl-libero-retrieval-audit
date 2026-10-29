from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

DEFAULT_PROMPTS = {
    20: "turn on the stove",
    21: "turn on the stove and put the frying pan on it",
    22: "close the bottom drawer of the cabinet",
    23: "close the bottom drawer of the cabinet and open the top drawer",
    24: "put the black bowl in the bottom drawer of the cabinet",
    25: "put the black bowl on top of the cabinet",
    26: "put the wine bottle in the bottom drawer of the cabinet",
    27: "put the wine bottle on the wine rack",
    28: "close the top drawer of the cabinet",
    29: "put the black bowl in the top drawer of the cabinet",
}


def parse_mapping(values: list[str]) -> dict[int, int]:
    mapping = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected TARGET:SOURCE mapping, got {value!r}.")
        target, source = value.split(":", 1)
        mapping[int(target)] = int(source)
    return mapping


def load_prompt(source_dir: Path, task_id: int) -> str:
    task_dir = source_dir / f"task_{task_id}"
    episode_path = next(iter(sorted(task_dir.glob("episode_*.npz"))), None)
    if episode_path is None:
        raise FileNotFoundError(f"No episode_*.npz files found in {task_dir}.")
    with np.load(episode_path, allow_pickle=True) as episode:
        if "prompt" not in episode:
            return DEFAULT_PROMPTS[task_id]
        return str(episode["prompt"].item())


def rewrite_episode(source_path: Path, target_path: Path, *, target_task: int, target_prompt: str, compressed: bool) -> None:
    with np.load(source_path, allow_pickle=True) as episode:
        payload = {key: episode[key] for key in episode.files}
    payload["task_index"] = np.asarray(target_task, dtype=np.int32)
    payload["prompt"] = np.asarray(target_prompt)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(target_path, **payload)


def copy_metadata(source_dir: Path, output_dir: Path, mapping: dict[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("max_distance.json",):
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, output_dir / filename)

    metadata_path = source_dir / "retrieval_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata["mismatched_retrieval"] = True
    metadata["target_to_source_task"] = {str(target): source for target, source in sorted(mapping.items())}
    metadata["task_indices"] = sorted(mapping)
    (output_dir / "retrieval_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LIBERO retrieval DB with intentionally mismatched task demos.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Source retrieval DB, e.g. .../retrieval.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output mismatched retrieval DB.")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="TARGET:SOURCE",
        help="Serve SOURCE demos when eval asks for TARGET, while relabeling prompts/task ids as TARGET.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit copied episodes per target task.")
    parser.add_argument("--compressed", action="store_true", help="Write compressed NPZ files. Slower but smaller.")
    args = parser.parse_args()

    mapping = parse_mapping(args.map)
    if not mapping:
        raise ValueError("At least one --map TARGET:SOURCE entry is required.")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    manifest = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "target_to_source_task": {},
    }
    copy_metadata(args.source_dir, args.output_dir, mapping)

    for target_task, source_task in sorted(mapping.items()):
        source_task_dir = args.source_dir / f"task_{source_task}"
        if not source_task_dir.exists():
            raise FileNotFoundError(f"Source task directory not found: {source_task_dir}")
        target_prompt = DEFAULT_PROMPTS.get(target_task) or load_prompt(args.source_dir, target_task)
        source_prompt = load_prompt(args.source_dir, source_task)
        target_task_dir = args.output_dir / f"task_{target_task}"
        episode_paths = sorted(source_task_dir.glob("episode_*.npz"))
        if args.max_episodes is not None:
            episode_paths = episode_paths[: args.max_episodes]
        for episode_path in episode_paths:
            rewrite_episode(
                episode_path,
                target_task_dir / episode_path.name,
                target_task=target_task,
                target_prompt=target_prompt,
                compressed=args.compressed,
            )
        manifest["target_to_source_task"][str(target_task)] = {
            "source_task": source_task,
            "target_prompt": target_prompt,
            "source_prompt": source_prompt,
            "num_episodes": len(episode_paths),
        }

    (args.output_dir / "mismatch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote mismatched retrieval DB to {args.output_dir}")


if __name__ == "__main__":
    main()
