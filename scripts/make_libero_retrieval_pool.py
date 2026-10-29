from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def parse_task_ids(value: str) -> tuple[int, ...]:
    task_ids: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending task range: {part!r}")
            task_ids.extend(range(start, end + 1))
        else:
            task_ids.append(int(part))
    if not task_ids:
        raise ValueError("At least one task ID is required.")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Task IDs must be unique: {value!r}")
    return tuple(sorted(task_ids))


def materialize_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(os.path.relpath(source, start=target.parent))
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode!r}")


def build_pool(
    source_dir: Path,
    output_dir: Path,
    task_ids: tuple[int, ...],
    *,
    materialization: str,
    overwrite: bool,
    max_episodes_per_task: int | None,
) -> dict:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source retrieval directory not found: {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    task_episode_paths: dict[int, list[Path]] = {}
    for task_id in task_ids:
        task_dir = source_dir / f"task_{task_id}"
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Source task directory not found: {task_dir}")
        episode_paths = sorted(task_dir.glob("episode_*.npz"), key=lambda path: int(path.stem.split("_")[-1]))
        if max_episodes_per_task is not None:
            episode_paths = episode_paths[:max_episodes_per_task]
        if not episode_paths:
            raise FileNotFoundError(f"No episode_*.npz files found in {task_dir}")
        task_episode_paths[task_id] = episode_paths

    output_dir.mkdir(parents=True)
    copied_metadata_files = []
    for filename in ("max_distance.json",):
        source_path = source_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, output_dir / filename)
            copied_metadata_files.append(filename)

    source_metadata_path = source_dir / "retrieval_metadata.json"
    metadata = json.loads(source_metadata_path.read_text()) if source_metadata_path.exists() else {}
    selected_by_task = metadata.get("selected_episode_indices_by_task")
    metadata.pop("index_file", None)
    metadata["task_indices"] = list(task_ids)
    metadata["num_retrieval_tasks"] = len(task_ids)
    metadata["num_retrieval_episodes"] = sum(len(paths) for paths in task_episode_paths.values())
    metadata["pool_source_dir"] = str(source_dir)
    metadata["pool_materialization"] = materialization
    if "max_distance.json" in copied_metadata_files:
        metadata["max_distance_file"] = str(output_dir / "max_distance.json")
    if selected_by_task is not None:
        metadata["selected_episode_indices_by_task"] = {
            str(task_id): selected_by_task[str(task_id)]
            for task_id in task_ids
            if str(task_id) in selected_by_task
        }
    if max_episodes_per_task is not None:
        metadata["max_episodes_per_task"] = max_episodes_per_task
    (output_dir / "retrieval_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "task_ids": list(task_ids),
        "num_tasks": len(task_ids),
        "num_episodes": sum(len(paths) for paths in task_episode_paths.values()),
        "materialization": materialization,
        "tasks": {},
    }
    for task_id, episode_paths in task_episode_paths.items():
        output_task_dir = output_dir / f"task_{task_id}"
        output_task_dir.mkdir()
        for source_path in episode_paths:
            materialize_file(source_path, output_task_dir / source_path.name, materialization)
        manifest["tasks"][str(task_id)] = {
            "num_episodes": len(episode_paths),
            "episode_files": [path.name for path in episode_paths],
        }

    (output_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered LIBERO retrieval episode pool.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=parse_task_ids, required=True, help="Comma-separated IDs and ranges.")
    parser.add_argument(
        "--materialization",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How episode files are exposed in the filtered pool.",
    )
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = build_pool(
        args.source_dir,
        args.output_dir,
        args.task_ids,
        materialization=args.materialization,
        overwrite=args.overwrite,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    print(
        f"Wrote {manifest['num_episodes']} episodes across {manifest['num_tasks']} tasks "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
