"""Convert LIBERO data to the OpenPI-compatible LeRobot format.

Official LIBERO-100 data is split into LIBERO-90 and LIBERO-10 folders. This
converter reads those HDF5 files directly by default and preserves the OpenPI
training fields used by ``LeRobotLiberoDataConfig``:

    image, wrist_image, state, actions

The previous RLDS/TFDS conversion path is still available with
``--source-format rlds``.

Examples:
    uv run examples/libero/convert_libero_data_to_lerobot.py \
        --raw-dir /path/to/LIBERO --repo-id my-org/libero100 --split libero_100_all --overwrite

    uv run examples/libero/convert_libero_data_to_lerobot.py \
        --raw-dir /path/to/LIBERO --repo-id my-org/libero90 --split libero_90 --task-ids 0,1,2 --overwrite
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Literal

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
import tqdm

SourceFormat = Literal["official_hdf5", "rlds"]
LiberoSplit = Literal["libero_90", "libero_10", "libero_100_all"]

DEFAULT_REPO_ID = "your_hf_username/libero"
DEFAULT_RLDS_DATASETS = (
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
)
OFFICIAL_SPLITS = {
    "libero_90": {
        "folder_candidates": ("LIBERO-90", "libero_90", "libero-90", "LIBERO_90"),
        "split_index": 0,
        "global_task_offset": 0,
    },
    "libero_10": {
        "folder_candidates": ("LIBERO-10", "libero_10", "libero-10", "LIBERO_10"),
        "split_index": 1,
        "global_task_offset": 90,
    },
}
IMAGE_SIZE = 256


@dataclass(frozen=True)
class SplitDir:
    split_name: str
    split_index: int
    global_task_offset: int
    path: Path


@dataclass(frozen=True)
class OfficialTask:
    split_name: str
    split_index: int
    task_id: int
    global_task_id: int
    task_name: str
    prompt: str
    hdf5_path: Path


@dataclass(frozen=True)
class OfficialEpisode:
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    actions: np.ndarray


def parse_task_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    try:
        task_ids = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Task IDs must be comma-separated integers.") from exc
    if any(task_id < 0 for task_id in task_ids):
        raise argparse.ArgumentTypeError("Task IDs must be non-negative.")
    return task_ids


def parse_csv(value: str | None) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def split_names_for_selection(split: LiberoSplit) -> tuple[str, ...]:
    if split == "libero_100_all":
        return ("libero_90", "libero_10")
    return (split,)


def resolve_official_libero_split_dirs(raw_dir: Path, split: LiberoSplit) -> list[SplitDir]:
    raw_dir = Path(raw_dir)
    split_dirs = []
    for split_name in split_names_for_selection(split):
        split_info = OFFICIAL_SPLITS[split_name]
        split_path = None
        for folder_name in split_info["folder_candidates"]:
            candidate = raw_dir / folder_name
            if candidate.exists() and candidate.is_dir():
                split_path = candidate
                break
        if split_path is None:
            candidates = ", ".join(str(raw_dir / folder_name) for folder_name in split_info["folder_candidates"])
            raise FileNotFoundError(f"Could not find {split_name} under {raw_dir}. Checked: {candidates}")

        split_dirs.append(
            SplitDir(
                split_name=split_name,
                split_index=int(split_info["split_index"]),
                global_task_offset=int(split_info["global_task_offset"]),
                path=split_path,
            )
        )
    return split_dirs


def parse_task_name_from_hdf5_filename(path: Path) -> str:
    task_name = path.stem
    task_name = re.sub(r"_demo$", "", task_name, flags=re.IGNORECASE)
    task_name = re.sub(r"^(libero[-_]?90|libero[-_]?10)_", "", task_name, flags=re.IGNORECASE)
    task_name = re.sub(r"^[a-z]+_scene\d+_", "", task_name, flags=re.IGNORECASE)
    task_name = task_name.replace("_", " ")
    return re.sub(r"\s+", " ", task_name).strip()


def _load_libero_task_prompts(split_name: str) -> dict[int, str]:
    try:
        from libero.libero import benchmark  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return {}

    try:
        task_suite = benchmark.get_benchmark(split_name)()
    except Exception:
        return {}

    prompts = {}
    n_tasks = getattr(task_suite, "n_tasks", 0)
    for task_id in range(n_tasks):
        try:
            task = task_suite.get_task(task_id)
        except Exception:
            continue
        prompt = (
            getattr(task, "language", None)
            or getattr(task, "language_instruction", None)
            or getattr(task, "description", None)
            or getattr(task, "name", None)
        )
        if prompt:
            prompts[task_id] = str(prompt)
    return prompts


def discover_official_hdf5_tasks(
    split_dirs: list[SplitDir],
    *,
    task_ids: tuple[int, ...] | None = None,
    task_name_filters: tuple[str, ...] | None = None,
) -> list[OfficialTask]:
    selected_ids = set(task_ids) if task_ids is not None else None
    filters = tuple(filter_text.lower() for filter_text in task_name_filters or ())
    tasks = []

    for split_dir in split_dirs:
        prompt_overrides = _load_libero_task_prompts(split_dir.split_name)
        hdf5_paths = sorted(split_dir.path.glob("*_demo.hdf5"))
        if not hdf5_paths:
            raise FileNotFoundError(f"No *_demo.hdf5 files found in {split_dir.path}")

        for task_id, hdf5_path in enumerate(hdf5_paths):
            task_name = parse_task_name_from_hdf5_filename(hdf5_path)
            if selected_ids is not None and task_id not in selected_ids:
                continue
            searchable_name = f"{task_name} {hdf5_path.stem}".lower()
            if filters and not any(filter_text in searchable_name for filter_text in filters):
                continue

            tasks.append(
                OfficialTask(
                    split_name=split_dir.split_name,
                    split_index=split_dir.split_index,
                    task_id=task_id,
                    global_task_id=split_dir.global_task_offset + task_id,
                    task_name=task_name,
                    prompt=prompt_overrides.get(task_id, task_name),
                    hdf5_path=hdf5_path,
                )
            )

    if not tasks:
        raise ValueError("No official LIBERO HDF5 tasks matched the requested split/filter selection.")
    return tasks


def _get_dataset(group: Any, dataset_path: str) -> np.ndarray | None:
    if dataset_path not in group:
        return None
    value = group[dataset_path]
    if not hasattr(value, "shape"):
        return None
    return np.asarray(value)


def _first_dataset(group: Any, dataset_paths: tuple[str, ...]) -> np.ndarray:
    for dataset_path in dataset_paths:
        value = _get_dataset(group, dataset_path)
        if value is not None:
            return value
    raise KeyError(f"Expected one of {dataset_paths} in HDF5 demo group.")


def _as_time_major_2d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return value[:, None]
    if value.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D state array, got {value.shape}.")
    return value


def _load_official_state(demo_group: Any) -> np.ndarray:
    incompatible_state_shapes = []
    for dataset_path in ("robot_states", "obs/robot_states"):
        value = _get_dataset(demo_group, dataset_path)
        if value is not None:
            value = _as_time_major_2d(value)
            if value.shape[-1] == 8:
                return value.astype(np.float32)
            incompatible_state_shapes.append(f"{dataset_path}={value.shape}")

    single_fallbacks = ("obs/gripper_states", "obs/ee_states", "obs/joint_states")
    for dataset_path in single_fallbacks:
        value = _get_dataset(demo_group, dataset_path)
        if value is not None:
            value = _as_time_major_2d(value)
            if value.shape[-1] == 8:
                return value.astype(np.float32)

    combo_fallbacks = (
        ("obs/ee_states", "obs/gripper_states"),
        ("obs/joint_states", "obs/gripper_states"),
    )
    for dataset_paths in combo_fallbacks:
        values = [_get_dataset(demo_group, dataset_path) for dataset_path in dataset_paths]
        if all(value is not None for value in values):
            arrays = [_as_time_major_2d(value) for value in values if value is not None]
            if len({array.shape[0] for array in arrays}) != 1:
                raise ValueError(f"State fallback {dataset_paths} has inconsistent time dimensions.")
            state = np.concatenate(arrays, axis=-1)
            if state.shape[-1] == 8:
                return state.astype(np.float32)

    incompatible_message = ""
    if incompatible_state_shapes:
        incompatible_message = f" Found incompatible state candidates: {', '.join(incompatible_state_shapes)}."
    raise KeyError(
        "Could not find an OpenPI-compatible 8-D LIBERO state. Tried robot_states, obs/robot_states, "
        "and concat-compatible obs/ee_states or obs/joint_states with obs/gripper_states."
        f"{incompatible_message}"
    )


def _resize_image(image: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    if image.shape[:2] == (image_size, image_size):
        return np.ascontiguousarray(image)
    return np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR))


def _as_uint8_images(images: np.ndarray, *, rotate_images_180: bool) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (T, H, W, C) or (T, C, H, W), got {images.shape}.")
    if images.shape[1] == 3 and images.shape[-1] != 3:
        images = np.moveaxis(images, 1, -1)
    if images.shape[-1] != 3:
        raise ValueError(f"Expected RGB images with 3 channels, got {images.shape}.")

    if np.issubdtype(images.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(images)) <= 1.0 else 1.0
        images = np.clip(images * scale, 0, 255).astype(np.uint8)
    else:
        images = np.clip(images, 0, 255).astype(np.uint8)

    if rotate_images_180:
        images = images[:, ::-1, ::-1, :]

    return np.stack([_resize_image(image) for image in images], axis=0).astype(np.uint8)


def load_official_demo_episode(h5_file: Any, demo_key: str, *, rotate_images_180: bool) -> OfficialEpisode:
    demo_group = h5_file[f"data/{demo_key}"]
    actions = np.asarray(_first_dataset(demo_group, ("actions",)), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected actions with shape (T, 7), got {actions.shape} in {demo_key}.")

    state = _load_official_state(demo_group)
    image = _as_uint8_images(
        _first_dataset(demo_group, ("obs/agentview_rgb", "obs/agentview_image", "obs/image")),
        rotate_images_180=rotate_images_180,
    )
    wrist_image = _as_uint8_images(
        _first_dataset(
            demo_group,
            ("obs/eye_in_hand_rgb", "obs/robot0_eye_in_hand_rgb", "obs/eye_in_hand_image", "obs/wrist_image"),
        ),
        rotate_images_180=rotate_images_180,
    )

    lengths = {len(actions), len(state), len(image), len(wrist_image)}
    if len(lengths) != 1:
        raise ValueError(
            f"Inconsistent episode lengths in {demo_key}: "
            f"actions={len(actions)}, state={len(state)}, image={len(image)}, wrist_image={len(wrist_image)}."
        )

    return OfficialEpisode(image=image, wrist_image=wrist_image, state=state, actions=actions)


def create_libero_lerobot_dataset(
    repo_id: str,
    *,
    root: Path | None = None,
    overwrite: bool = False,
    use_videos: bool = False,
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
) -> LeRobotDataset:
    output_path = (Path(root) / repo_id) if root is not None else LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    image_dtype = "video" if use_videos else "image"
    features = {
        "image": {
            "dtype": image_dtype,
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": image_dtype,
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
        "libero_split_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["libero_split_index"],
        },
        "libero_task_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["libero_task_id"],
        },
        "libero_global_task_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["libero_global_task_id"],
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=10,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def _sorted_demo_keys(h5_file: Any) -> list[str]:
    if "data" not in h5_file:
        raise KeyError("Expected official LIBERO HDF5 file to contain a top-level `data` group.")

    def sort_key(key: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", key)
        return (int(match.group(1)) if match else -1, key)

    demo_keys = sorted((key for key in h5_file["data"] if key.startswith("demo_")), key=sort_key)
    if not demo_keys:
        raise KeyError("Expected official LIBERO HDF5 file to contain data/demo_* groups.")
    return demo_keys


def _frame_metadata(task: OfficialTask) -> dict[str, int]:
    return {
        "libero_split_index": task.split_index,
        "libero_task_id": task.task_id,
        "libero_global_task_id": task.global_task_id,
    }


def _add_episode_to_dataset(dataset: LeRobotDataset, episode: OfficialEpisode, task: OfficialTask) -> int:
    frame_metadata = _frame_metadata(task)
    for frame_idx in range(len(episode.actions)):
        dataset.add_frame(
            {
                "image": episode.image[frame_idx],
                "wrist_image": episode.wrist_image[frame_idx],
                "state": episode.state[frame_idx],
                "actions": episode.actions[frame_idx],
                **frame_metadata,
            }
        )
    dataset.save_episode(task=task.prompt)
    return len(episode.actions)


def _write_manifest(dataset: LeRobotDataset, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(dataset.root) / "libero_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def convert_official_hdf5(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Official LIBERO HDF5 conversion requires `h5py`. Install it with `uv pip install h5py`.") from exc

    split_dirs = resolve_official_libero_split_dirs(args.raw_dir, args.split)
    tasks = discover_official_hdf5_tasks(
        split_dirs,
        task_ids=args.task_ids,
        task_name_filters=tuple(args.task_name_filter),
    )
    dataset = create_libero_lerobot_dataset(
        args.repo_id,
        root=args.root,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    manifest_tasks = []
    total_episodes = 0
    total_frames = 0
    for task in tqdm.tqdm(tasks, desc="Converting official LIBERO tasks"):
        demo_count = 0
        frame_count = 0
        with h5py.File(task.hdf5_path, "r") as h5_file:
            for demo_key in _sorted_demo_keys(h5_file):
                episode = load_official_demo_episode(h5_file, demo_key, rotate_images_180=args.rotate_images_180)
                frame_count += _add_episode_to_dataset(dataset, episode, task)
                demo_count += 1
                total_episodes += 1
                total_frames += len(episode.actions)

        manifest_tasks.append(
            {
                **asdict(task),
                "hdf5_path": str(task.hdf5_path),
                "demo_count": demo_count,
                "frame_count": frame_count,
            }
        )

    dataset.consolidate(run_compute_stats=False)
    manifest = {
        "source_format": "official_hdf5",
        "split": args.split,
        "selected_splits": [split_dir.split_name for split_dir in split_dirs],
        "task_ids": list(args.task_ids) if args.task_ids is not None else None,
        "task_name_filters": list(args.task_name_filter),
        "rotate_images_180": args.rotate_images_180,
        "repo_id": args.repo_id,
        "total_tasks": len(manifest_tasks),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "tasks": manifest_tasks,
    }
    manifest_path = _write_manifest(dataset, manifest)

    if args.push_to_hub:
        dataset.push_to_hub(tags=["libero", "libero-100", "panda"], private=False, push_videos=args.use_videos)

    print(f"Wrote LeRobot dataset to {dataset.root}")
    print(f"Wrote LIBERO conversion manifest to {manifest_path}")
    return manifest


def convert_rlds(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        raise RuntimeError(
            "RLDS conversion requires tensorflow_datasets and a compatible TensorFlow/protobuf environment. "
            "Use --source-format official_hdf5 to convert official LIBERO-100 HDF5 data without TFDS."
        ) from exc

    dataset = create_libero_lerobot_dataset(
        args.repo_id,
        root=args.root,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    rlds_datasets = args.rlds_datasets or DEFAULT_RLDS_DATASETS
    total_episodes = 0
    total_frames = 0

    for raw_dataset_name in rlds_datasets:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(args.raw_dir), split="train")
        for episode in tqdm.tqdm(raw_dataset, desc=f"Converting {raw_dataset_name}"):
            prompt = None
            frame_count = 0
            for step in episode["steps"].as_numpy_iterator():
                prompt = step["language_instruction"].decode()
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "libero_split_index": np.asarray([-1], dtype=np.int64),
                        "libero_task_id": np.asarray([-1], dtype=np.int64),
                        "libero_global_task_id": np.asarray([-1], dtype=np.int64),
                    }
                )
                frame_count += 1
            dataset.save_episode(task=prompt or raw_dataset_name)
            total_episodes += 1
            total_frames += frame_count

    dataset.consolidate(run_compute_stats=False)
    manifest = {
        "source_format": "rlds",
        "repo_id": args.repo_id,
        "rlds_datasets": list(rlds_datasets),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
    }
    manifest_path = _write_manifest(dataset, manifest)

    if args.push_to_hub:
        dataset.push_to_hub(tags=["libero", "panda", "rlds"], private=False, push_videos=args.use_videos)

    print(f"Wrote LeRobot dataset to {dataset.root}")
    print(f"Wrote LIBERO conversion manifest to {manifest_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", "--data-dir", "--data_dir", dest="raw_dir", type=Path, required=True)
    parser.add_argument("--repo-id", "--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--source-format", choices=("official_hdf5", "rlds"), default="official_hdf5")
    parser.add_argument("--split", choices=("libero_90", "libero_10", "libero_100_all"), default="libero_100_all")
    parser.add_argument("--task-ids", type=parse_task_ids, default=None)
    parser.add_argument("--task-name-filter", action="append", default=[])
    parser.add_argument("--rlds-datasets", type=parse_csv, default=None)
    parser.add_argument("--push-to-hub", "--push_to_hub", action="store_true")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rotate-images-180", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-writer-processes", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.source_format == "official_hdf5":
        return convert_official_hdf5(args)
    if args.source_format == "rlds":
        return convert_rlds(args)
    raise ValueError(f"Unsupported source format: {args.source_format}")


if __name__ == "__main__":
    main()
