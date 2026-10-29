from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

CONVERTER_PATH = Path(__file__).with_name("convert_libero_data_to_lerobot.py")
SPEC = importlib.util.spec_from_file_location("convert_libero_data_to_lerobot", CONVERTER_PATH)
assert SPEC is not None
assert SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def _tiny_images() -> np.ndarray:
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    images[:, 0, 0] = np.asarray([1, 2, 3], dtype=np.uint8)
    images[:, -1, -1] = np.asarray([9, 8, 7], dtype=np.uint8)
    return images


def _write_official_task(path: Path, *, state_key: str = "robot_states") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        demo = h5_file.create_group("data/demo_0")
        obs = demo.create_group("obs")
        demo.create_dataset("actions", data=np.arange(14, dtype=np.float32).reshape(2, 7))
        if state_key == "robot_states":
            demo.create_dataset("robot_states", data=np.arange(16, dtype=np.float32).reshape(2, 8))
        elif state_key == "official_robot_states":
            demo.create_dataset("robot_states", data=np.arange(18, dtype=np.float32).reshape(2, 9))
            obs.create_dataset("ee_states", data=np.arange(12, dtype=np.float32).reshape(2, 6))
            obs.create_dataset("gripper_states", data=np.ones((2, 2), dtype=np.float32))
        elif state_key == "fallback":
            obs.create_dataset("ee_states", data=np.arange(14, dtype=np.float32).reshape(2, 7))
            obs.create_dataset("gripper_states", data=np.ones((2, 1), dtype=np.float32))
        else:
            raise ValueError(state_key)
        obs.create_dataset("agentview_rgb", data=_tiny_images())
        obs.create_dataset("eye_in_hand_rgb", data=_tiny_images() + 1)


def test_resolve_official_split_dirs_accepts_case_variants_and_rejects_missing(tmp_path):
    raw_dir = tmp_path / "LIBERO"
    (raw_dir / "LIBERO-90").mkdir(parents=True)
    (raw_dir / "libero_10").mkdir()

    split_dirs = converter.resolve_official_libero_split_dirs(raw_dir, "libero_100_all")

    assert [split_dir.split_name for split_dir in split_dirs] == ["libero_90", "libero_10"]
    assert [split_dir.split_index for split_dir in split_dirs] == [0, 1]

    with pytest.raises(FileNotFoundError):
        converter.resolve_official_libero_split_dirs(tmp_path / "missing", "libero_90")


def test_parse_task_name_from_hdf5_filename_strips_libero_scene_and_demo_prefixes():
    assert (
        converter.parse_task_name_from_hdf5_filename(
            Path("KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_demo.hdf5")
        )
        == "open the top drawer of the cabinet"
    )
    assert converter.parse_task_name_from_hdf5_filename(Path("libero_10_put_the_mug_on_the_plate_demo.hdf5")) == (
        "put the mug on the plate"
    )


def test_discover_official_hdf5_tasks_filters_by_task_id_and_name(tmp_path):
    raw_dir = tmp_path / "LIBERO"
    split_dir = raw_dir / "LIBERO-90"
    _write_official_task(split_dir / "task_alpha_demo.hdf5")
    _write_official_task(split_dir / "task_mug_demo.hdf5")
    _write_official_task(split_dir / "task_plate_demo.hdf5")
    split_dirs = converter.resolve_official_libero_split_dirs(raw_dir, "libero_90")

    by_id = converter.discover_official_hdf5_tasks(split_dirs, task_ids=(1,))
    by_name = converter.discover_official_hdf5_tasks(split_dirs, task_name_filters=("plate",))

    assert [task.task_name for task in by_id] == ["task mug"]
    assert [task.task_id for task in by_name] == [2]
    assert by_name[0].global_task_id == 2


def test_load_official_demo_episode_prefers_robot_states_and_rotates_images(tmp_path):
    hdf5_path = tmp_path / "LIBERO-10" / "task_mug_demo.hdf5"
    _write_official_task(hdf5_path)

    with h5py.File(hdf5_path, "r") as h5_file:
        episode = converter.load_official_demo_episode(h5_file, "demo_0", rotate_images_180=False)
        rotated_episode = converter.load_official_demo_episode(h5_file, "demo_0", rotate_images_180=True)

    assert episode.actions.shape == (2, 7)
    assert episode.state.shape == (2, 8)
    assert episode.image.shape == (2, 256, 256, 3)
    np.testing.assert_array_equal(episode.image[0, 0, 0], np.asarray([1, 2, 3], dtype=np.uint8))
    np.testing.assert_array_equal(rotated_episode.image[0, 0, 0], np.asarray([9, 8, 7], dtype=np.uint8))


def test_load_official_demo_episode_supports_concat_state_fallback(tmp_path):
    hdf5_path = tmp_path / "LIBERO-90" / "task_fallback_demo.hdf5"
    _write_official_task(hdf5_path, state_key="fallback")

    with h5py.File(hdf5_path, "r") as h5_file:
        episode = converter.load_official_demo_episode(h5_file, "demo_0", rotate_images_180=False)

    assert episode.state.shape == (2, 8)
    np.testing.assert_allclose(episode.state[:, -1], np.ones((2,), dtype=np.float32))


def test_load_official_demo_episode_uses_fallback_when_robot_states_is_9d(tmp_path):
    hdf5_path = tmp_path / "LIBERO-90" / "task_official_demo.hdf5"
    _write_official_task(hdf5_path, state_key="official_robot_states")

    with h5py.File(hdf5_path, "r") as h5_file:
        episode = converter.load_official_demo_episode(h5_file, "demo_0", rotate_images_180=False)

    assert episode.state.shape == (2, 8)
    np.testing.assert_allclose(episode.state[:, :6], np.arange(12, dtype=np.float32).reshape(2, 6))
    np.testing.assert_allclose(episode.state[:, 6:], np.ones((2, 2), dtype=np.float32))


def test_convert_official_hdf5_smoke_writes_manifest_and_metadata_features(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_official_task(raw_dir / "LIBERO-90" / "task_a_demo.hdf5")
    _write_official_task(raw_dir / "LIBERO-10" / "task_b_demo.hdf5")
    output_root = tmp_path / "lerobot"

    manifest = converter.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--repo-id",
            "test/libero100",
            "--root",
            str(output_root),
            "--split",
            "libero_100_all",
            "--overwrite",
            "--image-writer-processes",
            "0",
            "--image-writer-threads",
            "0",
        ]
    )

    dataset_root = output_root / "test" / "libero100"
    manifest_path = dataset_root / "libero_conversion_manifest.json"
    info_path = dataset_root / "meta" / "info.json"

    assert manifest["total_tasks"] == 2
    assert manifest["total_episodes"] == 2
    assert manifest["selected_splits"] == ["libero_90", "libero_10"]
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text())["total_frames"] == 4

    info = json.loads(info_path.read_text())
    for feature_key in [
        "image",
        "wrist_image",
        "state",
        "actions",
        "libero_split_index",
        "libero_task_id",
        "libero_global_task_id",
    ]:
        assert feature_key in info["features"]


def test_cli_parses_split_filters_and_keeps_rlds_lazy():
    parser = converter.build_arg_parser()

    args = parser.parse_args(["--raw-dir", "/tmp/raw", "--split", "libero_90", "--task-ids", "0,2"])
    rlds_args = parser.parse_args(
        ["--raw-dir", "/tmp/rlds", "--source-format", "rlds", "--rlds-datasets", "libero_10_no_noops"]
    )
    filter_args = parser.parse_args(["--raw-dir", "/tmp/raw", "--split", "libero_10", "--task-name-filter", "mug"])

    assert args.task_ids == (0, 2)
    assert rlds_args.rlds_datasets == ("libero_10_no_noops",)
    assert filter_args.task_name_filter == ["mug"]
