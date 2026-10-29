import pathlib

import numpy as np
import pytest

from openpi.shared import libero_trajectory_visualization as viz


def test_slice_trajectory_points_clips_near_episode_end() -> None:
    states = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)

    points = viz.slice_trajectory_points(states, step_idx=3, horizon=20)

    np.testing.assert_array_equal(points, states[3:, :3])


def test_load_retrieval_trajectory_from_npz(tmp_path: pathlib.Path) -> None:
    episode_path = tmp_path / "episode_7.npz"
    states = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    np.savez(episode_path, state=states)

    trajectory = viz.load_retrieval_trajectory(
        {
            "rank": 2,
            "distance": 4.5,
            "episode_path": str(episode_path),
            "step_idx": 4,
        },
        query_idx=3,
        timestep=25,
        horizon=10,
    )

    assert trajectory.rank == 2
    assert trajectory.distance == 4.5
    assert trajectory.query_idx == 3
    assert trajectory.timestep == 25
    assert trajectory.points == states[4:, :3].astype(float).tolist()


def test_color_for_rank_is_stable() -> None:
    assert viz.color_for_rank(0) == (255, 145, 35)
    assert viz.color_for_rank(1) == (255, 214, 76)
    assert viz.color_for_rank(4) == (70, 220, 220)
    assert viz.color_for_rank(99) == (70, 220, 220)


def test_trajectory_segments_downsamples_long_paths() -> None:
    points = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)

    segments = viz.trajectory_segments(points, max_segments=3)

    assert len(segments) <= 3
    np.testing.assert_array_equal(segments[0][0], points[0])
    np.testing.assert_array_equal(segments[-1][1], points[-1])


def test_build_generated_action_trajectory_integrates_delta_positions() -> None:
    query_state = np.asarray([1.0, 2.0, 3.0, 0.0], dtype=np.float32)
    action_chunk = np.asarray(
        [
            [0.1, 0.0, 0.0, 0.0],
            [0.0, -0.2, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
        ],
        dtype=np.float32,
    )

    points = viz.build_generated_action_trajectory(query_state, action_chunk, horizon=2, action_scale=2.0)

    expected = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.2, 2.0, 3.0],
            [1.2, 1.6, 3.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(points, expected)


def test_add_mujoco_marker_geoms_to_xml() -> None:
    xml = "<mujoco><worldbody /></mujoco>"

    updated = viz.add_mujoco_marker_geoms_to_xml(xml, marker_count=3)

    assert "openpi_traj_marker_body_0000" in updated
    assert "openpi_traj_marker_geom_0002" in updated
    assert "openpi_traj_left" in updated
    assert "openpi_traj_front" in updated
    assert "openpi_traj_right" in updated
    assert 'mocap="true"' in updated
    assert 'contype="0"' in updated


def test_mujoco_renderer_uses_compiled_model_markers() -> None:
    class FakeModel:
        body_mocapid = np.asarray([0])

        def geom_name2id(self, name: str) -> int:
            if name == "openpi_traj_marker_geom_0000":
                return 0
            raise ValueError(name)

        def body_name2id(self, name: str) -> int:
            if name == "openpi_traj_marker_body_0000":
                return 0
            raise ValueError(name)

    class FakeSim:
        model = FakeModel()

    class FakeEnv:
        sim = FakeSim()

    visualizer = viz.LiberoTrajectoryVisualizer(horizon=20, top_k=5, frame_size=224, renderer="mujoco")

    assert visualizer.detect_renderer_mode(FakeEnv()) == "mujoco_model_marker_geoms"


def test_real_retrieval_episode_has_usable_state_trajectory() -> None:
    episode_path = pathlib.Path(
        "ricl_libero_preprocessing/libero90_official20to29_spread20/retrieval/task_20/episode_5.npz"
    )
    if not episode_path.exists():
        pytest.skip(f"Local retrieval episode not found: {episode_path}")

    trajectory = viz.load_retrieval_trajectory(
        {
            "rank": 0,
            "distance": 1.0,
            "episode_path": str(episode_path),
            "step_idx": 98,
        },
        query_idx=0,
        timestep=10,
        horizon=20,
    )

    assert trajectory.points
    assert len(trajectory.points) <= 20
    assert len(trajectory.points[0]) == 3
