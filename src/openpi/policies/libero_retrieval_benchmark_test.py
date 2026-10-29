import numpy as np
import pytest

from openpi.policies import libero_retrieval_benchmark as benchmark


def test_select_query_steps_is_complete_unique_and_deterministic():
    steps = benchmark.select_query_steps(100, count=10, action_horizon=10)

    np.testing.assert_array_equal(steps, benchmark.select_query_steps(100, count=10, action_horizon=10))
    assert len(np.unique(steps)) == 10
    assert steps[0] == 0
    assert steps[-1] == 90


def test_select_query_steps_rejects_incomplete_episode():
    with pytest.raises(ValueError, match="shorter than action horizon"):
        benchmark.select_query_steps(9, count=1, action_horizon=10)


def test_deterministic_top_k_breaks_ties_by_episode_then_frame():
    selected = benchmark.deterministic_top_k(
        np.asarray([1.0, 1.0, 1.0, 0.5, 2.0]),
        np.asarray([2, 1, 1, 9, 0]),
        np.asarray([0, 5, 3, 8, 0]),
        k=4,
    )

    np.testing.assert_array_equal(selected, np.asarray([3, 2, 1, 0]))


def test_random_top_k_is_seeded_and_preserves_random_order():
    first = benchmark.random_top_k(20, repeats=3, seed=7)
    second = benchmark.random_top_k(20, repeats=3, seed=7)

    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert all(len(np.unique(indices)) == 4 for indices in first)


def test_hybrid_distances_median_scales_components():
    image = np.asarray([1.0, 2.0, 3.0])
    state = np.asarray([2.0, 4.0, 6.0])

    np.testing.assert_allclose(benchmark.hybrid_distances(image, state, beta=1.0), [1.0, 2.0, 3.0])


def test_pi0_observation_input_has_normal_prefix_fields_and_no_actions():
    raw = benchmark.make_pi0_observation_input(
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
        np.arange(8, dtype=np.float32),
        "pick up the bowl",
    )

    assert set(raw) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert "actions" not in raw
    assert raw["prompt"] == "pick up the bowl"


def test_retrieval_metrics_use_best_and_mean_top_four():
    candidate_chunks = np.asarray([np.full((2, 1), value, dtype=np.float32) for value in [0, 1, 2, 3, 4]])
    metrics = benchmark.compute_retrieval_metrics(
        selected_indices=np.asarray([3, 2, 1, 4]),
        candidate_action_chunks=candidate_chunks,
        query_action_chunk=np.zeros((2, 1), dtype=np.float32),
        candidate_states=np.arange(5, dtype=np.float32)[:, None],
        query_state=np.zeros(1, dtype=np.float32),
        candidate_progress=np.linspace(0, 1, 5),
        query_progress=0.0,
        candidate_episode_indices=np.asarray([0, 0, 1, 2, 3]),
    )

    assert metrics.top1_action_rmse == pytest.approx(3.0)
    assert metrics.best_of_4_action_rmse == pytest.approx(1.0)
    assert metrics.mean_top4_action_rmse == pytest.approx(2.5)
    assert metrics.oracle_action_rmse == pytest.approx(0.0)
    assert metrics.oracle_regret == pytest.approx(1.0)
    assert metrics.unique_episodes == 4


def test_macro_average_weights_tasks_equally():
    rows = [
        {"method": "a", "task_id": 1, "score": 0.0},
        {"method": "a", "task_id": 1, "score": 0.0},
        {"method": "a", "task_id": 2, "score": 1.0},
    ]

    per_task, summary = benchmark.macro_average(rows, metric_names=["score"])

    assert len(per_task) == 2
    assert summary[0]["score"] == pytest.approx(0.5)


def test_paired_task_bootstrap_delta_sign():
    delta, low, high = benchmark.paired_task_bootstrap(
        {1: 1.0, 2: 2.0, 3: 3.0},
        {1: 0.5, 2: 1.5, 3: 2.5},
        samples=1000,
        seed=0,
    )

    assert delta == pytest.approx(-0.5)
    assert low == pytest.approx(-0.5)
    assert high == pytest.approx(-0.5)
