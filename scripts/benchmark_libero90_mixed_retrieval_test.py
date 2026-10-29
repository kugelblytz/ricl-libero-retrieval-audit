from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import benchmark_libero90_mixed_retrieval as mixed
import numpy as np
import pytest

from openpi.policies import libero_retrieval_benchmark as benchmark


def test_deterministic_global_top_k_breaks_cross_task_ties():
    selected = benchmark.deterministic_global_top_k(
        np.asarray([1.0, 1.0, 1.0, 0.5, 2.0]),
        np.asarray([2, 1, 1, 9, 0]),
        np.asarray([0, 5, 3, 8, 0]),
        np.asarray([0, 5, 3, 8, 0]),
        k=4,
    )

    np.testing.assert_array_equal(selected, np.asarray([3, 2, 1, 0]))


def test_mixed_metrics_give_wrong_task_candidates_zero_credit():
    actions = np.asarray([np.full((2, 1), value, dtype=np.float32) for value in [0, 1, 2, 3, 0, 0]])
    metrics = benchmark.compute_mixed_task_metrics(
        selected_indices=np.asarray([4, 5, 4, 5]),
        candidate_task_indices=np.asarray([1, 1, 1, 1, 2, 2]),
        query_task_index=1,
        candidate_action_chunks=actions,
        query_action_chunk=np.zeros((2, 1), dtype=np.float32),
    )

    assert metrics.top1_task_accuracy == 0
    assert metrics.task_recall_at4 == 0
    assert metrics.task_aware_ndcg_at4 == 0
    assert metrics.conditional_best_action_rmse is None


def test_mixed_metrics_task_action_oracle_scores_one():
    actions = np.asarray([np.full((2, 1), value, dtype=np.float32) for value in [0, 1, 2, 3, 0, 0]])
    selected = np.asarray([0, 1, 2, 3])
    metrics = benchmark.compute_mixed_task_metrics(
        selected_indices=selected,
        candidate_task_indices=np.asarray([1, 1, 1, 1, 2, 2]),
        query_task_index=1,
        candidate_action_chunks=actions,
        query_action_chunk=np.zeros((2, 1), dtype=np.float32),
        oracle_indices=selected,
    )

    assert metrics.top1_task_accuracy == 1
    assert metrics.task_recall_at4 == 1
    assert metrics.correct_task_fraction_at4 == 1
    assert metrics.task_aware_ndcg_at4 == pytest.approx(1)
    assert metrics.exact_oracle_top1 == 1
    assert metrics.oracle_overlap_count_at4 == 4


def test_shuffled_prompt_map_never_uses_same_prompt():
    tasks = (20, 21, 30, 31)
    prompts = {
        20: "turn on the stove",
        21: "put the pan on the stove",
        30: "put the bowl on the plate",
        31: "put the bowl on the cabinet",
    }

    mapping = mixed.shuffled_prompt_map(tasks, prompts)

    assert set(mapping) == set(tasks)
    assert all(prompts[task] != prompts[other] for task, other in mapping.items())


def test_average_optional_metrics_ignores_unmatched_conditional_values():
    averaged = mixed.average_optional_metrics(
        [
            {"score": 1.0, "conditional": None},
            {"score": 0.0, "conditional": 0.2},
        ]
    )

    assert averaged["score"] == pytest.approx(0.5)
    assert averaged["conditional"] == pytest.approx(0.2)
