# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_PATH = Path(__file__).with_name("analyze_libero90_closed_loop_retrieval.py")
SPEC = importlib.util.spec_from_file_location("analyze_libero90_closed_loop_retrieval", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _trace(task_id: int, scene_id: int, trial_id: int, correct_by_step: list[bool]) -> dict:
    items = []
    for step, correct in enumerate(correct_by_step, start=1):
        items.extend(
            [
                {
                    "policy_step": step * 5,
                    "rank": rank,
                    "retrieved_task_id": task_id if correct else task_id + 100,
                }
                for rank in range(4)
            ]
        )
    return {
        "task_id": task_id,
        "scene_setup_id": scene_id,
        "trial_id": trial_id,
        "error": None,
        "retrieval_trace": items,
    }


def test_protocols_partition_tasks_and_preserve_driver_trial_ids(tmp_path: Path) -> None:
    tasks = list(range(25))
    paths = []
    for name, selected in (("primary", tasks[:10]), ("extension", tasks[10:])):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "capability_task_ids": tasks,
                    "evaluation_task_ids": selected,
                    "per_task": {str(task_id): {"final": list(range(10, 20))} for task_id in selected},
                }
            )
        )
        paths.append(path)

    task_ids, scenes, expected = analysis._load_protocols(paths)

    assert task_ids == tasks
    assert scenes[0] == list(range(10, 20))
    assert (0, 10, 0) in expected
    assert (0, 19, 9) in expected
    assert (0, 19, 0) not in expected
    assert len(expected) == 250


def test_matched_expert_queries_use_episode_modulo_50(tmp_path: Path) -> None:
    final_scenes = {20: [2, 7], 21: [12]}
    rows = []
    for method in analysis.OFFLINE_METHODS.values():
        if method == "random_global":
            continue
        for task_id, scenes in final_scenes.items():
            for scene_id in scenes:
                rows.extend(
                    [
                        {
                            "method": method,
                            "task_id": task_id,
                            "query_episode_index": scene_id + 100,
                            "query_frame_index": frame,
                            "correct_task_fraction_at4": 0.75,
                            "top1_task_accuracy": 1.0,
                            "task_aware_ndcg_at4": 0.8,
                        }
                        for frame in range(2)
                    ]
                )
            # Same local task, but a scene outside the frozen manifest.
            rows.append(
                {
                    "method": method,
                    "task_id": task_id,
                    "query_episode_index": 149,
                    "query_frame_index": 0,
                    "correct_task_fraction_at4": 0.0,
                    "top1_task_accuracy": 0.0,
                    "task_aware_ndcg_at4": 0.0,
                }
            )
    path = tmp_path / "queries.jsonl"
    _write_jsonl(path, rows)

    summaries, scenes = analysis._matched_expert_scenes(path, final_scenes, expected_queries_per_scene=2)

    assert len(summaries) == 3
    assert all(row["matched_expert_queries"] == 6 for row in summaries)
    assert summaries[0]["correct_task_fraction"] == 0.75
    assert ("dino_global", (20, 2)) in scenes
    assert ("dino_global", (20, 49)) not in scenes


def test_early_feedback_uses_only_first_ten_queries() -> None:
    expected = {(1, 10, 0), (1, 11, 1), (2, 10, 0), (2, 11, 1)}
    results = {}
    traces = {}
    for condition in analysis.INFORMED_CONDITIONS:
        results[condition] = {
            (1, 10, 0): {"success": True},
            (1, 11, 1): {"success": False},
            (2, 10, 0): {"success": True},
            (2, 11, 1): {"success": False},
        }
        traces[condition] = {
            (1, 10, 0): _trace(1, 10, 0, [True] * 10 + [False] * 5),
            (1, 11, 1): _trace(1, 11, 1, [False] * 15),
            (2, 10, 0): _trace(2, 10, 0, [True] * 5 + [False] * 10),
            (2, 11, 1): _trace(2, 11, 1, [False] * 15),
        }

    rows = analysis._early_retrieval_feedback(results, traces, expected, 100, 7)

    assert all(row["eligible_tasks_with_success_and_failure"] == 2 for row in rows)
    assert all(np.isclose(row["successful_minus_failed_correct_task_fraction_pp"], 75.0) for row in rows)


def test_incomplete_condition_is_withheld_from_summary() -> None:
    expected = {(1, 10, 0), (1, 11, 1)}
    results = {}
    traces = {}
    for condition in analysis.CONDITIONS:
        keys = expected if condition != "random_global" else {(1, 10, 0)}
        results[condition] = {key: {"success": key[1] == 10} for key in keys}
        traces[condition] = {key: _trace(*key, [True] * 10) for key in keys}

    rows, _, _ = analysis._closed_loop_summaries(
        results, traces, expected, dict.fromkeys(analysis.CONDITIONS, 0), 100, 7
    )
    by_condition = {row["condition"]: row for row in rows}

    assert by_condition["dino_global"]["success_rate"] == 0.5
    assert by_condition["random_global"]["complete"] is False
    assert by_condition["random_global"]["observed_successes"] == 1
    assert by_condition["random_global"]["success_rate"] is None


def test_random_claim_requires_interval_and_holm_threshold() -> None:
    expected = {(1, scene_id, scene_id) for scene_id in range(10)}
    results = {
        condition: {key: {"success": condition != "random_global"} for key in expected}
        for condition in analysis.CONDITIONS
    }

    rows = analysis._paired_comparisons(results, expected, 100, 7)
    random_rows = [row for row in rows if row["right"] == "random_global"]

    assert len(rows) == 6
    assert all(row["difference_ci95_low_pp"] > 0 for row in random_rows)
    assert all(row["holm_adjusted_p"] < 0.05 for row in random_rows)
    assert all(row["informed_beats_random_claim_supported"] for row in random_rows)
