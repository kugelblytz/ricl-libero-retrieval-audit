"""Reproduce the unified paper analysis for the 25-task retrieval study.

The script combines the two execution batches, validates their frozen scene
manifests, matches expert queries to those exact scenes, and writes the tables
and four-panel figure used by the paper. By default it refuses to publish an
analysis until all four conditions contain 250 canonical result and trace rows.
Use ``--allow-incomplete`` only for a clearly marked work-in-progress report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CONDITIONS = ("dino_global", "dino_state_global", "pi_global", "random_global")
INFORMED_CONDITIONS = CONDITIONS[:3]
CONDITION_LABELS = {
    "dino_global": "DINO",
    "dino_state_global": "DINO+State",
    "pi_global": "pi0-FF",
    "random_global": "Random",
}
OFFLINE_METHODS = {
    "dino_global": "dino_top_global",
    "dino_state_global": "dino_top_state_global",
    "pi_global": "pi0_fast_first_fusion_global",
    "random_global": "random_global",
}
CONFIRMATORY_COMPARISONS = (
    ("dino_state_global", "dino_global", "DINO+State minus DINO", "primary"),
    ("pi_global", "dino_state_global", "pi0-FF minus DINO+State", "primary"),
    ("dino_global", "random_global", "DINO minus random", "control"),
    ("dino_state_global", "random_global", "DINO+State minus random", "control"),
    ("pi_global", "random_global", "pi0-FF minus random", "control"),
)
DESCRIPTIVE_COMPARISONS = (("pi_global", "dino_global", "pi0-FF minus DINO", "descriptive"),)

DEFAULT_PRIMARY_ROOT = Path("data/libero/videos/libero90_retrieval_final10_global25_disjoint_20260810_211205")
DEFAULT_EXTENSION_ROOT = Path("data/libero/videos/libero90_retrieval_extension15_global25_disjoint_20260811_145749")
DEFAULT_PRIMARY_PROTOCOL = Path("scripts/libero90_final10_protocol.json")
DEFAULT_EXTENSION_PROTOCOL = Path("scripts/libero90_extension15_protocol.json")
DEFAULT_OFFLINE_QUERIES = Path("data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1/full/queries.jsonl")
DEFAULT_OFFLINE_PER_TASK = Path("data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1/full/per_task.csv")
DEFAULT_OUTPUT_DIR = Path("data/libero/reports/libero90_closed_loop_retrieval")

TrialKey = tuple[int, int, int]
SceneKey = tuple[int, int]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSON at {path}:{line_number}: {error}") from error
    return rows


def _trial_key(row: dict[str, Any]) -> TrialKey:
    return int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])


def _canonical_valid(rows: list[dict[str, Any]]) -> dict[TrialKey, dict[str, Any]]:
    """Return the latest error-free record for each canonical rollout identity."""
    canonical = {}
    for row in rows:
        if row.get("error") is None:
            canonical[_trial_key(row)] = row
    return canonical


def _stable_rng(seed: int, label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode()).digest()
    offset = int.from_bytes(digest[:4], "little")
    return np.random.default_rng((seed + offset) % 2**32)


def _percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _hierarchical_mean_ci(
    values_by_task: dict[int, np.ndarray],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not values_by_task or repetitions <= 0:
        raise ValueError("Hierarchical bootstrap requires data and positive repetitions")
    task_values = list(values_by_task.values())
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_tasks = rng.integers(0, len(task_values), size=len(task_values))
        resampled_tasks = []
        for task_position in selected_tasks:
            values = task_values[int(task_position)]
            indices = rng.integers(0, len(values), size=len(values))
            resampled_tasks.append(values[indices])
        draws[repetition] = float(np.mean(np.concatenate(resampled_tasks)))
    return _percentile_interval(draws)


def _task_bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        draws[repetition] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    return _percentile_interval(draws)


def _exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(discordant_a, discordant_b) + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _load_protocols(protocol_paths: list[Path]) -> tuple[list[int], dict[int, list[int]], set[TrialKey]]:
    task_order = []
    final_scenes: dict[int, list[int]] = {}
    capability_sets = []
    for path in protocol_paths:
        protocol = json.loads(path.read_text())
        capability_sets.append(tuple(int(value) for value in protocol["capability_task_ids"]))
        for task_id_raw in protocol["evaluation_task_ids"]:
            task_id = int(task_id_raw)
            if task_id in final_scenes:
                raise ValueError(f"Task {task_id} appears in more than one protocol")
            scenes = [int(value) for value in protocol["per_task"][str(task_id)]["final"]]
            if len(scenes) != 10 or len(set(scenes)) != 10:
                raise ValueError(f"Task {task_id} must have exactly 10 distinct final scenes")
            task_order.append(task_id)
            final_scenes[task_id] = scenes
    if len(set(capability_sets)) != 1:
        raise ValueError("Protocol capability sets differ")
    capability_tasks = set(capability_sets[0])
    if set(task_order) != capability_tasks or len(task_order) != 25:
        raise ValueError("The protocols must partition the frozen 25-task capability set")
    task_order = sorted(task_order)
    # The rollout driver numbers the single trial consecutively across the ten
    # selected scenes (0..9), rather than resetting trial_id to zero per scene.
    expected_keys = {
        (task_id, scene_id, trial_id)
        for task_id in task_order
        for trial_id, scene_id in enumerate(final_scenes[task_id])
    }
    return task_order, final_scenes, expected_keys


def _load_rollouts(
    roots: list[Path], expected_keys: set[TrialKey]
) -> tuple[
    dict[str, dict[TrialKey, dict[str, Any]]],
    dict[str, dict[TrialKey, dict[str, Any]]],
    dict[str, int],
]:
    results: dict[str, dict[TrialKey, dict[str, Any]]] = {}
    traces: dict[str, dict[TrialKey, dict[str, Any]]] = {}
    error_counts = {}
    for condition in CONDITIONS:
        result_rows = []
        trace_rows = []
        for root in roots:
            result_rows.extend(_read_jsonl(root / condition / "results.jsonl"))
            trace_rows.extend(_read_jsonl(root / condition / "retrieval_trace.jsonl"))
        condition_results = _canonical_valid(result_rows)
        condition_traces = _canonical_valid(trace_rows)
        unexpected = (set(condition_results) | set(condition_traces)) - expected_keys
        if unexpected:
            raise ValueError(f"{condition} contains unexpected rollout identities: {sorted(unexpected)[:5]}")
        results[condition] = condition_results
        traces[condition] = condition_traces
        error_counts[condition] = sum(row.get("error") is not None for row in result_rows)
    return results, traces, error_counts


def _trace_metric(trace: dict[str, Any], task_id: int, first_queries: int | None = None) -> dict[str, float]:
    items = trace.get("retrieval_trace", [])
    if first_queries is not None:
        steps = sorted({int(item["policy_step"]) for item in items})
        if len(steps) < first_queries:
            raise ValueError(
                f"Task {task_id} scene {trace.get('scene_setup_id')} has only {len(steps)} retrieval queries"
            )
        selected_steps = set(steps[:first_queries])
        items = [item for item in items if int(item["policy_step"]) in selected_steps]
    if not items:
        raise ValueError("A canonical trace contains no retrieval items")
    correct = np.asarray([int(item["retrieved_task_id"]) == task_id for item in items], dtype=np.float64)
    top1 = np.asarray(
        [int(item["retrieved_task_id"]) == task_id for item in items if int(item.get("rank", -1)) == 0],
        dtype=np.float64,
    )
    if not len(top1):
        raise ValueError("A canonical trace contains no rank-zero retrieval items")
    return {"correct_task_fraction": float(np.mean(correct)), "top1_task_accuracy": float(np.mean(top1))}


def _closed_loop_summaries(
    results: dict[str, dict[TrialKey, dict[str, Any]]],
    traces: dict[str, dict[TrialKey, dict[str, Any]]],
    expected_keys: set[TrialKey],
    error_counts: dict[str, int],
    repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, int], float]]:
    summaries = []
    per_task_rows = []
    task_success_rates = {}
    for condition in CONDITIONS:
        condition_results = results[condition]
        condition_traces = traces[condition]
        complete = set(condition_results) == expected_keys and set(condition_traces) == expected_keys
        observed_successes = sum(bool(row["success"]) for row in condition_results.values())
        summary: dict[str, Any] = {
            "condition": condition,
            "method_label": CONDITION_LABELS[condition],
            "expected_rollouts": len(expected_keys),
            "valid_result_rollouts": len(condition_results),
            "valid_trace_rollouts": len(condition_traces),
            "infrastructure_error_rows": error_counts[condition],
            "complete": complete,
            "observed_successes": observed_successes,
            "successes": observed_successes if complete else None,
            "success_rate": observed_successes / len(expected_keys) if complete else None,
            "success_ci95_low": None,
            "success_ci95_high": None,
            "correct_task_fraction": None,
            "top1_task_accuracy": None,
        }
        if complete:
            success_by_task = {
                task_id: np.asarray(
                    [float(condition_results[key]["success"]) for key in sorted(expected_keys) if key[0] == task_id]
                )
                for task_id in sorted({key[0] for key in expected_keys})
            }
            low, high = _hierarchical_mean_ci(
                success_by_task, repetitions, _stable_rng(seed, f"closed-loop-rate-{condition}")
            )
            metrics = [_trace_metric(condition_traces[key], key[0]) for key in sorted(expected_keys)]
            summary.update(
                {
                    "success_ci95_low": low,
                    "success_ci95_high": high,
                    # One rollout per task-scene makes this equal task-and-scene weighting.
                    "correct_task_fraction": float(np.mean([metric["correct_task_fraction"] for metric in metrics])),
                    "top1_task_accuracy": float(np.mean([metric["top1_task_accuracy"] for metric in metrics])),
                }
            )
            for task_id, task_values in success_by_task.items():
                task_keys = sorted(key for key in expected_keys if key[0] == task_id)
                trace_metrics = [_trace_metric(condition_traces[key], task_id) for key in task_keys]
                rate = float(np.mean(task_values))
                task_success_rates[(condition, task_id)] = rate
                per_task_rows.append(
                    {
                        "condition": condition,
                        "method_label": CONDITION_LABELS[condition],
                        "task_id": task_id,
                        "successes": int(np.sum(task_values)),
                        "trials": len(task_values),
                        "success_rate": rate,
                        "correct_task_fraction": float(
                            np.mean([metric["correct_task_fraction"] for metric in trace_metrics])
                        ),
                        "top1_task_accuracy": float(
                            np.mean([metric["top1_task_accuracy"] for metric in trace_metrics])
                        ),
                    }
                )
        summaries.append(summary)
    return summaries, per_task_rows, task_success_rates


def _paired_comparison(
    left: str,
    right: str,
    label: str,
    family: str,
    results: dict[str, dict[TrialKey, dict[str, Any]]],
    expected_keys: set[TrialKey],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    left_rows = results[left]
    right_rows = results[right]
    if set(left_rows) != expected_keys or set(right_rows) != expected_keys:
        raise ValueError(f"Cannot compare incomplete conditions: {left}, {right}")
    ordered_keys = sorted(expected_keys)
    left_success = np.asarray([bool(left_rows[key]["success"]) for key in ordered_keys], dtype=bool)
    right_success = np.asarray([bool(right_rows[key]["success"]) for key in ordered_keys], dtype=bool)
    difference_by_task = {
        task_id: np.asarray(
            [
                float(left_rows[key]["success"]) - float(right_rows[key]["success"])
                for key in ordered_keys
                if key[0] == task_id
            ]
        )
        for task_id in sorted({key[0] for key in ordered_keys})
    }
    low, high = _hierarchical_mean_ci(difference_by_task, repetitions, _stable_rng(seed, f"paired-{left}-{right}"))
    left_only = int(np.sum(left_success & ~right_success))
    right_only = int(np.sum(~left_success & right_success))
    task_deltas = [float(np.mean(values)) for values in difference_by_task.values()]
    return {
        "comparison": label,
        "family": family,
        "left": left,
        "right": right,
        "paired_rollouts": len(ordered_keys),
        "success_rate_difference_pp": float(np.mean(left_success) - np.mean(right_success)) * 100.0,
        "difference_ci95_low_pp": low * 100.0,
        "difference_ci95_high_pp": high * 100.0,
        "left_only_successes": left_only,
        "right_only_successes": right_only,
        "mcnemar_exact_p": _exact_mcnemar_p(left_only, right_only),
        "holm_adjusted_p": None,
        "tasks_won": sum(value > 0 for value in task_deltas),
        "tasks_lost": sum(value < 0 for value in task_deltas),
        "tasks_tied": sum(value == 0 for value in task_deltas),
        "informed_beats_random_claim_supported": None,
    }


def _paired_comparisons(
    results: dict[str, dict[TrialKey, dict[str, Any]]],
    expected_keys: set[TrialKey],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    confirmatory_complete = all(set(results[condition]) == expected_keys for condition in CONDITIONS)
    for spec in (*CONFIRMATORY_COMPARISONS, *DESCRIPTIVE_COMPARISONS):
        left, right, _, _ = spec
        if set(results[left]) == expected_keys and set(results[right]) == expected_keys:
            rows.append(_paired_comparison(*spec, results, expected_keys, repetitions, seed))
    confirmatory_rows = [row for row in rows if row["family"] != "descriptive"]
    if confirmatory_complete and len(confirmatory_rows) == len(CONFIRMATORY_COMPARISONS):
        adjusted = _holm_adjust([float(row["mcnemar_exact_p"]) for row in confirmatory_rows])
        for row, adjusted_p in zip(confirmatory_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = adjusted_p
            if row["right"] == "random_global":
                row["informed_beats_random_claim_supported"] = bool(
                    float(row["difference_ci95_low_pp"]) > 0 and adjusted_p < 0.05
                )
    return rows


def _matched_expert_scenes(
    offline_queries: Path,
    final_scenes: dict[int, list[int]],
    expected_queries_per_scene: int = 10,
) -> tuple[list[dict[str, Any]], dict[tuple[str, SceneKey], dict[str, float]]]:
    selected_methods = {value: key for key, value in OFFLINE_METHODS.items() if key in INFORMED_CONDITIONS}
    grouped: dict[tuple[str, SceneKey], list[dict[str, Any]]] = {}
    for row in _read_jsonl(offline_queries):
        condition = selected_methods.get(str(row.get("method")))
        if condition is None:
            continue
        task_id = int(row["task_id"])
        local_scene = int(row["query_episode_index"]) % 50
        if task_id in final_scenes and local_scene in final_scenes[task_id]:
            grouped.setdefault((condition, (task_id, local_scene)), []).append(row)
    expected_scene_count = sum(len(scenes) for scenes in final_scenes.values())
    scene_metrics = {}
    summary_rows = []
    for condition in INFORMED_CONDITIONS:
        condition_keys = [key for key in grouped if key[0] == condition]
        if len(condition_keys) != expected_scene_count:
            raise ValueError(
                f"{condition} matched {len(condition_keys)} expert scenes; expected {expected_scene_count}"
            )
        for key in condition_keys:
            rows = grouped[key]
            if len(rows) != expected_queries_per_scene:
                raise ValueError(f"{condition} scene {key[1]} has {len(rows)} expert queries")
            scene_metrics[key] = {
                "correct_task_fraction": float(np.mean([row["correct_task_fraction_at4"] for row in rows])),
                "top1_task_accuracy": float(np.mean([row["top1_task_accuracy"] for row in rows])),
                "task_aware_ndcg_at4": float(np.mean([row["task_aware_ndcg_at4"] for row in rows])),
            }
        metrics = [scene_metrics[key] for key in sorted(condition_keys)]
        summary_rows.append(
            {
                "condition": condition,
                "method_label": CONDITION_LABELS[condition],
                "matched_expert_queries": len(condition_keys) * expected_queries_per_scene,
                "matched_scenes": len(condition_keys),
                "correct_task_fraction": float(np.mean([value["correct_task_fraction"] for value in metrics])),
                "top1_task_accuracy": float(np.mean([value["top1_task_accuracy"] for value in metrics])),
                "task_aware_ndcg_at4": float(np.mean([value["task_aware_ndcg_at4"] for value in metrics])),
            }
        )
    return summary_rows, scene_metrics


def _distribution_shift(
    expert_scene_metrics: dict[tuple[str, SceneKey], dict[str, float]],
    traces: dict[str, dict[TrialKey, dict[str, Any]]],
    expected_keys: set[TrialKey],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for condition in INFORMED_CONDITIONS:
        if set(traces[condition]) != expected_keys:
            raise ValueError(f"{condition} needs all traces for matched-scene analysis")
        expert = []
        policy = []
        drops_by_task: dict[int, list[float]] = {}
        for trial_key in sorted(expected_keys):
            task_id, scene_id, _ = trial_key
            expert_value = expert_scene_metrics[(condition, (task_id, scene_id))]["correct_task_fraction"]
            policy_value = _trace_metric(traces[condition][trial_key], task_id)["correct_task_fraction"]
            expert.append(expert_value)
            policy.append(policy_value)
            drops_by_task.setdefault(task_id, []).append(expert_value - policy_value)
        drop_arrays = {task_id: np.asarray(values) for task_id, values in drops_by_task.items()}
        low, high = _hierarchical_mean_ci(drop_arrays, repetitions, _stable_rng(seed, f"matched-shift-{condition}"))
        rows.append(
            {
                "condition": condition,
                "method_label": CONDITION_LABELS[condition],
                "tasks": len(drop_arrays),
                "scenes": len(expert),
                "expert_correct_task_fraction": float(np.mean(expert)),
                "policy_correct_task_fraction": float(np.mean(policy)),
                "expert_minus_policy_drop_pp": float(np.mean(np.asarray(expert) - np.asarray(policy))) * 100.0,
                "drop_ci95_low_pp": low * 100.0,
                "drop_ci95_high_pp": high * 100.0,
                "interpretation": "descriptive on-policy distribution shift; not a causal effect",
            }
        )
    return rows


def _early_retrieval_feedback(
    results: dict[str, dict[TrialKey, dict[str, Any]]],
    traces: dict[str, dict[TrialKey, dict[str, Any]]],
    expected_keys: set[TrialKey],
    repetitions: int,
    seed: int,
    first_queries: int = 10,
) -> list[dict[str, Any]]:
    rows = []
    for condition in INFORMED_CONDITIONS:
        if set(results[condition]) != expected_keys or set(traces[condition]) != expected_keys:
            raise ValueError(f"{condition} needs all results and traces for early retrieval analysis")
        by_task: dict[int, list[tuple[bool, float]]] = {}
        for key in sorted(expected_keys):
            fraction = _trace_metric(traces[condition][key], key[0], first_queries)["correct_task_fraction"]
            by_task.setdefault(key[0], []).append((bool(results[condition][key]["success"]), fraction))
        task_deltas = []
        for values in by_task.values():
            successes = [fraction for success, fraction in values if success]
            failures = [fraction for success, fraction in values if not success]
            if successes and failures:
                task_deltas.append(float(np.mean(successes) - np.mean(failures)))
        delta_array = np.asarray(task_deltas)
        low, high = _task_bootstrap_mean_ci(delta_array, repetitions, _stable_rng(seed, f"early-feedback-{condition}"))
        rows.append(
            {
                "condition": condition,
                "method_label": CONDITION_LABELS[condition],
                "first_retrieval_queries": first_queries,
                "eligible_tasks_with_success_and_failure": len(delta_array),
                "successful_minus_failed_correct_task_fraction_pp": float(np.mean(delta_array)) * 100.0,
                "task_bootstrap_ci95_low_pp": low * 100.0,
                "task_bootstrap_ci95_high_pp": high * 100.0,
                "exploratory": True,
                "interpretation": "within-task association; not evidence of causal direction",
            }
        )
    return rows


def _read_offline_per_task(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = set(OFFLINE_METHODS.values())
    return {
        (str(row["method"]), int(row["task_id"])): float(row["task_aware_ndcg_at4"])
        for row in rows
        if row["method"] in selected
    }


def _spearman_task_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    def spearman(left: np.ndarray, right: np.ndarray) -> float:
        def rank(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values, kind="mergesort")
            ranked = np.empty(len(values), dtype=np.float64)
            start = 0
            while start < len(values):
                stop = start + 1
                while stop < len(values) and values[order[stop]] == values[order[start]]:
                    stop += 1
                ranked[order[start:stop]] = (start + stop - 1) / 2.0
                start = stop
            return ranked

        left_rank = rank(left)
        right_rank = rank(right)
        if np.std(left_rank) == 0 or np.std(right_rank) == 0:
            return math.nan
        return float(np.corrcoef(left_rank, right_rank)[0, 1])

    estimate = spearman(x, y)
    draws = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(x), size=len(x))
        value = spearman(x[indices], y[indices])
        if math.isfinite(value):
            draws.append(value)
    low, high = _percentile_interval(np.asarray(draws))
    return estimate, low, high


def _offline_online_correlations(
    offline: dict[tuple[str, int], float],
    task_success_rates: dict[tuple[str, int], float],
    task_ids: list[int],
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    specs = [(condition, None, f"absolute {CONDITION_LABELS[condition]}") for condition in INFORMED_CONDITIONS] + [
        ("dino_state_global", "dino_global", "DINO+State minus DINO"),
        ("pi_global", "dino_state_global", "pi0-FF minus DINO+State"),
    ]
    for left, right, label in specs:
        if right is None:
            x = np.asarray([offline[(OFFLINE_METHODS[left], task_id)] for task_id in task_ids])
            y = np.asarray([task_success_rates[(left, task_id)] for task_id in task_ids])
            kind = "absolute"
        else:
            x = np.asarray(
                [
                    offline[(OFFLINE_METHODS[left], task_id)] - offline[(OFFLINE_METHODS[right], task_id)]
                    for task_id in task_ids
                ]
            )
            y = np.asarray(
                [task_success_rates[(left, task_id)] - task_success_rates[(right, task_id)] for task_id in task_ids]
            )
            kind = "paired_delta"
        estimate, low, high = _spearman_task_bootstrap(
            x, y, repetitions, _stable_rng(seed, f"correlation-{left}-{right}")
        )
        rows.append(
            {
                "analysis": label,
                "kind": kind,
                "task_count": len(task_ids),
                "spearman_rho": estimate,
                "task_bootstrap_ci95_low": low,
                "task_bootstrap_ci95_high": high,
                "exploratory": True,
            }
        )
    return rows


def _task20_exception(
    results: dict[str, dict[TrialKey, dict[str, Any]]],
    traces: dict[str, dict[TrialKey, dict[str, Any]]],
) -> dict[str, Any]:
    keys = sorted(key for key in results["dino_global"] if key[0] == 20)
    retrieved = [
        int(item["retrieved_task_id"]) for key in keys for item in traces["dino_global"][key].get("retrieval_trace", [])
    ]
    return {
        "condition": "dino_global",
        "task_id": 20,
        "successes": sum(bool(results["dino_global"][key]["success"]) for key in keys),
        "rollouts": len(keys),
        "retrieved_slots": len(retrieved),
        "exact_task_slots": sum(task_id == 20 for task_id in retrieved),
        "retrieved_task_histogram": {str(task_id): retrieved.count(task_id) for task_id in sorted(set(retrieved))},
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_main_figure(
    output_path: Path,
    offline: dict[tuple[str, int], float],
    task_ids: list[int],
    closed_loop: list[dict[str, Any]],
    distribution_shift: list[dict[str, Any]],
    task_success_rates: dict[tuple[str, int], float],
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "dino_global": "#4477AA",
        "dino_state_global": "#228833",
        "pi_global": "#CC6677",
        "random_global": "#999999",
    }
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    offline_conditions = list(CONDITIONS)
    offline_positions = np.arange(len(offline_conditions))
    offline_means = [
        np.mean([offline[(OFFLINE_METHODS[condition], task_id)] for task_id in task_ids])
        for condition in offline_conditions
    ]
    axes[0, 0].bar(offline_positions, offline_means, color=[colors[value] for value in offline_conditions])
    axes[0, 0].set_xticks(offline_positions, [CONDITION_LABELS[value] for value in offline_conditions])
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Global nDCG@4")
    axes[0, 0].set_title("A  Held-out expert retrieval")

    complete_rows = [row for row in closed_loop if row["complete"]]
    condition_positions = np.arange(len(complete_rows))
    rates = np.asarray([row["success_rate"] for row in complete_rows], dtype=np.float64)
    lows = np.asarray([row["success_ci95_low"] for row in complete_rows], dtype=np.float64)
    highs = np.asarray([row["success_ci95_high"] for row in complete_rows], dtype=np.float64)
    bar_colors = [colors.get(str(row["condition"]), "#999999") for row in complete_rows]
    axes[0, 1].bar(
        condition_positions,
        rates,
        yerr=np.vstack((rates - lows, highs - rates)),
        capsize=3,
        color=bar_colors,
    )
    axes[0, 1].set_xticks(condition_positions, [row["method_label"] for row in complete_rows])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_ylabel("Success rate")
    axes[0, 1].set_title("B  Closed-loop control")

    width = 0.36
    informed_positions = np.arange(len(INFORMED_CONDITIONS))
    expert = [row["expert_correct_task_fraction"] for row in distribution_shift]
    policy = [row["policy_correct_task_fraction"] for row in distribution_shift]
    axes[1, 0].bar(informed_positions - width / 2, expert, width, label="Expert", color="#66CCEE")
    axes[1, 0].bar(informed_positions + width / 2, policy, width, label="On-policy", color="#AA4499")
    axes[1, 0].set_xticks(informed_positions, [CONDITION_LABELS[value] for value in INFORMED_CONDITIONS])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Exact-task retrieved fraction")
    axes[1, 0].set_title("C  Matched-scene query shift")
    axes[1, 0].legend(frameon=False)

    comparison_specs = (
        ("dino_state_global", "dino_global", "DINO+State - DINO", "#228833", "o"),
        ("pi_global", "dino_state_global", "pi0-FF - DINO+State", "#CC6677", "s"),
    )
    for left, right, label, color, marker in comparison_specs:
        x = [
            offline[(OFFLINE_METHODS[left], task_id)] - offline[(OFFLINE_METHODS[right], task_id)]
            for task_id in task_ids
        ]
        y = [task_success_rates[(left, task_id)] - task_success_rates[(right, task_id)] for task_id in task_ids]
        axes[1, 1].scatter(x, y, label=label, color=color, marker=marker, alpha=0.8)
    axes[1, 1].axhline(0, color="0.75", linewidth=1)
    axes[1, 1].axvline(0, color="0.75", linewidth=1)
    axes[1, 1].set_xlabel("Offline nDCG@4 delta")
    axes[1, 1].set_ylabel("Online success delta")
    axes[1, 1].set_title("D  Offline vs. online deltas")
    axes[1, 1].legend(frameon=False, fontsize=8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240)
    figure.savefig(output_path.with_suffix(".pdf"))
    plt.close(figure)


def analyze(
    roots: list[Path],
    protocols: list[Path],
    offline_queries_path: Path,
    offline_per_task_path: Path,
    output_dir: Path,
    repetitions: int,
    seed: int,
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    task_ids, final_scenes, expected_keys = _load_protocols(protocols)
    results, traces, error_counts = _load_rollouts(roots, expected_keys)
    incomplete = [
        condition
        for condition in CONDITIONS
        if set(results[condition]) != expected_keys or set(traces[condition]) != expected_keys
    ]
    informed_incomplete = [condition for condition in incomplete if condition in INFORMED_CONDITIONS]
    if informed_incomplete:
        raise ValueError(f"Informed conditions are incomplete: {informed_incomplete}")
    if incomplete and not allow_incomplete:
        raise ValueError(
            f"Final analysis requires 250 canonical results and traces per condition; incomplete: {incomplete}"
        )

    closed_loop, per_task, task_success_rates = _closed_loop_summaries(
        results, traces, expected_keys, error_counts, repetitions, seed
    )
    comparisons = _paired_comparisons(results, expected_keys, repetitions, seed)
    expert_summary, expert_scene_metrics = _matched_expert_scenes(offline_queries_path, final_scenes)
    distribution_shift = _distribution_shift(expert_scene_metrics, traces, expected_keys, repetitions, seed)
    early_feedback = _early_retrieval_feedback(results, traces, expected_keys, repetitions, seed)
    offline = _read_offline_per_task(offline_per_task_path)
    correlations = _offline_online_correlations(offline, task_success_rates, task_ids, repetitions, seed)
    offline_summary = [
        {
            "condition": condition,
            "method_label": CONDITION_LABELS[condition],
            "tasks": len(task_ids),
            "global_task_aware_ndcg_at4": float(
                np.mean([offline[(OFFLINE_METHODS[condition], task_id)] for task_id in task_ids])
            ),
        }
        for condition in CONDITIONS
    ]
    summary = {
        "analysis_status": "work_in_progress" if incomplete else "final_complete",
        "incomplete_conditions": incomplete,
        "bootstrap": {"repetitions": repetitions, "seed": seed},
        "validation": {
            "tasks": len(task_ids),
            "scenes_per_task": 10,
            "expected_rollouts_per_condition": len(expected_keys),
            "matched_expert_queries_per_informed_method": 2500,
            "expert_episode_mapping": "query_episode_index % 50",
            "aggregation": "equal task and scene weighting",
        },
        "closed_loop": closed_loop,
        "paired_comparisons": comparisons,
        "offline_global": offline_summary,
        "matched_expert": expert_summary,
        "matched_scene_distribution_shift": distribution_shift,
        "early_retrieval_feedback": early_feedback,
        "offline_online_correlations": correlations,
        "task20_qualitative_exception": _task20_exception(results, traces),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(output_dir / "closed_loop_conditions.csv", closed_loop)
    _write_csv(output_dir / "closed_loop_per_task.csv", per_task)
    _write_csv(output_dir / "paired_comparisons.csv", comparisons)
    _write_csv(output_dir / "offline_global.csv", offline_summary)
    _write_csv(output_dir / "matched_expert_retrieval.csv", expert_summary)
    _write_csv(output_dir / "matched_scene_distribution_shift.csv", distribution_shift)
    _write_csv(output_dir / "early_retrieval_feedback.csv", early_feedback)
    _write_csv(output_dir / "offline_online_correlations.csv", correlations)
    _make_main_figure(
        output_dir / "figures" / "main_four_panel.png",
        offline,
        task_ids,
        closed_loop,
        distribution_shift,
        task_success_rates,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--primary-protocol", type=Path, default=DEFAULT_PRIMARY_PROTOCOL)
    parser.add_argument("--extension-protocol", type=Path, default=DEFAULT_EXTENSION_PROTOCOL)
    parser.add_argument("--offline-queries", type=Path, default=DEFAULT_OFFLINE_QUERIES)
    parser.add_argument("--offline-per-task", type=Path, default=DEFAULT_OFFLINE_PER_TASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a work-in-progress report while withholding incomplete-condition estimates.",
    )
    args = parser.parse_args()
    summary = analyze(
        [args.primary_root, args.extension_root],
        [args.primary_protocol, args.extension_protocol],
        args.offline_queries,
        args.offline_per_task,
        args.output_dir,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({"analysis_status": summary["analysis_status"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
