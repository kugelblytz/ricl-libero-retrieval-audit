from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FINAL_COMPARISONS = (
    ("dino_state_global", "dino_global", "DINO+state minus DINO"),
    ("pi_global", "dino_state_global", "Pi0-FF minus DINO+state"),
    ("dino_global", "random_global", "DINO minus random"),
    ("dino_state_global", "random_global", "DINO+state minus random"),
    ("pi_global", "random_global", "Pi0-FF minus random"),
)
CONDITION_LABELS = {
    "dino_global": "DINO",
    "dino_state_global": "DINO+state",
    "pi_global": "Pi0-FF",
    "random_global": "random",
}
OFFLINE_METHODS = {
    "dino_global": "dino_top_global",
    "dino_state_global": "dino_top_state_global",
    "pi_global": "pi0_fast_first_fusion_global",
}
DEFAULT_OFFLINE_PER_TASK = Path(
    "data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1/full/per_task.csv"
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSON at {path}:{line_number}: {error}") from error
    return rows


def _trial_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])


def _canonical_valid(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    valid = {}
    for row in rows:
        if row.get("error") is None:
            valid[_trial_key(row)] = row
    return valid


def _exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    total = discordant_a + discordant_b
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(discordant_a, discordant_b) + 1)) / 2**total
    return min(1.0, 2.0 * tail)


def _by_task(
    rows: dict[tuple[int, int, int], dict[str, Any]],
    *,
    value_key: str = "success",
) -> dict[int, np.ndarray]:
    grouped: dict[int, list[float]] = {}
    for key in sorted(rows):
        grouped.setdefault(key[0], []).append(float(rows[key][value_key]))
    return {task_id: np.asarray(values, dtype=np.float64) for task_id, values in grouped.items()}


def _hierarchical_rate_ci(
    rows: dict[tuple[int, int, int], dict[str, Any]], repetitions: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    grouped = _by_task(rows)
    if not grouped or repetitions <= 0:
        return None, None
    task_values = list(grouped.values())
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_tasks = rng.integers(0, len(task_values), size=len(task_values))
        resampled = []
        for task_position in selected_tasks:
            values = task_values[int(task_position)]
            resampled.append(values[rng.integers(0, len(values), size=len(values))])
        draws[repetition] = float(np.mean(np.concatenate(resampled)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _hierarchical_paired_ci(
    left: dict[tuple[int, int, int], dict[str, Any]],
    right: dict[tuple[int, int, int], dict[str, Any]],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    shared = sorted(set(left) & set(right))
    grouped: dict[int, list[float]] = {}
    for key in shared:
        grouped.setdefault(key[0], []).append(float(left[key]["success"]) - float(right[key]["success"]))
    if not grouped or repetitions <= 0:
        return None, None
    task_values = [np.asarray(values, dtype=np.float64) for values in grouped.values()]
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected_tasks = rng.integers(0, len(task_values), size=len(task_values))
        resampled = []
        for task_position in selected_tasks:
            values = task_values[int(task_position)]
            resampled.append(values[rng.integers(0, len(values), size=len(values))])
        draws[repetition] = float(np.mean(np.concatenate(resampled))) * 100.0
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _request_mean(row: dict[str, Any]) -> float | None:
    request = row.get("timings_ms", {}).get("request_total")
    if not isinstance(request, dict) or "mean" not in request:
        return None
    value = float(request["mean"])
    return value if math.isfinite(value) else None


def _trace_rates(
    traces: dict[tuple[int, int, int], dict[str, Any]], task_id: int | None = None
) -> tuple[float | None, float | None]:
    items = [
        (key, item)
        for key, trace in traces.items()
        if task_id is None or key[0] == task_id
        for item in trace.get("retrieval_trace", [])
    ]
    if not items:
        return None, None
    correct_fraction = sum(int(item["retrieved_task_id"]) == key[0] for key, item in items) / len(items)
    top1 = [(key, item) for key, item in items if int(item.get("rank", -1)) == 0]
    top1_rate = (
        sum(int(item["retrieved_task_id"]) == key[0] for key, item in top1) / len(top1) if top1 else None
    )
    return top1_rate, correct_fraction


def summarize(
    output_root: Path,
    *,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 7,
) -> dict[str, Any]:
    manifest = json.loads((output_root / "run_manifest.json").read_text())
    condition_rows: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    condition_traces: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    summary_rows = []
    per_task_rows = []
    expected_per_condition = manifest.get("expected_trials_per_condition")
    rng = np.random.default_rng(bootstrap_seed)

    for condition in manifest["conditions"]:
        all_rows = _read_jsonl(output_root / condition / "results.jsonl")
        valid = _canonical_valid(all_rows)
        traces = _canonical_valid(_read_jsonl(output_root / condition / "retrieval_trace.jsonl"))
        condition_rows[condition] = valid
        condition_traces[condition] = traces
        successes = sum(bool(row["success"]) for row in valid.values())
        request_means = [value for row in valid.values() if (value := _request_mean(row)) is not None]
        grouped = _by_task(valid)
        task_rates = [float(np.mean(values)) for values in grouped.values()]
        ci_low, ci_high = _hierarchical_rate_ci(valid, bootstrap_repetitions, rng)
        top1_rate, correct_fraction = _trace_rates(traces)
        summary_rows.append(
            {
                "condition": condition,
                "method_label": CONDITION_LABELS.get(condition, condition),
                "valid_trials": len(valid),
                "infrastructure_error_rows": sum(row.get("error") is not None for row in all_rows),
                "unresolved_infrastructure_errors": (
                    max(int(expected_per_condition) - len(valid), 0) if expected_per_condition is not None else 0
                ),
                "successes": successes,
                "success_rate": successes / len(valid) if valid else None,
                "equal_weight_task_success_rate": float(np.mean(task_rates)) if task_rates else None,
                "success_ci95_low": ci_low,
                "success_ci95_high": ci_high,
                "mean_request_ms": float(np.mean(request_means)) if request_means else None,
                "correct_task_top1_rate": top1_rate,
                "correct_task_retrieval_fraction": correct_fraction,
                # Retained for compatibility with pilot reports.
                "cross_task_retrieval_rate": None if correct_fraction is None else 1.0 - correct_fraction,
            }
        )

        for task_id, task_values in sorted(grouped.items()):
            task_valid = {key: row for key, row in valid.items() if key[0] == task_id}
            task_request_means = [
                value for row in task_valid.values() if (value := _request_mean(row)) is not None
            ]
            task_top1, task_correct_fraction = _trace_rates(traces, task_id)
            per_task_rows.append(
                {
                    "condition": condition,
                    "method_label": CONDITION_LABELS.get(condition, condition),
                    "task_id": task_id,
                    "successes": int(np.sum(task_values)),
                    "trials": len(task_values),
                    "success_rate": float(np.mean(task_values)),
                    "mean_episode_length": float(
                        np.mean([float(row["elapsed_steps"]) for row in task_valid.values()])
                    ),
                    "correct_task_top1_rate": task_top1,
                    "correct_task_retrieval_rate": task_correct_fraction,
                    "mean_request_ms": float(np.mean(task_request_means)) if task_request_means else None,
                }
            )

    comparisons = []
    for left, right, label in FINAL_COMPARISONS:
        if left not in condition_rows or right not in condition_rows:
            continue
        shared = sorted(set(condition_rows[left]) & set(condition_rows[right]))
        left_successes = np.asarray([bool(condition_rows[left][key]["success"]) for key in shared], dtype=bool)
        right_successes = np.asarray([bool(condition_rows[right][key]["success"]) for key in shared], dtype=bool)
        left_only = int(np.sum(left_successes & ~right_successes))
        right_only = int(np.sum(right_successes & ~left_successes))
        task_deltas = []
        for task_id in sorted({key[0] for key in shared}):
            positions = [position for position, key in enumerate(shared) if key[0] == task_id]
            task_deltas.append(float(np.mean(left_successes[positions]) - np.mean(right_successes[positions])))
        ci_low, ci_high = _hierarchical_paired_ci(
            condition_rows[left], condition_rows[right], bootstrap_repetitions, rng
        )
        comparisons.append(
            {
                "comparison": label,
                "left": left,
                "right": right,
                "paired_trials": len(shared),
                "left_success_rate": float(np.mean(left_successes)) if shared else None,
                "right_success_rate": float(np.mean(right_successes)) if shared else None,
                "success_rate_difference_pp": (
                    float(np.mean(left_successes) - np.mean(right_successes)) * 100.0 if shared else None
                ),
                "difference_ci95_low_pp": ci_low,
                "difference_ci95_high_pp": ci_high,
                "left_only_successes": left_only,
                "right_only_successes": right_only,
                "mcnemar_exact_p": _exact_mcnemar_p(left_only, right_only),
                "tasks_won": sum(delta > 0 for delta in task_deltas),
                "tasks_lost": sum(delta < 0 for delta in task_deltas),
                "tasks_tied": sum(delta == 0 for delta in task_deltas),
            }
        )
    adjusted = _holm_adjust([float(row["mcnemar_exact_p"]) for row in comparisons])
    for row, adjusted_p in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = adjusted_p

    return {
        "manifest": manifest,
        "bootstrap": {"repetitions": bootstrap_repetitions, "seed": bootstrap_seed},
        "conditions": summary_rows,
        "per_task": per_task_rows,
        "paired_comparisons": comparisons,
    }


def _read_offline_per_task(path: Path) -> dict[tuple[str, int], float]:
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open()))
    return {
        (row["method"], int(row["task_id"])): float(row["task_aware_ndcg_at4"])
        for row in rows
        if row["method"] in OFFLINE_METHODS.values()
    }


def _spearman_task_bootstrap(
    x: np.ndarray, y: np.ndarray, repetitions: int, rng: np.random.Generator
) -> tuple[float | None, float | None, float | None]:
    from scipy import stats

    if len(x) < 2:
        return None, None, None
    estimate = float(stats.spearmanr(x, y).statistic)
    draws = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(x), size=len(x))
        value = float(stats.spearmanr(x[indices], y[indices]).statistic)
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        return estimate, None, None
    low, high = np.percentile(draws, [2.5, 97.5])
    return estimate, float(low), float(high)


def _write_final_analysis(
    output_root: Path,
    summary: dict[str, Any],
    offline_per_task: Path,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> None:
    required = set(CONDITION_LABELS)
    present = {row["condition"] for row in summary["conditions"]}
    offline = _read_offline_per_task(offline_per_task)
    if not required.issubset(present) or not offline:
        return

    import matplotlib.pyplot as plt

    per_task = {(row["condition"], int(row["task_id"])): row for row in summary["per_task"]}
    online_tasks = sorted({task_id for _, task_id in per_task})
    offline_tasks = sorted(
        task_id for method, task_id in offline if method == OFFLINE_METHODS["dino_global"]
    )
    offline_rows = []
    for condition, offline_method in OFFLINE_METHODS.items():
        values = [offline[(offline_method, task_id)] for task_id in offline_tasks]
        offline_rows.append(
            {
                "condition": condition,
                "method_label": CONDITION_LABELS[condition],
                "task_count": len(values),
                "mean_task_aware_ndcg_at4": float(np.mean(values)),
            }
        )
    _write_csv(output_root / "offline_ndcg_summary.csv", offline_rows)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar([row["method_label"] for row in offline_rows], [row["mean_task_aware_ndcg_at4"] for row in offline_rows])
    axis.set_ylabel("Global offline task-aware nDCG@4")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(figure_dir / "global_offline_ndcg.png", dpi=180)
    plt.close(figure)

    condition_rows = {row["condition"]: row for row in summary["conditions"]}
    ordered_conditions = list(CONDITION_LABELS)
    rates = np.asarray([condition_rows[condition]["success_rate"] for condition in ordered_conditions])
    low = np.asarray([condition_rows[condition]["success_ci95_low"] for condition in ordered_conditions])
    high = np.asarray([condition_rows[condition]["success_ci95_high"] for condition in ordered_conditions])
    low = np.minimum(low, rates)
    high = np.maximum(high, rates)
    figure, axis = plt.subplots(figsize=(7, 4))
    positions = np.arange(len(ordered_conditions))
    axis.bar(positions, rates, yerr=np.vstack((rates - low, high - rates)), capsize=4)
    axis.set_xticks(positions, [CONDITION_LABELS[condition] for condition in ordered_conditions])
    axis.set_ylabel("Closed-loop success rate")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(figure_dir / "closed_loop_success.png", dpi=180)
    plt.close(figure)

    heatmap = np.asarray(
        [
            [per_task[(condition, task_id)]["success_rate"] for task_id in online_tasks]
            for condition in ordered_conditions
        ]
    )
    figure, axis = plt.subplots(figsize=(12, 3.5))
    image = axis.imshow(heatmap, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(online_tasks)), online_tasks, rotation=90)
    axis.set_yticks(np.arange(len(ordered_conditions)), [CONDITION_LABELS[value] for value in ordered_conditions])
    axis.set_xlabel("LIBERO task ID")
    figure.colorbar(image, ax=axis, label="Success rate")
    figure.tight_layout()
    figure.savefig(figure_dir / "per_task_closed_loop_heatmap.png", dpi=180)
    plt.close(figure)

    correlation_specs = (
        ("dino_state_global", "dino_global", "DINO+state minus DINO"),
        ("pi_global", "dino_state_global", "Pi0-FF minus DINO+state"),
    )
    correlation_rows = []
    scatter_values = []
    rng = np.random.default_rng(bootstrap_seed + 1)
    for left, right, label in correlation_specs:
        x = np.asarray(
            [
                offline[(OFFLINE_METHODS[left], task_id)] - offline[(OFFLINE_METHODS[right], task_id)]
                for task_id in online_tasks
            ]
        )
        y = np.asarray(
            [
                per_task[(left, task_id)]["success_rate"] - per_task[(right, task_id)]["success_rate"]
                for task_id in online_tasks
            ]
        )
        estimate, low_ci, high_ci = _spearman_task_bootstrap(x, y, bootstrap_repetitions, rng)
        correlation_rows.append(
            {
                "comparison": label,
                "task_count": len(online_tasks),
                "spearman_rho": estimate,
                "task_bootstrap_ci95_low": low_ci,
                "task_bootstrap_ci95_high": high_ci,
                "exploratory": True,
            }
        )
        scatter_values.append((label, x, y))
    _write_csv(output_root / "offline_online_correlations.csv", correlation_rows)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (label, x, y) in zip(axes, scatter_values, strict=True):
        axis.scatter(x, y)
        axis.axhline(0, color="0.7", linewidth=1)
        axis.axvline(0, color="0.7", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Offline nDCG@4 delta")
    axes[0].set_ylabel("Online success-rate delta")
    figure.tight_layout()
    figure.savefig(figure_dir / "offline_vs_online_paired_delta.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    for condition in ordered_conditions:
        x = [per_task[(condition, task_id)]["correct_task_retrieval_rate"] for task_id in online_tasks]
        y = [per_task[(condition, task_id)]["success_rate"] for task_id in online_tasks]
        axis.scatter(x, y, label=CONDITION_LABELS[condition], alpha=0.8)
    axis.set_xlabel("Correct-task retrieval fraction")
    axis.set_ylabel("Closed-loop success rate")
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.03, 1.03)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "correct_task_retrieval_vs_success.png", dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7)
    parser.add_argument("--offline-per-task", type=Path, default=DEFAULT_OFFLINE_PER_TASK)
    args = parser.parse_args()
    summary = summarize(
        args.output_root,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.output_root / "condition_summary.csv", summary["conditions"])
    _write_csv(args.output_root / "per_task_results.csv", summary["per_task"])
    _write_csv(args.output_root / "paired_comparisons.csv", summary["paired_comparisons"])
    _write_final_analysis(
        args.output_root,
        summary,
        args.offline_per_task,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
