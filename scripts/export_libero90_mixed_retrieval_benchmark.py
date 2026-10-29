#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import zipfile

import matplotlib as mpl

mpl.use("Agg")
from export_libero90_nointerp_solvable_xlsx import STYLE_TIE
from export_libero90_nointerp_solvable_xlsx import STYLE_WINNER
from export_libero90_nointerp_solvable_xlsx import StyledCell
from export_libero90_nointerp_solvable_xlsx import write_xlsx
import matplotlib.pyplot as plt
import numpy as np

PRIMARY_METRIC = "task_aware_ndcg_at4"
PRACTICAL_METHODS = (
    "state_global",
    "dino_top_global",
    "dino_top_wrist_global",
    "dino_top_state_global",
    "dino_top_wrist_state_global",
    "pi0_fast_first_fusion_global",
    "pi0_fast_first_fusion_no_prompt_global",
    "pi0_fast_first_fusion_shuffled_prompt_global",
    "prompt_gated_dino_top",
)
DISPLAY_NAMES = {
    "random_global": "Random",
    "state_global": "State",
    "dino_top_global": "DINO top",
    "dino_top_wrist_global": "DINO top+wrist",
    "dino_top_state_global": "DINO top+state",
    "dino_top_wrist_state_global": "DINO top+wrist+state",
    "pi0_fast_first_fusion_global": "Pi0 full",
    "pi0_fast_first_fusion_no_prompt_global": "Pi0 no prompt",
    "pi0_fast_first_fusion_shuffled_prompt_global": "Pi0 shuffled prompt",
    "prompt_gated_dino_top": "Prompt-gated DINO",
    "task_oracle_gate_dino_top": "Task-gated DINO",
    "oracle_action_task": "Action oracle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mixed-task LIBERO retrieval results.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/libero/reports/libero90_solvable25_mixed_global_v1.manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/libero/reports/libero90_solvable25_mixed_global_v1.xlsx"),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("data/libero/reports/libero90_solvable25_mixed_global_v1_plots"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full = args.benchmark_root / "full"
    summary = read_csv(full / "summary.csv")
    per_task = read_csv(full / "per_task.csv")
    pairwise = read_csv(full / "pairwise.csv")
    confusion = read_csv(full / "confusion.csv")
    queries = read_jsonl(full / "queries.jsonl")
    manifest = json.loads(args.manifest.read_text())
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = create_plots(summary, per_task, confusion, args.plots_dir)

    workbook = {
        "Summary": summary_sheet(summary, manifest),
        "PerTask": per_task_sheet(per_task),
        "PromptStress": prompt_stress_sheet(summary, per_task),
        "OracleProximity": oracle_proximity_sheet(summary),
        "Pairwise": mapping_rows_sheet(pairwise),
        "Confusion": mapping_rows_sheet(confusion),
        "Queries": queries_sheet(queries),
        "Visuals": [["Visualization", "Path"], *[[name, str(path)] for name, path in plot_paths.items()]],
        "Artifacts": artifacts_sheet(manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output, workbook)
    validate_xlsx(args.output, set(workbook))
    print(f"Wrote {args.output}")
    for path in plot_paths.values():
        print(f"Wrote {path}")


def summary_sheet(rows: list[dict[str, str]], manifest: dict) -> list[list[object]]:
    output = [
        ["Mixed-task LIBERO retrieval benchmark"],
        ["Status", manifest["status"]],
        ["Global candidate tasks", manifest["candidate_tasks"]],
        ["Candidate frames", manifest["candidate_frames"]],
        ["Held-out queries", manifest["queries"]],
        ["Primary metric", "Task-aware nDCG@4", "Higher is better"],
        [
            "Relevance",
            "Wrong-task retrievals receive zero credit; correct-task frames are graded by future-action percentile.",
        ],
        [],
        [
            "Rank",
            "Method",
            "Task-aware nDCG@4",
            "Top-1 task accuracy",
            "Task recall@4",
            "Correct-task fraction@4",
            "Useful retrieval@4",
            "Conditional action RMSE",
            "Conditional matched queries",
            "Any oracle overlap@4",
            "Exact oracle top-1",
            "Latency ms",
        ],
    ]
    output.extend(
        [
            [
                int(row["rank"]),
                DISPLAY_NAMES.get(row["method"], row["method"]),
                number(row[PRIMARY_METRIC]),
                number(row["top1_task_accuracy"]),
                number(row["task_recall_at4"]),
                number(row["correct_task_fraction_at4"]),
                number(row["useful_retrieval_at4"]),
                number(row["conditional_best_action_rmse"]),
                int(row["conditional_query_count"]),
                number(row["any_oracle_overlap_at4"]),
                number(row["exact_oracle_top1"]),
                number(row["retrieval_latency_ms"]),
            ]
            for row in sorted(rows, key=lambda value: int(value["rank"]))
        ]
    )
    output.extend(
        [
            [],
            ["Task-gated DINO", "Control that uses the known task ID; it reproduces the previous same-task benchmark."],
            ["Action oracle", "Reporting-only top four selected from correct-task future expert actions."],
            ["Conditional RMSE", "Averaged only over queries where at least one correct-task frame was retrieved."],
        ]
    )
    return output


def per_task_sheet(rows: list[dict[str, str]]) -> list[list[object]]:
    methods = [method for method in PRACTICAL_METHODS if any(row["method"] == method for row in rows)]
    tasks = sorted({int(row["task_id"]) for row in rows})
    lookup = {(int(row["task_id"]), row["method"]): row for row in rows}
    output = [["Task", "Queries", *[DISPLAY_NAMES[method] for method in methods], "Winner(s)"]]
    for task_id in tasks:
        values = {method: number(lookup[(task_id, method)][PRIMARY_METRIC]) for method in methods}
        best = max(values.values())
        winners = [method for method, value in values.items() if value == best]
        row: list[object] = [task_id, int(lookup[(task_id, methods[0])]["query_count"])]
        for method in methods:
            style = STYLE_WINNER if len(winners) == 1 and method in winners else STYLE_TIE if method in winners else 0
            row.append(StyledCell(values[method], style))
        row.append(", ".join(DISPLAY_NAMES[method] for method in winners))
        output.append(row)
    return output


def prompt_stress_sheet(
    summary: list[dict[str, str]],
    per_task: list[dict[str, str]],
) -> list[list[object]]:
    methods = (
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
        "prompt_gated_dino_top",
        "task_oracle_gate_dino_top",
    )
    lookup = {row["method"]: row for row in summary}
    output = [
        ["Prompt sensitivity"],
        [
            "Method",
            "Task-aware nDCG@4",
            "Top-1 task accuracy",
            "Task recall@4",
            "Useful retrieval@4",
        ],
    ]
    for method in methods:
        row = lookup[method]
        output.append(
            [
                DISPLAY_NAMES[method],
                number(row[PRIMARY_METRIC]),
                number(row["top1_task_accuracy"]),
                number(row["task_recall_at4"]),
                number(row["useful_retrieval_at4"]),
            ]
        )
    output.extend([[], ["Task", *[DISPLAY_NAMES[method] for method in methods]]])
    task_lookup = {(int(row["task_id"]), row["method"]): row for row in per_task}
    output.extend(
        [
            [
                task_id,
                *[number(task_lookup[(task_id, method)][PRIMARY_METRIC]) for method in methods],
            ]
            for task_id in sorted({int(row["task_id"]) for row in per_task})
        ]
    )
    return output


def oracle_proximity_sheet(summary: list[dict[str, str]]) -> list[list[object]]:
    output = [
        ["Distance to task-aware action oracle"],
        [
            "Method",
            "Task-aware nDCG@4",
            "Any exact oracle frame in top 4",
            "Exact oracle top 1",
            "Mean oracle overlap count",
            "Conditional oracle regret",
        ],
    ]
    output.extend(
        [
            [
                DISPLAY_NAMES.get(row["method"], row["method"]),
                number(row[PRIMARY_METRIC]),
                number(row["any_oracle_overlap_at4"]),
                number(row["exact_oracle_top1"]),
                number(row["oracle_overlap_count_at4"]),
                number(row["conditional_oracle_regret"]),
            ]
            for row in sorted(summary, key=lambda value: -number(value[PRIMARY_METRIC]))
        ]
    )
    return output


def queries_sheet(rows: list[dict]) -> list[list[object]]:
    methods = sorted({str(row["method"]) for row in rows})
    grouped: dict[tuple[int, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (
            int(row["task_id"]),
            int(row["query_episode_index"]),
            int(row["query_frame_index"]),
        )
        grouped.setdefault(key, {})[str(row["method"])] = row
    header = ["Task", "Prompt", "Query episode", "Query frame"]
    for method in methods:
        header.extend(
            [
                f"{DISPLAY_NAMES.get(method, method)} nDCG",
                f"{DISPLAY_NAMES.get(method, method)} retrieved task:episode:frame",
            ]
        )
    output = [header]
    for key, method_rows in sorted(grouped.items()):
        first = next(iter(method_rows.values()))
        row: list[object] = [key[0], first["prompt"], key[1], key[2]]
        for method in methods:
            value = method_rows[method]
            selected = ", ".join(
                f"{task}:{episode}:{frame}"
                for task, episode, frame in zip(
                    value["retrieved_task_ids"],
                    value["retrieved_episode_indices"],
                    value["retrieved_frame_indices"],
                    strict=True,
                )
            )
            row.extend([value[PRIMARY_METRIC], selected])
        output.append(row)
    return output


def artifacts_sheet(manifest: dict) -> list[list[object]]:
    output = [
        ["Field", "Value"],
        ["Created", manifest["created_at"]],
        ["Benchmark version", manifest["benchmark_version"]],
        ["Global mixed index", manifest["mixed_task_global_index"]],
        ["Task ID given to global retrievers", manifest["task_id_available_to_global_retrievers"]],
        ["Task IDs", ",".join(str(task) for task in manifest["task_ids"])],
        ["Normal config", manifest["normal_config"]],
        ["Normal checkpoint", manifest["normal_checkpoint"]],
        ["Primary metric", manifest["primary_metric"]],
        ["Task-aware relevance", manifest["task_aware_relevance"]],
        ["Useful retrieval definition", manifest["useful_retrieval_definition"]],
        ["Pi0 representation", json.dumps(manifest["pi0_representation"], sort_keys=True)],
        ["CUDA", json.dumps(manifest["cuda"], sort_keys=True)],
        ["Git commit", manifest["git_commit"]],
        ["Git status", manifest["git_status"]],
        ["Command", manifest["command"]],
    ]
    for name, path in sorted(manifest["outputs"].items()):
        output.append([f"Output: {name}", path])
    for name, digest in sorted(manifest["checkpoint_identity"].items()):
        output.append([f"Checkpoint SHA256: {name}", digest])
    for name, version in sorted(manifest["packages"].items()):
        output.append([f"Package: {name}", version])
    output.append(["Git diff", manifest["git_diff"]])
    return output


def create_plots(
    summary: list[dict[str, str]],
    per_task: list[dict[str, str]],
    confusion: list[dict[str, str]],
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "Overview": output_dir / "overview.png",
        "Tradeoff": output_dir / "task_recall_vs_action_error.png",
        "Prompt confusion": output_dir / "pi0_prompt_confusion.png",
        "Per-task heatmap": output_dir / "per_task_ndcg_heatmap.png",
        "Oracle proximity": output_dir / "oracle_proximity.png",
    }
    plot_overview(summary, paths["Overview"])
    plot_tradeoff(summary, paths["Tradeoff"])
    plot_prompt_confusion(confusion, paths["Prompt confusion"])
    plot_per_task(per_task, paths["Per-task heatmap"])
    plot_oracle_proximity(summary, paths["Oracle proximity"])
    return paths


def plot_overview(rows: list[dict[str, str]], output: Path) -> None:
    practical = [row for row in rows if row["method"] in PRACTICAL_METHODS]
    practical.sort(key=lambda row: number(row[PRIMARY_METRIC]))
    labels = [DISPLAY_NAMES[row["method"]] for row in practical]
    colors = ["#2f7d32" if row["method"] == "pi0_fast_first_fusion_global" else "#3572a5" for row in practical]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(labels, [number(row[PRIMARY_METRIC]) for row in practical], color=colors)
    axes[0].set_xlabel("Task-aware nDCG@4 (higher is better)")
    axes[0].set_title("Overall mixed-task retrieval quality")
    axes[0].set_xlim(0, 1)
    axes[1].barh(labels, [number(row["task_recall_at4"]) for row in practical], color=colors)
    axes[1].set_xlabel("Queries with a correct-task result in top 4")
    axes[1].set_title("Correct-task recall")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, str]], output: Path) -> None:
    practical = [row for row in rows if row["method"] in PRACTICAL_METHODS]
    fig, ax = plt.subplots(figsize=(10, 7))
    for row in practical:
        x = number(row["task_recall_at4"])
        y = number(row["conditional_best_action_rmse"])
        ax.scatter(x, y, s=70)
        ax.annotate(DISPLAY_NAMES[row["method"]], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Task recall@4 (higher is better)")
    ax.set_ylabel("Conditional action RMSE (lower is better)")
    ax.set_title("Task selection versus within-task phase selection")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_prompt_confusion(rows: list[dict[str, str]], output: Path) -> None:
    methods = (
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
    )
    tasks = sorted(
        {int(row["query_task_id"]) for row in rows if row["method"] in methods}
        | {int(row["retrieved_top1_task_id"]) for row in rows if row["method"] in methods}
    )
    index = {task: position for position, task in enumerate(tasks)}
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for ax, method in zip(axes, methods, strict=True):
        matrix = np.zeros((len(tasks), len(tasks)), dtype=np.float64)
        for row in rows:
            if row["method"] != method:
                continue
            matrix[index[int(row["query_task_id"])]][index[int(row["retrieved_top1_task_id"])]] += int(row["count"])
        totals = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(matrix / totals, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(DISPLAY_NAMES[method])
        ax.set_xlabel("Retrieved task")
        ax.set_ylabel("Query task")
        ax.set_xticks(range(len(tasks)), tasks, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tasks)), tasks, fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.75, label="Top-1 fraction")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_per_task(rows: list[dict[str, str]], output: Path) -> None:
    methods = list(PRACTICAL_METHODS)
    tasks = sorted({int(row["task_id"]) for row in rows})
    lookup = {(row["method"], int(row["task_id"])): number(row[PRIMARY_METRIC]) for row in rows}
    matrix = np.asarray([[lookup[(method, task)] for task in tasks] for method in methods])
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tasks)), tasks, rotation=90)
    ax.set_yticks(range(len(methods)), [DISPLAY_NAMES[method] for method in methods])
    ax.set_xlabel("LIBERO task")
    ax.set_title("Task-aware nDCG@4 by task")
    fig.colorbar(image, ax=ax, label="nDCG@4")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_oracle_proximity(rows: list[dict[str, str]], output: Path) -> None:
    practical = [row for row in rows if row["method"] in PRACTICAL_METHODS]
    practical.sort(key=lambda row: number(row["any_oracle_overlap_at4"]))
    labels = [DISPLAY_NAMES[row["method"]] for row in practical]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, [number(row["any_oracle_overlap_at4"]) for row in practical], color="#4c78a8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Queries sharing at least one exact frame with the action oracle")
    ax.set_title("Exact oracle-frame overlap")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def mapping_rows_sheet(rows: list[dict[str, str]]) -> list[list[object]]:
    if not rows:
        return [[]]
    keys = list(rows[0])
    return [keys, *[[coerce(row.get(key, "")) for key in keys] for row in rows]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value: object) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def coerce(value: str) -> object:
    if value == "":
        return ""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def validate_xlsx(path: Path, expected_sheets: set[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"XLSX was not created: {path}")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"XLSX archive is corrupt: {path}")
        workbook = archive.read("xl/workbook.xml").decode()
    missing = [sheet for sheet in expected_sheets if f'name="{sheet}"' not in workbook]
    if missing:
        raise ValueError(f"XLSX is missing sheets: {missing}")


if __name__ == "__main__":
    main()
