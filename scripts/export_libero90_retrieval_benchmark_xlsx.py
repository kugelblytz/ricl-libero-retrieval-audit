#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import zipfile

from export_libero90_nointerp_solvable_xlsx import STYLE_TIE
from export_libero90_nointerp_solvable_xlsx import STYLE_WINNER
from export_libero90_nointerp_solvable_xlsx import StyledCell
from export_libero90_nointerp_solvable_xlsx import write_xlsx

PRIMARY_METRIC = "best_of_4_action_rmse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the offline LIBERO retrieval benchmark to XLSX.")
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/libero/retrieval_benchmark/libero90_solvable25_offline_v1"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/libero/reports/libero90_solvable25_offline_v1.manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/libero/reports/libero90_solvable25_offline_v1.xlsx"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_csv(args.benchmark_root / "full" / "summary.csv")
    per_task = read_csv(args.benchmark_root / "full" / "per_task.csv")
    pairwise = read_csv(args.benchmark_root / "full" / "pairwise.csv")
    queries = read_jsonl(args.benchmark_root / "full" / "queries.jsonl")
    calibration = json.loads((args.benchmark_root / "calibration.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    cache_sizes = method_cache_sizes(args.benchmark_root)

    workbook = {
        "Summary": summary_sheet(summary, cache_sizes, manifest),
        "PerTask": per_task_sheet(per_task),
        "Pairwise": mapping_rows_sheet(pairwise),
        "Queries": queries_sheet(queries),
        "Calibration": calibration_sheet(calibration),
        "Artifacts": artifacts_sheet(manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output, workbook)
    validate_xlsx(args.output, expected_sheets=set(workbook))
    print(f"Wrote {args.output}")


def summary_sheet(
    summary: list[dict[str, str]],
    cache_sizes: dict[str, int],
    manifest: dict,
) -> list[list[object]]:
    rows = [
        ["Offline LIBERO retrieval benchmark"],
        ["Status", manifest["status"]],
        ["Tasks", len(manifest["task_ids"])],
        ["Queries", manifest["queries"]],
        ["Candidate episodes", manifest["candidate_episodes"]],
        ["Candidate frames", manifest["candidate_frames"]],
        ["Primary metric", PRIMARY_METRIC, "Lower is better"],
        [],
        [
            "Rank",
            "Method",
            "Tasks",
            "Queries",
            "Best-of-4 action RMSE",
            "Top-1 action RMSE",
            "Mean top-4 action RMSE",
            "Oracle regret",
            "Progress error",
            "State error",
            "Unique episodes",
            "Retrieval latency ms",
            "Feature cache GiB",
        ],
    ]
    ordered = sorted(summary, key=lambda row: int(row["rank"]))
    for row in ordered:
        method = row["method"]
        rows.append(
            [
                int(row["rank"]),
                method,
                int(row["task_count"]),
                int(row["query_count"]),
                number(row[PRIMARY_METRIC]),
                number(row["top1_action_rmse"]),
                number(row["mean_top4_action_rmse"]),
                number(row["oracle_regret"]),
                number(row["progress_error"]),
                number(row["state_error"]),
                number(row["unique_episodes"]),
                number(row["retrieval_latency_ms"]),
                cache_sizes.get(method, 0) / 1024**3,
            ]
        )
    rows.extend(
        [
            [],
            ["Interpretation", "Each task contributes equally to the macro average."],
            ["Oracle", "oracle_action is a reporting-only upper bound and is not an input retriever."],
            ["Color legend", "Green = lowest per-task primary score; yellow = exact tie."],
        ]
    )
    return rows


def per_task_sheet(rows: list[dict[str, str]]) -> list[list[object]]:
    methods = sorted({row["method"] for row in rows})
    tasks = sorted({int(row["task_id"]) for row in rows})
    lookup = {(int(row["task_id"]), row["method"]): row for row in rows}
    output = [["Task", "Queries", *methods, "Winner(s)"]]
    for task_id in tasks:
        values = {method: number(lookup[(task_id, method)][PRIMARY_METRIC]) for method in methods}
        comparable_values = {method: value for method, value in values.items() if method != "oracle_action"}
        best = min(comparable_values.values())
        winners = [method for method, value in comparable_values.items() if value == best]
        row: list[object] = [task_id, int(lookup[(task_id, methods[0])]["query_count"])]
        for method in methods:
            style = STYLE_WINNER if len(winners) == 1 and method in winners else STYLE_TIE if method in winners else 0
            row.append(StyledCell(values[method], style))
        row.append(", ".join(winners))
        output.append(row)
    return output


def queries_sheet(rows: list[dict]) -> list[list[object]]:
    methods = sorted({str(row["method"]) for row in rows})
    grouped: dict[tuple[int, int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["query_episode_index"]), int(row["query_frame_index"]))
        grouped.setdefault(key, {})[str(row["method"])] = row
    header = ["Task", "Prompt", "Query episode", "Query frame", "Query progress"]
    for method in methods:
        header.extend([f"{method} score", f"{method} retrieved episode:frame"])
    output = [header]
    for key, method_rows in sorted(grouped.items()):
        first = next(iter(method_rows.values()))
        row: list[object] = [
            key[0],
            first["prompt"],
            key[1],
            key[2],
            first["query_progress"],
        ]
        for method in methods:
            value = method_rows[method]
            selected = ", ".join(
                f"{episode}:{frame}"
                for episode, frame in zip(
                    value["retrieved_episode_indices"],
                    value["retrieved_frame_indices"],
                    strict=True,
                )
            )
            row.extend([value[PRIMARY_METRIC], selected])
        output.append(row)
    return output


def calibration_sheet(calibration: dict) -> list[list[object]]:
    output = [
        ["Calibration tasks", ",".join(str(task) for task in calibration["calibration_tasks"])],
        ["Selection metric", calibration["selection_metric"]],
        [],
        ["Family", "Selected beta", "Selected macro best-of-4 action RMSE"],
    ]
    for family, selection in sorted(calibration["selections"].items()):
        output.append([family, selection["beta"], selection["macro_best_of_4_action_rmse"]])
    output.extend([[], ["Family", "Beta", "Task", "Queries", "Best-of-4 action RMSE"]])
    output.extend(
        [
            [
                row["family"],
                row["beta"],
                row["task_id"],
                row["query_count"],
                row["best_of_4_action_rmse"],
            ]
            for row in calibration["rows"]
        ]
    )
    return output


def artifacts_sheet(manifest: dict) -> list[list[object]]:
    rows = [
        ["Field", "Value"],
        ["Created at", manifest["created_at"]],
        ["Benchmark version", manifest["benchmark_version"]],
        ["Normal config", manifest["normal_config"]],
        ["Normal checkpoint", manifest["normal_checkpoint"]],
        ["Pi0 representation", json.dumps(manifest.get("pi0_representation", {}), sort_keys=True)],
        ["Task IDs", ",".join(str(task) for task in manifest["task_ids"])],
        ["Same-task retrieval only", manifest["same_task_retrieval_only"]],
        ["Datasets", "\n".join(manifest["datasets"])],
        ["Retrieval directories", "\n".join(manifest["retrieval_dirs"])],
        ["CUDA", manifest["cuda"]],
        ["Git commit", manifest["git_commit"]],
        ["Git status", manifest["git_status"]],
        ["Command", manifest["command"]],
    ]
    for name, path in sorted(manifest["outputs"].items()):
        rows.append([f"Output: {name}", path])
    for name, digest in sorted(manifest["checkpoint_identity"].items()):
        rows.append([f"Checkpoint SHA256: {name}", digest])
    for name, version in sorted(manifest["packages"].items()):
        rows.append([f"Package: {name}", version])
    rows.append(["Git diff", manifest["git_diff"]])
    return rows


def mapping_rows_sheet(rows: list[dict[str, str]]) -> list[list[object]]:
    if not rows:
        return [[]]
    keys = list(rows[0])
    return [keys, *[[coerce(row[key]) for key in keys] for row in rows]]


def method_cache_sizes(root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    aliases = {
        "dino_top": ("dino_top",),
        "dino_top_wrist": ("dino_top", "dino_wrist"),
        "dino_top_state": ("dino_top",),
        "dino_top_wrist_state": ("dino_top", "dino_wrist"),
        "pi0_fast_first_fusion": ("pi0_fast_first_fusion",),
        "pi0_fast_first_fusion_no_prompt": ("pi0_fast_first_fusion_no_prompt",),
    }
    for method, feature_names in aliases.items():
        total = 0
        for feature_name in feature_names:
            total += sum(path.stat().st_size for path in (root / "cache").glob(f"**/{feature_name}/task_*.npy"))
        sizes[method] = total
    return sizes


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value: object) -> float:
    return float(value)


def coerce(value: str) -> object:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def validate_xlsx(path: Path, *, expected_sheets: set[str]) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Generated report is not a valid XLSX archive: {path}")
    with zipfile.ZipFile(path) as archive:
        workbook = archive.read("xl/workbook.xml").decode()
        missing = [sheet for sheet in expected_sheets if f'name="{sheet}"' not in workbook]
        if missing:
            raise ValueError(f"Generated XLSX is missing sheets: {missing}")


if __name__ == "__main__":
    main()
