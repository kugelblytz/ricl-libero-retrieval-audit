from __future__ import annotations

from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
import dataclasses
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

REPORT_MD = Path("data/libero/reports/ricl_libero_findings_20260703.md")
REPORT_PDF = Path("data/libero/reports/ricl_libero_findings_20260703.pdf")
AUDIT_PATH = Path("data/libero/audits/ricl_libero90_finetune_episode_map.json")
MAX_DISTANCE = 158.5906982421875
RICL_PAPER_URL = "https://arxiv.org/html/2508.02062v1"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    label: str
    short_label: str
    trace_path: Path
    expected_successes: int
    expected_rollouts: int
    retrieval_demos_per_task: int
    eval_scenes_per_task: int
    normalized_note: str


RUNS = (
    RunConfig(
        label="20 retrieval demos/task, 10 held-out eval scenes/task",
        short_label="20mem_leftout10",
        trace_path=Path("data/libero/videos/libero90_leftout10/retrieval_trace.jsonl"),
        expected_successes=52,
        expected_rollouts=100,
        retrieval_demos_per_task=20,
        eval_scenes_per_task=10,
        normalized_note="Observed 10 rollouts/task.",
    ),
    RunConfig(
        label="40 retrieval demos/task, 10 held-out eval scenes/task",
        short_label="40mem_leftout10",
        trace_path=Path("data/libero/videos/libero90_leftout10_spread40_20260701_133248/retrieval_trace.jsonl"),
        expected_successes=49,
        expected_rollouts=100,
        retrieval_demos_per_task=40,
        eval_scenes_per_task=10,
        normalized_note="Observed 10 rollouts/task.",
    ),
    RunConfig(
        label="49 retrieval demos/task, 1 held-out eval scene/task",
        short_label="49mem_leftout1",
        trace_path=Path("data/libero/videos/libero90_leftout1_spread49_20260701_133248/retrieval_trace.jsonl"),
        expected_successes=5,
        expected_rollouts=10,
        retrieval_demos_per_task=49,
        eval_scenes_per_task=1,
        normalized_note="Projected each 1-rollout task result to 10 rollouts/task.",
    ),
    RunConfig(
        label="Mismatched retrieval, successful-at-least-once tasks only",
        short_label="mismatch_successful_tasks",
        trace_path=Path("data/libero/videos/libero90_successful_tasks_mismatch_20260703_102406/retrieval_trace.jsonl"),
        expected_successes=36,
        expected_rollouts=60,
        retrieval_demos_per_task=10,
        eval_scenes_per_task=10,
        normalized_note="Adds tasks 22, 23, 26, and 27 as assumed 0/10.",
    ),
)


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "it",
    "of",
    "on",
    "the",
    "to",
}

OBJECT_PHRASES = (
    "black bowl",
    "bottom drawer",
    "cabinet",
    "drawer",
    "frying pan",
    "stove",
    "top drawer",
    "wine bottle",
    "wine rack",
)

TASK_CLOSENESS_OVERRIDES = {
    20: ("related", "Training contains stove/pan tasks, including the longer stove+pan task, but not this exact short task."),
    21: ("exact", "Exact prompt exists in the finetune audit."),
    22: ("partial", "Training has open bottom drawer and close top drawer, but not close bottom drawer."),
    23: ("partial", "Training contains drawer open/close components, but not this exact compound sequence."),
    24: ("related", "Training has black-bowl top-drawer insertion and cabinet placement, but not bottom-drawer insertion."),
    25: ("exact", "Exact prompt exists in the finetune audit."),
    26: ("partial", "Training has drawer analogs but no wine-bottle object family."),
    27: ("absent", "No close wine-bottle or wine-rack task family appears in the finetune audit."),
    28: ("exact", "Exact prompt exists in the finetune audit."),
    29: ("exact", "Exact prompt exists in the finetune audit."),
}


def main() -> None:
    run_summaries = [load_run(config) for config in RUNS]
    validate_expected_counts(run_summaries)
    train_tasks = load_train_tasks(AUDIT_PATH)
    eval_prompts = collect_eval_prompts(run_summaries)
    closeness_rows = build_closeness_rows(eval_prompts, train_tasks)

    markdown = build_markdown(run_summaries, closeness_rows)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(markdown)
    write_pdf(markdown, REPORT_PDF)

    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PDF}")


def load_run(config: RunConfig) -> dict[str, Any]:
    if not config.trace_path.exists():
        raise FileNotFoundError(f"Missing trace file: {config.trace_path}")

    plans: dict[int, dict[str, Any]] = {}
    queries: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()

    with config.trace_path.open() as f:
        for line in f:
            record = json.loads(line)
            event_type = record["event_type"]
            event_counts[event_type] += 1
            if event_type == "scene_setup_plan":
                plans[int(record["task_id"])] = record
            elif event_type == "retrieval_query":
                queries.append(record)
            elif event_type == "rollout_summary":
                rollouts.append(record)

    task_stats = compute_task_stats(rollouts, queries)
    aggregate_stats = compute_aggregate_stats(rollouts, queries)
    successes = sum(int(record["success"]) for record in rollouts)
    normalized_successes, normalized_rollouts = compute_normalized_successes(config, task_stats)

    return {
        "config": config,
        "event_counts": event_counts,
        "plans": plans,
        "queries": queries,
        "rollouts": rollouts,
        "task_stats": task_stats,
        "aggregate_stats": aggregate_stats,
        "successes": successes,
        "rollout_count": len(rollouts),
        "normalized_successes": normalized_successes,
        "normalized_rollouts": normalized_rollouts,
    }


def compute_task_stats(rollouts: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rollouts_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    queries_by_rollout: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    for rollout in rollouts:
        rollouts_by_task[int(rollout["task_id"])].append(rollout)
    for query in queries:
        key = (int(query["task_id"]), int(query["scene_setup_number"]), int(query["trial_number"]))
        queries_by_rollout[key].append(query)

    stats = {}
    for task_id, task_rollouts in sorted(rollouts_by_task.items()):
        top1_distances = []
        query_counts = []
        over_max = 0
        query_total = 0
        for rollout in task_rollouts:
            key = (int(rollout["task_id"]), int(rollout["scene_setup_number"]), int(rollout["trial_number"]))
            rollout_queries = queries_by_rollout[key]
            query_counts.append(len(rollout_queries))
            for query in rollout_queries:
                retrieval_trace = query.get("retrieval_trace") or []
                if not retrieval_trace:
                    continue
                distance = float(retrieval_trace[0]["distance"])
                top1_distances.append(distance)
                query_total += 1
                over_max += int(distance > MAX_DISTANCE)
        stats[task_id] = {
            "successes": sum(int(record["success"]) for record in task_rollouts),
            "rollouts": len(task_rollouts),
            "prompt": task_rollouts[0]["task_description"],
            "mean_final_timestep": mean(float(record["final_timestep"]) for record in task_rollouts),
            "mean_queries": mean(query_counts),
            "mean_top1_distance": mean(top1_distances),
            "over_max": over_max,
            "query_total": query_total,
        }
    return stats


def compute_aggregate_stats(rollouts: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    queries_by_rollout: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        key = (int(query["task_id"]), int(query["scene_setup_number"]), int(query["trial_number"]))
        queries_by_rollout[key].append(query)

    aggregates = {}
    for label, keep_success in (("success", True), ("failure", False)):
        selected_rollouts = [record for record in rollouts if bool(record["success"]) is keep_success]
        top1_distances = []
        query_counts = []
        over_max = 0
        query_total = 0
        for rollout in selected_rollouts:
            key = (int(rollout["task_id"]), int(rollout["scene_setup_number"]), int(rollout["trial_number"]))
            rollout_queries = queries_by_rollout[key]
            query_counts.append(len(rollout_queries))
            for query in rollout_queries:
                retrieval_trace = query.get("retrieval_trace") or []
                if not retrieval_trace:
                    continue
                distance = float(retrieval_trace[0]["distance"])
                top1_distances.append(distance)
                query_total += 1
                over_max += int(distance > MAX_DISTANCE)
        aggregates[label] = {
            "rollouts": len(selected_rollouts),
            "mean_final_timestep": mean(float(record["final_timestep"]) for record in selected_rollouts),
            "mean_queries": mean(query_counts),
            "mean_top1_distance": mean(top1_distances),
            "over_max": over_max,
            "query_total": query_total,
        }
    return aggregates


def validate_expected_counts(run_summaries: list[dict[str, Any]]) -> None:
    for summary in run_summaries:
        config = summary["config"]
        if summary["successes"] != config.expected_successes or summary["rollout_count"] != config.expected_rollouts:
            raise ValueError(
                f"{config.short_label} parsed {summary['successes']}/{summary['rollout_count']}, "
                f"expected {config.expected_successes}/{config.expected_rollouts}."
            )


def compute_normalized_successes(config: RunConfig, task_stats: dict[int, dict[str, Any]]) -> tuple[int, int]:
    target_task_ids = range(20, 30)
    normalized_successes = 0
    normalized_rollouts = 0
    for task_id in target_task_ids:
        stats = task_stats.get(task_id)
        normalized_rollouts += 10
        if not stats:
            continue
        if stats["rollouts"] == 10:
            normalized_successes += int(stats["successes"])
        elif stats["rollouts"] == 1:
            normalized_successes += int(stats["successes"]) * 10
        else:
            raise ValueError(
                f"{config.short_label} task {task_id} has {stats['rollouts']} rollouts; "
                "normalization expects 1 or 10."
            )
    return normalized_successes, normalized_rollouts


def load_train_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing audit file: {path}")
    audit = json.loads(path.read_text())
    tasks = audit.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"Expected {path} to contain a list field named 'tasks'.")
    return tasks


def collect_eval_prompts(run_summaries: list[dict[str, Any]]) -> dict[int, str]:
    prompts = {}
    for summary in run_summaries:
        for task_id, stats in summary["task_stats"].items():
            prompts.setdefault(task_id, stats["prompt"])
    return dict(sorted(prompts.items()))


def build_closeness_rows(eval_prompts: dict[int, str], train_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for task_id, prompt in sorted(eval_prompts.items()):
        best_score, best_task = max(
            ((score_prompt_similarity(prompt, task["prompt"]), task) for task in train_tasks),
            key=lambda item: item[0],
        )
        category, rationale = TASK_CLOSENESS_OVERRIDES[task_id]
        rows.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "category": category,
                "nearest_train_official_id": best_task.get("official_converter_task_id"),
                "nearest_train_prompt": best_task["prompt"],
                "nearest_score": best_score,
                "rationale": rationale,
            }
        )
    return rows


def score_prompt_similarity(a: str, b: str) -> float:
    return 0.55 * token_jaccard(a, b) + 0.45 * SequenceMatcher(None, a.lower(), b.lower()).ratio()


def token_jaccard(a: str, b: str) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a or tokens_b else 0.0


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def build_markdown(run_summaries: list[dict[str, Any]], closeness_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RICL LIBERO Findings",
        "",
        "Generated: 2026-07-03",
        "",
        "## Executive Summary",
        "",
        "- Four local LIBERO eval experiments were compared: 20-memory, 40-memory, 49-memory/1-scene, and the expanded mismatched-context diagnostic.",
        "- Increasing retrieval memory from 20 to 40 to 49 episodes/task did not improve aggregate success.",
        "- Retrieval quality still correlates with success in the matched-context runs: successful rollouts have lower top-1 retrieval distances than failed rollouts.",
        "- The expanded mismatch run scored 36/60 on the six previously successful-at-least-once tasks, or 36/100 after assuming the historically failed tasks remain 0/10.",
        "- The mismatch result weakens the causal RICL claim for several known-good tasks: some skills still work with deliberately wrong context.",
        "- Persistent failures align with weak or absent finetune task-family overlap, especially bottom-drawer and wine-bottle/wine-rack tasks.",
        "",
        "## Raw Run Comparison",
        "",
        markdown_table(
            ["Run", "Retrieval demos/task", "Eval tasks", "Eval scenes/task", "Success", "Success rate", "Retrieval queries"],
            [
                [
                    summary["config"].short_label,
                    str(summary["config"].retrieval_demos_per_task),
                    format_task_set(summary["task_stats"]),
                    str(summary["config"].eval_scenes_per_task),
                    f"{summary['successes']}/{summary['rollout_count']}",
                    pct(summary["successes"], summary["rollout_count"]),
                    str(summary["event_counts"]["retrieval_query"]),
                ]
                for summary in run_summaries
            ],
        ),
        "",
        "## Normalized 100-Rollout Comparison",
        "",
        "This table uses 10 eval rollouts/task across tasks 20-29. For the 49-memory run, the single observed rollout per task is projected to 10. For the mismatch run, tasks 22, 23, 26, and 27 are assumed to remain 0/10 because they always failed earlier.",
        "",
        markdown_table(
            ["Run", "Normalized success", "Normalized rate", "Normalization"],
            [
                [
                    summary["config"].short_label,
                    f"{summary['normalized_successes']}/{summary['normalized_rollouts']}",
                    pct(summary["normalized_successes"], summary["normalized_rollouts"]),
                    summary["config"].normalized_note,
                ]
                for summary in run_summaries
            ],
        ),
        "",
        "## Per-Task Success",
        "",
        markdown_table(
            ["Task", "Prompt", *[summary["config"].short_label for summary in run_summaries]],
            [
                [
                    str(task_id),
                    get_prompt_for_task(run_summaries, task_id),
                    *[
                        format_success(summary["task_stats"].get(task_id, {}))
                        for summary in run_summaries
                    ],
                ]
                for task_id in range(20, 30)
            ],
        ),
        "",
        "Normalized mismatch per-task assumption: tasks 22, 23, 26, and 27 are counted as 0/10 in the 100-rollout view because they were not included in the expanded mismatch run and had never succeeded in prior completed matched-context runs.",
        "",
        "## Retrieval Distance Findings",
        "",
        f"Distance threshold used for above-max counts: `{MAX_DISTANCE:.4f}`.",
        "",
        markdown_table(
            ["Run", "Outcome", "Rollouts", "Mean top-1 dist", "Mean queries", "Above max"],
            [
                [
                    summary["config"].short_label,
                    outcome,
                    str(stats["rollouts"]),
                    format_float(stats["mean_top1_distance"]),
                    format_float(stats["mean_queries"]),
                    format_count_pct(stats["over_max"], stats["query_total"]),
                ]
                for summary in run_summaries
                for outcome, stats in summary["aggregate_stats"].items()
            ],
        ),
        "",
        "Interpretation: failures remain farther from retrieval neighbors than successes in the matched-context runs. However, the mismatch run has high top-1 distances for both successes and failures, showing that distance alone does not prove RICL is causally helping when the finetuned policy already knows the skill.",
        "",
        "## Task Closeness To Finetune Data",
        "",
        markdown_table(
            ["Task", "Eval prompt", "Closeness", "Nearest train task", "Score", "Finding"],
            [
                [
                    str(row["task_id"]),
                    row["prompt"],
                    row["category"],
                    f"{row['nearest_train_official_id']}: {row['nearest_train_prompt']}",
                    format_float(row["nearest_score"]),
                    row["rationale"],
                ]
                for row in closeness_rows
            ],
        ),
        "",
        "## Comparison To Official RICL Paper",
        "",
        f"Source: {RICL_PAPER_URL}",
        "",
        "- The official RICL paper evaluates with same-task retrieval data: 20 demonstrations are collected for each evaluation task and used as the retrieval buffer.",
        "- Their eval tasks are described as unseen or novel relative to priming/base data, but still generally sit inside the base VLA's broad manipulation envelope.",
        "- The paper explicitly notes that RICL does not typically enable tasks too far beyond what the base VLA can perform.",
        "- Our setup is structurally similar in that eval uses same-task retrieval buffers, but our results split strongly by finetune task-family closeness.",
        "",
        "## Conclusions",
        "",
        "- Our current evidence supports: retrieval quality is associated with success in matched-context evaluations.",
        "- Our current evidence does not prove: RICL materially improves performance beyond the finetuned policy.",
        "- Increasing retrieval memory from 20 to 40 demos/task improved retrieval-distance statistics but reduced aggregate success from 52% to 49%.",
        "- The 49-demo/1-scene run is useful as a sanity check but is not directly comparable to 10-scene runs.",
        "- The expanded mismatch run produced 36/60 raw success and 36/100 normalized success, with task 29 still at 10/10 despite wrong-but-related context.",
        "- Task 21 collapsed to 0/10 under mismatched context, suggesting it may depend more on correct contextual retrieval or be especially unstable.",
        "- Hard failures are concentrated where finetune overlap is partial or absent.",
        "",
        "## Recommended Next Experiments",
        "",
        "1. Run no-retrieval or dummy-retrieval controls on the same eval scenes.",
        "2. Compare correct vs mismatched retrieval using identical held-out scene indices.",
        "3. Repeat the expanded mismatch run with the same scene indices as the 20-memory matched run.",
        "4. Add a nearest-action / retrieve-and-play baseline to separate interpolation from learned policy behavior.",
        "",
    ]
    return "\n".join(lines)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    escaped_headers = [escape_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(escaped_headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(escape_cell(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def format_task_set(task_stats: dict[int, dict[str, Any]]) -> str:
    task_ids = sorted(task_stats)
    if not task_ids:
        return "-"
    contiguous = task_ids == list(range(task_ids[0], task_ids[-1] + 1))
    if contiguous and len(task_ids) > 2:
        return f"{task_ids[0]}-{task_ids[-1]}"
    return ",".join(str(task_id) for task_id in task_ids)


def get_prompt_for_task(run_summaries: list[dict[str, Any]], task_id: int) -> str:
    for summary in run_summaries:
        stats = summary["task_stats"].get(task_id)
        if stats:
            return stats["prompt"]
    return ""


def escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def format_success(stats: dict[str, Any]) -> str:
    if not stats:
        return "-"
    return f"{stats['successes']}/{stats['rollouts']}"


def format_count_pct(count: int, total: int) -> str:
    if total == 0:
        return "-"
    return f"{count}/{total} ({pct(count, total)})"


def pct(count: int | float, total: int | float) -> str:
    if total == 0:
        return "-"
    return f"{100.0 * count / total:.1f}%"


def format_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.1f}"


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def write_pdf(markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pages = paginate_markdown(markdown)
    with PdfPages(output_path) as pdf:
        for page_number, page_lines in enumerate(pages, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            y = 0.965
            for line in page_lines:
                font_size = 13 if line.startswith("# ") else 10
                font_weight = "bold" if line.startswith("#") else "normal"
                ax.text(
                    0.055,
                    y,
                    strip_markdown_heading(line),
                    ha="left",
                    va="top",
                    fontsize=font_size,
                    fontweight=font_weight,
                    family="DejaVu Sans Mono" if line.startswith("|") else "DejaVu Sans",
                )
                y -= 0.030 if font_size == 10 else 0.040
            ax.text(0.5, 0.025, f"Page {page_number}", ha="center", va="bottom", fontsize=8, color="#666666")
            pdf.savefig(fig)
            plt.close(fig)


def paginate_markdown(markdown: str, *, max_lines: int = 34, width: int = 108) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            pages.append(current)
            current = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        wrapped_lines = wrap_line(line, width)
        if line.startswith("## ") and len(current) > max_lines - 8:
            flush()
        for wrapped in wrapped_lines:
            if len(current) >= max_lines:
                flush()
            current.append(wrapped)
    flush()
    return pages


def wrap_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]
    if line.startswith("|"):
        return textwrap.wrap(line, width=width, subsequent_indent="  ", break_long_words=False) or [line]
    if line.startswith("#"):
        return [line]
    return textwrap.wrap(line, width=width, break_long_words=False) or [line]


def strip_markdown_heading(line: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", line)


if __name__ == "__main__":
    main()
