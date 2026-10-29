#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.policies import libero_retrieval_benchmark as rb

SAME_ROOT_DEFAULT = Path("data/libero/retrieval_benchmark/libero90_solvable25_offline_v1")
MIXED_ROOT_DEFAULT = Path("data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1")
OUTPUT_DEFAULT = Path("data/libero/reports/libero90_retrieval_benchmarks_combined_analysis.md")
PLOTS_DEFAULT = Path("data/libero/reports/libero90_retrieval_benchmarks_combined_analysis_plots")

METHOD_PAIRS = {
    "state": "state_global",
    "dino_top": "dino_top_global",
    "dino_top_wrist": "dino_top_wrist_global",
    "dino_top_state": "dino_top_state_global",
    "dino_top_wrist_state": "dino_top_wrist_state_global",
    "pi0_fast_first_fusion": "pi0_fast_first_fusion_global",
    "pi0_fast_first_fusion_no_prompt": "pi0_fast_first_fusion_no_prompt_global",
}
SAME_NAMES = {
    "random": "Random",
    "state": "State",
    "dino_top": "DINO top",
    "dino_top_wrist": "DINO top+wrist",
    "dino_top_state": "DINO top+state",
    "dino_top_wrist_state": "DINO top+wrist+state",
    "pi0_fast_first_fusion": "Pi0 full",
    "pi0_fast_first_fusion_no_prompt": "Pi0 no prompt",
    "oracle_action": "Action oracle",
}
MIXED_NAMES = {
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
PRACTICAL_MIXED = (
    "state_global",
    "dino_top_global",
    "dino_top_wrist_global",
    "dino_top_state_global",
    "dino_top_wrist_state_global",
    "pi0_fast_first_fusion_global",
    "pi0_fast_first_fusion_no_prompt_global",
    "pi0_fast_first_fusion_shuffled_prompt_global",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a combined illustrated analysis of both LIBERO retrieval benchmarks."
    )
    parser.add_argument("--same-root", type=Path, default=SAME_ROOT_DEFAULT)
    parser.add_argument("--mixed-root", type=Path, default=MIXED_ROOT_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    same_summary = keyed(read_csv(args.same_root / "full" / "summary.csv"))
    same_per_task = read_csv(args.same_root / "full" / "per_task.csv")
    same_pairwise = keyed(read_csv(args.same_root / "full" / "pairwise.csv"))
    mixed_summary = keyed(read_csv(args.mixed_root / "full" / "summary.csv"))
    mixed_per_task = read_csv(args.mixed_root / "full" / "per_task.csv")
    mixed_confusion = read_csv(args.mixed_root / "full" / "confusion.csv")
    same_manifest = json.loads((args.same_root / "manifest.json").read_text())
    mixed_manifest = json.loads((args.mixed_root / "manifest.json").read_text())
    prompts = load_task_prompts(args.mixed_root / "full" / "queries.jsonl")

    validate_sources(
        args,
        same_summary,
        mixed_summary,
        same_manifest,
        mixed_manifest,
    )
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plots = {
        "same_ranking": args.plots_dir / "01_same_task_ranking.png",
        "same_deltas": args.plots_dir / "02_same_task_paired_deltas.png",
        "mixed_ranking": args.plots_dir / "03_mixed_task_ranking.png",
        "selection_phase": args.plots_dir / "04_task_selection_and_phase.png",
        "prompt_ablation": args.plots_dir / "05_prompt_ablation.png",
        "cross_rmse": args.plots_dir / "06_cross_benchmark_conditional_rmse.png",
        "per_task": args.plots_dir / "07_mixed_per_task_heatmap.png",
        "pi0_confusion": args.plots_dir / "08_pi0_prompt_confusion.png",
    }
    plot_same_ranking(same_summary, plots["same_ranking"])
    plot_same_deltas(same_pairwise, plots["same_deltas"])
    plot_mixed_ranking(mixed_summary, plots["mixed_ranking"])
    plot_selection_phase(mixed_summary, plots["selection_phase"])
    plot_prompt_ablation(mixed_summary, plots["prompt_ablation"])
    plot_cross_rmse(same_summary, mixed_summary, plots["cross_rmse"])
    plot_per_task(mixed_per_task, plots["per_task"])
    plot_pi0_confusion(mixed_confusion, plots["pi0_confusion"])

    derived = compute_derived(
        same_summary,
        same_per_task,
        mixed_summary,
        mixed_per_task,
        mixed_confusion,
        prompts,
    )
    report = render_report(
        args,
        plots,
        same_summary,
        same_pairwise,
        mixed_summary,
        same_manifest,
        mixed_manifest,
        derived,
        prompts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(derived, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    for path in plots.values():
        print(f"Wrote {path}")


def validate_sources(
    args: argparse.Namespace,
    same_summary: dict[str, dict[str, str]],
    mixed_summary: dict[str, dict[str, str]],
    same_manifest: dict[str, Any],
    mixed_manifest: dict[str, Any],
) -> None:
    if same_manifest["status"] != "complete" or mixed_manifest["status"] != "complete":
        raise ValueError("Both source benchmarks must be complete.")
    if same_manifest["queries"] != 7_500 or mixed_manifest["queries"] != 7_500:
        raise ValueError("Both source benchmarks must contain exactly 7,500 queries.")
    if same_manifest["candidate_frames"] != 69_990 or mixed_manifest["candidate_frames"] != 69_990:
        raise ValueError("Both source benchmarks must use exactly 69,990 candidate frames.")
    if set(same_manifest["task_ids"]) != set(mixed_manifest["task_ids"]):
        raise ValueError("Source benchmarks use different task sets.")
    if len(same_summary) != 9 or len(mixed_summary) != 12:
        raise ValueError("Unexpected source method count.")
    if sum(1 for _ in (args.same_root / "full" / "queries.jsonl").open()) != 67_500:
        raise ValueError("Same-task query result count is not 67,500.")
    if sum(1 for _ in (args.mixed_root / "full" / "queries.jsonl").open()) != 90_000:
        raise ValueError("Mixed-task query result count is not 90,000.")


def compute_derived(
    same_summary: dict[str, dict[str, str]],
    same_per_task: list[dict[str, str]],
    mixed_summary: dict[str, dict[str, str]],
    mixed_per_task: list[dict[str, str]],
    mixed_confusion: list[dict[str, str]],
    prompts: dict[int, str],
) -> dict[str, Any]:
    random_rmse = num(same_summary["random"]["best_of_4_action_rmse"])
    oracle_rmse = num(same_summary["oracle_action"]["best_of_4_action_rmse"])
    same_gap_closed = {}
    for method, row in same_summary.items():
        score = num(row["best_of_4_action_rmse"])
        same_gap_closed[method] = (random_rmse - score) / (random_rmse - oracle_rmse)

    prompt_full = mixed_summary["pi0_fast_first_fusion_global"]
    prompt_blank = mixed_summary["pi0_fast_first_fusion_no_prompt_global"]
    prompt_wrong = mixed_summary["pi0_fast_first_fusion_shuffled_prompt_global"]
    prompt_deltas = {
        "full_minus_blank_ndcg": num(prompt_full["task_aware_ndcg_at4"]) - num(prompt_blank["task_aware_ndcg_at4"]),
        "full_minus_blank_top1": num(prompt_full["top1_task_accuracy"]) - num(prompt_blank["top1_task_accuracy"]),
        "full_minus_blank_recall4": num(prompt_full["task_recall_at4"]) - num(prompt_blank["task_recall_at4"]),
        "full_minus_wrong_ndcg": num(prompt_full["task_aware_ndcg_at4"]) - num(prompt_wrong["task_aware_ndcg_at4"]),
        "full_minus_wrong_top1": num(prompt_full["top1_task_accuracy"]) - num(prompt_wrong["top1_task_accuracy"]),
        "full_minus_wrong_recall4": num(prompt_full["task_recall_at4"]) - num(prompt_wrong["task_recall_at4"]),
    }
    per_task_values = {
        method: {
            int(row["task_id"]): num(row["task_aware_ndcg_at4"]) for row in mixed_per_task if row["method"] == method
        }
        for method in (
            "dino_top_global",
            "dino_top_state_global",
            "pi0_fast_first_fusion_global",
            "pi0_fast_first_fusion_no_prompt_global",
            "pi0_fast_first_fusion_shuffled_prompt_global",
        )
    }
    prompt_comparisons = {}
    for name, baseline, comparison in (
        (
            "full_vs_blank",
            "pi0_fast_first_fusion_no_prompt_global",
            "pi0_fast_first_fusion_global",
        ),
        (
            "full_vs_wrong",
            "pi0_fast_first_fusion_shuffled_prompt_global",
            "pi0_fast_first_fusion_global",
        ),
        ("dino_state_vs_dino", "dino_top_global", "dino_top_state_global"),
    ):
        delta, low, high = rb.paired_task_bootstrap(
            per_task_values[baseline],
            per_task_values[comparison],
            samples=10_000,
            seed=0,
        )
        prompt_comparisons[name] = {
            "mean_delta_ndcg": delta,
            "ci95_low": low,
            "ci95_high": high,
            "wins": sum(
                per_task_values[comparison][task] > per_task_values[baseline][task]
                for task in per_task_values[baseline]
            ),
            "losses": sum(
                per_task_values[comparison][task] < per_task_values[baseline][task]
                for task in per_task_values[baseline]
            ),
        }

    same_winners = winner_counts(
        same_per_task,
        metric="best_of_4_action_rmse",
        methods=set(METHOD_PAIRS),
        maximize=False,
    )
    mixed_winners = winner_counts(
        mixed_per_task,
        metric="task_aware_ndcg_at4",
        methods=set(PRACTICAL_MIXED),
        maximize=True,
    )
    hardest = {}
    for method in (
        "dino_top_state_global",
        "dino_top_global",
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
    ):
        rows = sorted(
            (row for row in mixed_per_task if row["method"] == method),
            key=lambda row: num(row["task_aware_ndcg_at4"]),
        )
        hardest[method] = [
            {
                "task_id": int(row["task_id"]),
                "prompt": prompts[int(row["task_id"])],
                "ndcg": num(row["task_aware_ndcg_at4"]),
                "top1_task_accuracy": num(row["top1_task_accuracy"]),
                "task_recall_at4": num(row["task_recall_at4"]),
            }
            for row in rows[:5]
        ]
    confusions = {}
    for method in (
        "dino_top_state_global",
        "dino_top_global",
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
    ):
        wrong = sorted(
            (
                row
                for row in mixed_confusion
                if row["method"] == method and row["query_task_id"] != row["retrieved_top1_task_id"]
            ),
            key=lambda row: int(row["count"]),
            reverse=True,
        )
        confusions[method] = [
            {
                "query_task": int(row["query_task_id"]),
                "retrieved_task": int(row["retrieved_top1_task_id"]),
                "count": int(row["count"]),
            }
            for row in wrong[:5]
        ]
    conditional_change = {}
    for same_method, mixed_method in METHOD_PAIRS.items():
        same_rmse = num(same_summary[same_method]["best_of_4_action_rmse"])
        mixed_rmse = num(mixed_summary[mixed_method]["conditional_best_action_rmse"])
        conditional_change[same_method] = {
            "same_task_rmse": same_rmse,
            "mixed_conditional_rmse": mixed_rmse,
            "relative_change": (mixed_rmse - same_rmse) / same_rmse,
            "mixed_task_recall_at4": num(mixed_summary[mixed_method]["task_recall_at4"]),
        }
    return {
        "task_prompts": {str(task): prompt for task, prompt in sorted(prompts.items())},
        "same_task_oracle_gap_closed": same_gap_closed,
        "prompt_deltas": prompt_deltas,
        "paired_comparisons": prompt_comparisons,
        "same_task_winner_counts": same_winners,
        "mixed_task_winner_counts": mixed_winners,
        "conditional_action_quality_change": conditional_change,
        "hardest_tasks": hardest,
        "top_confusions": confusions,
    }


def render_report(
    args: argparse.Namespace,
    plots: dict[str, Path],
    same_summary: dict[str, dict[str, str]],
    same_pairwise: dict[str, dict[str, str]],
    mixed_summary: dict[str, dict[str, str]],
    same_manifest: dict[str, Any],
    mixed_manifest: dict[str, Any],
    derived: dict[str, Any],
    prompts: dict[int, str],
) -> str:
    image = {key: path.relative_to(args.output.parent).as_posix() for key, path in plots.items()}
    same_rows = sorted(
        same_summary.values(),
        key=lambda row: num(row["best_of_4_action_rmse"]),
    )
    mixed_rows = sorted(
        mixed_summary.values(),
        key=lambda row: num(row["task_aware_ndcg_at4"]),
        reverse=True,
    )
    conditional = derived["conditional_action_quality_change"]
    paired = derived["paired_comparisons"]
    hardest_dino = derived["hardest_tasks"]["dino_top_state_global"]
    hardest_pi0 = derived["hardest_tasks"]["pi0_fast_first_fusion_global"]
    dino_confusions = derived["top_confusions"]["dino_top_state_global"]
    pi0_confusions = derived["top_confusions"]["pi0_fast_first_fusion_global"]

    lines = [
        "# LIBERO Retrieval Benchmarks: Combined Analysis",
        "",
        "## Executive Summary",
        "",
        "Two offline benchmarks were run over the same 25 LIBERO-90 tasks, 500 retrieval episodes, "
        "69,990 candidate frames, and 7,500 held-out query frames.",
        "",
        "1. The **task-specific benchmark** assumes the correct task is already known and measures how well "
        "each method retrieves the correct task phase or future action.",
        "2. The **mixed-task benchmark** searches one global 25-task memory and therefore measures both task "
        "identification and within-task phase retrieval.",
        "",
        "The central result is that **DINO top + robot state is the strongest general-purpose retriever in both "
        "settings**. In the task-specific benchmark it has the lowest practical action RMSE (`0.2083`), and in "
        "the mixed benchmark it has the highest ungated task-aware nDCG@4 (`0.7897`).",
        "",
        "The prompt is genuinely useful to Pi0 when tasks are mixed. Pi0 full reaches `0.7280` nDCG, compared "
        "with `0.6821` without a prompt and `0.5664` with a deliberately wrong prompt. In contrast, prompt and "
        "no-prompt Pi0 were almost identical when retrieval was already restricted to one task.",
        "",
        "A hard exact-prompt filter followed by DINO reaches `0.8643` nDCG and 100% task recall. This is a strong "
        "control, but it assumes candidate and query instructions use exactly matching text.",
        "",
        "## What Was Held Constant",
        "",
        f"- Tasks: `{','.join(str(task) for task in same_manifest['task_ids'])}`",
        "- Candidate memory: 20 episodes per task, 500 episodes total",
        "- Candidate frames: 69,990",
        "- Query split: 30 held-out episodes per task, with no candidate/query episode overlap",
        "- Queries: 10 complete-horizon frames per held-out episode, 7,500 total",
        "- Retrieval count: top four frames",
        "- Action target: normalized next 10 expert actions",
        "- Pi0 checkpoint: normal non-RICL `pi0_fast_libero90_normal` checkpoint",
        "- Pi0 representation: first-block attention output at the last valid prefix token",
        "- No policy inference, action generation, simulator execution, or RICL interpolation was run",
        "",
        "## Experimental Protocol",
        "",
        "### Candidate Memory",
        "",
        "For every benchmark task, the candidate memory contains the same 20 seeded retrieval episodes used by the "
        "RICL20 evaluation. Every frame in those episodes is searchable, giving 500 candidate episodes and 69,990 "
        "candidate frames across the 25 tasks. A candidate frame contains a top-view image, wrist image, robot state, "
        "task prompt, episode/frame identity, normalized trajectory progress, and its next 10 expert actions.",
        "",
        "All candidate frames are retained to match actual RICL memory lookup. When fewer than 10 future actions remain "
        "near the end of a candidate episode, the final available action is repeated to complete the scoring chunk. "
        "Held-out query frames, by contrast, are selected only where 10 real future actions remain.",
        "",
        "The future actions are retained only as hidden evaluation labels. They are not supplied to DINO, state, Pi0, "
        "prompt-gated, or task-gated retrieval.",
        "",
        "### Held-Out Queries",
        "",
        "The 30 episodes per task absent from the retrieval memory form the held-out query set. Ten frames are selected "
        "at evenly spaced trajectory positions from each held-out episode. Query frames must have a complete 10-step "
        "future action chunk. This produces `30 x 10 = 300` queries per task and `7,500` queries in total.",
        "",
        "Candidate and query episode IDs are explicitly checked for overlap. The benchmark therefore measures retrieval "
        "into separate demonstrations, not recovery of another frame from the query episode.",
        "",
        "### Two Search Conditions",
        "",
        "1. **Task-specific:** a query searches only candidate frames from its own task. This isolates trajectory-phase "
        "and action matching after the task has already been identified.",
        "2. **Mixed-task:** a query searches one global index containing all 69,990 frames from all 25 tasks. Unless a "
        "method is explicitly called gated, it receives no numeric task ID. This tests task identification and "
        "within-task phase matching together.",
        "",
        "All methods return exactly four frames. Distance ties are resolved deterministically by `(distance, "
        "episode_index, frame_index)` in the task-specific benchmark and `(distance, task_index, episode_index, "
        "frame_index)` in the mixed benchmark.",
        "",
        "## Exact Retrieval Methods",
        "",
        "The table summarizes what each method may inspect. A check means the information affects retrieval; expert "
        "actions are available only to the reporting oracle.",
        "",
        "| Method | Top image | Wrist image | State | Prompt | Task ID | Future actions |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        "| Random |  |  |  |  |  |  |",
        "| State |  |  | ✓ |  |  |  |",
        "| DINO top | ✓ |  |  |  |  |  |",
        "| DINO top+wrist | ✓ | ✓ |  |  |  |  |",
        "| DINO top+state | ✓ |  | ✓ |  |  |  |",
        "| DINO top+wrist+state | ✓ | ✓ | ✓ |  |  |  |",
        "| Pi0 full | ✓ | ✓ | ✓ | ✓ |  |  |",
        "| Pi0 no prompt | ✓ | ✓ | ✓ | blank |  |  |",
        "| Pi0 shuffled prompt | ✓ | ✓ | ✓ | wrong |  |  |",
        "| Prompt-gated DINO | ✓ |  |  | exact filter |  |  |",
        "| Task-gated DINO | ✓ |  |  |  | ✓ |  |",
        "| Action oracle |  |  |  |  | ✓ | ✓ |",
        "",
        "### Random",
        "",
        "Random retrieval samples four distinct candidate frames without replacement. Each query is evaluated with 20 "
        "deterministic seeded repeats and the metrics are averaged. It supplies a chance-level lower bound. In the "
        "task-specific benchmark it samples within the query task; in the mixed benchmark it samples from the complete "
        "25-task memory.",
        "",
        "### State",
        "",
        "The 8-dimensional LIBERO robot state is standardized with the normal checkpoint statistics:",
        "",
        "```text",
        "normalized_state = (state - checkpoint_mean) / (checkpoint_std + 1e-6)",
        "distance = L2(query_normalized_state, candidate_normalized_state)",
        "```",
        "",
        "This method has no visual or language information. It retrieves configurations with similar end-effector, "
        "gripper, and proprioceptive state, which often acts as a proxy for trajectory phase.",
        "",
        "### DINO Top",
        "",
        "The top-view RGB image is resized with padding to `224x224`, scaled to `[0,1]`, and normalized with ImageNet "
        "mean and standard deviation. It is passed through frozen `dinov2_vitb14`. The model's `16x16 = 256` normalized "
        "patch tokens are spatially pooled in non-overlapping `2x2` groups, producing an `8x8` grid of 64 vectors. Each "
        "vector has 768 dimensions, and the grid is flattened into one `64x768 = 49,152` dimensional feature.",
        "",
        "Candidate frames are ranked by Euclidean distance between these flattened spatial features. Retaining the "
        "8x8 layout preserves coarse object location better than using only the DINO CLS token or a global average.",
        "",
        "### DINO Top+Wrist",
        "",
        "The same frozen DINO pipeline independently embeds the top and wrist images. The combined distance is:",
        "",
        "```text",
        "distance = sqrt(top_L2_squared + wrist_L2_squared)",
        "```",
        "",
        "This is equivalent to concatenating the two 49,152-dimensional features and taking Euclidean distance. It adds "
        "close-range gripper and held-object evidence while preserving the top-view scene.",
        "",
        "### DINO Top+State",
        "",
        "DINO top distance and normalized-state distance have different numerical scales, so each query's candidate "
        "distance vector is divided by its own median before combination:",
        "",
        "```text",
        "distance = dino_top / median(dino_top) + beta * state / median(state)",
        "```",
        "",
        "The tested beta grid was `0, 0.1, 0.25, 0.5, 1, 2`. Calibration used only tasks `0-19`, never benchmark tasks "
        "`20-69`. The selected value was `beta=1.0`, minimizing macro-average best-of-four action RMSE on calibration "
        "tasks.",
        "",
        "### DINO Top+Wrist+State",
        "",
        "This method first forms the joint top+wrist DINO distance, then combines it with state using the same "
        "per-query median scaling:",
        "",
        "```text",
        "distance = dino_top_wrist / median(dino_top_wrist) + beta * state / median(state)",
        "```",
        "",
        "The independently calibrated value was `beta=0.5`.",
        "",
        "### Pi0 Full",
        "",
        "This method uses the normal, non-RICL checkpoint `pi0_fast_libero90_normal`. Each frame is passed through the "
        "normal Pi0-FAST preprocessing path with exactly these observation fields:",
        "",
        "```text",
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
        "```",
        "",
        "The checkpoint normalization is applied first. Pi0's SigLIP `So400m/14` image encoder converts both images to "
        "image tokens. The normal FAST tokenizer encodes the prompt together with the normalized state. No action tokens, "
        "retrieved demonstrations, RICL input structure, distances, or interpolation values are supplied.",
        "",
        "The image tokens and prompt/state tokens form one prefix. The retrieval feature is the 2,048-dimensional output "
        "projection of transformer block 1 self-attention at the last valid prefix token. This is the earliest single "
        "token that has attended across the complete image-and-text/state prefix. It is captured before block 1's residual "
        "addition, feed-forward network, and all later transformer blocks, then L2-normalized. Retrieval uses Euclidean "
        "distance between these normalized features.",
        "",
        "### Pi0 No Prompt",
        "",
        "This is identical to Pi0 full, including the same checkpoint, images, state, normalization, FAST tokenizer, "
        "feature location, and ranking. The only change is replacing the instruction with an empty string for both "
        "candidate and query frames. State remains present in the FAST token sequence. The comparison therefore isolates "
        "the information contributed by task language.",
        "",
        "### Pi0 Shuffled Prompt",
        "",
        "This mixed-task-only diagnostic leaves candidate embeddings unchanged with their correct prompts. Each query is "
        "re-embedded with a deterministic prompt borrowed from a different task, explicitly chosen to have different "
        "text. Its images and state remain correct. A performance drop shows that the Pi0 representation uses prompt "
        "semantics rather than merely tolerating the prompt tokens.",
        "",
        "### Prompt-Gated DINO",
        "",
        "This mixed-task control first keeps only candidate frames whose prompt string exactly equals the query prompt, "
        "then ranks that eligible set with DINO top distance. It does not receive the numeric task ID. If two tasks share "
        "identical text, both remain eligible. In this benchmark, DINO's visual ranking disambiguated those remaining "
        "cases, yielding 100% correct-task recall.",
        "",
        "This is exact string matching, not a language embedding or semantic paraphrase matcher.",
        "",
        "### Task-Gated DINO",
        "",
        "This reporting control uses the known numeric task ID to restrict candidates to the correct task and then applies "
        "DINO top retrieval. It reproduces the original task-specific DINO selections exactly. It represents performance "
        "when task identification is perfect, but it is not an ungated mixed-memory retriever.",
        "",
        "### Action Oracle",
        "",
        "The oracle ranks candidates by RMSE between the candidate's and query's next 10 expert actions after checkpoint "
        "quantile normalization:",
        "",
        "```text",
        "normalized_action = 2 * (action - q01) / (q99 - q01 + 1e-6) - 1",
        "distance = RMSE(candidate_action[10,7], query_action[10,7])",
        "```",
        "",
        "In mixed retrieval, the oracle is also restricted to the correct task. It is a reporting-only upper bound and "
        "cannot be used at deployment because future query actions are unknown.",
        "",
        "## Scoring and Aggregation",
        "",
        "### Task-Specific Metrics",
        "",
        "- **Top-1 action RMSE:** action error of the nearest retrieved frame.",
        "- **Best-of-4 action RMSE:** lowest action error among the four retrieved frames; this is the primary metric.",
        "- **Mean top-4 action RMSE:** average action error across all four frames.",
        "- **Oracle regret:** best-of-four RMSE minus the lowest possible same-task action RMSE.",
        "- **Progress error:** mean absolute difference in normalized trajectory position.",
        "- **State error:** mean normalized-state L2 error of the four retrieved frames.",
        "- **Unique episodes:** number of distinct demonstrations represented among the top four.",
        "",
        "### Mixed-Task Metrics",
        "",
        "- **Top-1 task accuracy:** whether the nearest frame belongs to the query task.",
        "- **Task recall@4:** whether at least one of the four frames belongs to the query task.",
        "- **Correct-task fraction@4:** fraction of the four frames belonging to the query task.",
        "- **Useful retrieval@4:** whether at least one correct-task frame falls in the best 10% of same-task candidates "
        "by future-action RMSE.",
        "- **Conditional action RMSE:** best action RMSE among correct-task hits, reported only when at least one exists.",
        "- **Task-aware nDCG@4:** the primary mixed metric. Wrong-task frames receive zero relevance. A correct-task "
        "frame's relevance is its percentile rank among all same-task candidates by future-action RMSE. Rank positions "
        "are discounted by `1/log2(rank+1)` and normalized by the ideal action-oracle ordering.",
        "",
        "Each metric is first averaged over the 300 queries within a task. The reported overall score then averages the "
        "25 task means equally, preventing longer candidate episodes or particular tasks from dominating. Pairwise "
        "confidence intervals use a seeded 10,000-sample bootstrap over tasks.",
        "",
        "## Validation and Execution Guards",
        "",
        "Before the full task-specific benchmark, a CUDA smoke test ran every method on tasks `20` and `30`, which cover "
        "both retrieval-artifact families. It used all 20 candidate episodes, the lowest-numbered held-out episode, and "
        "three evenly spaced query frames per task. The mixed benchmark repeated this smoke structure against the global "
        "index.",
        "",
        "The implementation checked that:",
        "",
        "- CUDA was used for both DINO and Pi0 feature extraction.",
        "- Every method returned exactly four candidates.",
        "- Candidate and query episodes never overlapped.",
        "- Task-specific candidates always belonged to the query task.",
        "- Pi0 observation dictionaries contained no expert actions or RICL demonstrations.",
        "- Correct-prompt and blank-prompt token sequences differed.",
        "- Shuffled prompts differed from the corresponding correct prompts.",
        "- Feature arrays and distance vectors contained no NaNs or infinities.",
        "- Repeated ranking calls produced identical candidate orderings.",
        "- Hybrid calibration read only tasks `0-19`.",
        "- Task-gated mixed DINO matched all 7,500 original task-specific DINO selections.",
        "",
        "Both full manifests are marked `complete`. The final task-specific file contains 67,500 rows "
        "(`7,500 queries x 9 methods`), and the mixed-task file contains 90,000 rows "
        "(`7,500 queries x 12 methods`).",
        "",
        "## Benchmark 1: Task-Specific Retrieval",
        "",
        "Every query searched only the 20 retrieval episodes belonging to its known task. The main metric is "
        "**best-of-four action RMSE**, where lower is better.",
        "",
        f"![Task-specific ranking]({image['same_ranking']})",
        "",
        "### Overall Results",
        "",
        "| Rank | Method | Best-of-4 RMSE | Relative to DINO top | Random-to-oracle gap closed |",
        "|---:|---|---:|---:|---:|",
    ]
    dino_same = num(same_summary["dino_top"]["best_of_4_action_rmse"])
    for rank, row in enumerate(same_rows, start=1):
        method = row["method"]
        value = num(row["best_of_4_action_rmse"])
        relative = (dino_same - value) / dino_same
        gap = derived["same_task_oracle_gap_closed"][method]
        lines.append(f"| {rank} | {SAME_NAMES[method]} | {value:.4f} | {relative:+.1%} | {gap:.1%} |")
    lines.extend(
        [
            "",
            "DINO top+state improves action RMSE by **7.5%** over DINO top-only and closes **61.5%** of the "
            "random-to-oracle gap. DINO top+wrist and DINO top+wrist+state are close behind.",
            "",
            "State alone also beats DINO top-only, showing that robot configuration contains substantial task-phase "
            "information. However, combining state with vision is better than either alone.",
            "",
            "Pi0 full and Pi0 without prompt improve only slightly over DINO top-only. Their paired task-bootstrap "
            "confidence intervals cross zero, so this benchmark does not establish a reliable Pi0 advantage over "
            "top-view DINO.",
            "",
            f"![Paired deltas]({image['same_deltas']})",
            "",
            "### Consistency Across Tasks",
            "",
            "- DINO top+state beats DINO top on 22 of 25 tasks.",
            "- DINO top+wrist beats DINO top on 23 of 25 tasks.",
            "- DINO top+wrist+state beats DINO top on 22 of 25 tasks.",
            "- Pi0 full beats DINO top on 12 tasks and loses on 13.",
            "- DINO top+state is the best practical method on 13 tasks; DINO top+wrist+state wins 6.",
            "",
            "Exact frame overlap with the action oracle is low for every practical method. This is expected: multiple "
            "demonstration frames can encode nearly equivalent future motions. Action RMSE and oracle regret are more "
            "meaningful than exact frame identity.",
            "",
            "## Benchmark 2: Global Mixed-Task Retrieval",
            "",
            "Each query searched all 69,990 frames from all 25 tasks. Global methods did not receive the numeric task "
            "ID. The primary metric is **task-aware nDCG@4**, where wrong-task frames receive zero relevance and "
            "correct-task frames are graded by their percentile among same-task action matches.",
            "",
            f"![Mixed-task ranking]({image['mixed_ranking']})",
            "",
            "### Overall Results",
            "",
            "| Rank | Method | nDCG@4 | Top-1 task | Task recall@4 | Useful retrieval@4 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(mixed_rows, start=1):
        method = row["method"]
        lines.append(
            f"| {rank} | {MIXED_NAMES[method]} | {num(row['task_aware_ndcg_at4']):.4f} | "
            f"{num(row['top1_task_accuracy']):.1%} | {num(row['task_recall_at4']):.1%} | "
            f"{num(row['useful_retrieval_at4']):.1%} |"
        )
    lines.extend(
        [
            "",
            "Among ungated methods, DINO top+state is best. It improves nDCG by `0.0436` over DINO top-only, "
            f"with a paired 95% CI of `[{paired['dino_state_vs_dino']['ci95_low']:.4f}, "
            f"{paired['dino_state_vs_dino']['ci95_high']:.4f}]`, and wins on all 25 tasks.",
            "",
            "The state term barely changes task recall, but materially improves nDCG and useful retrieval rate. This "
            "means state primarily helps choose the correct **phase** within candidate tasks rather than identify the task.",
            "",
            f"![Task selection and phase quality]({image['selection_phase']})",
            "",
            "## Prompt Analysis",
            "",
            f"![Prompt ablation]({image['prompt_ablation']})",
            "",
            f"- Pi0 full versus blank prompt: `+{derived['prompt_deltas']['full_minus_blank_ndcg']:.4f}` nDCG, "
            f"`+{100 * derived['prompt_deltas']['full_minus_blank_top1']:.1f}` percentage points in "
            f"top-1 task accuracy, and `+{100 * derived['prompt_deltas']['full_minus_blank_recall4']:.1f}` "
            "percentage points in recall@4.",
            f"- Pi0 full versus wrong prompt: `+{derived['prompt_deltas']['full_minus_wrong_ndcg']:.4f}` nDCG, "
            f"`+{100 * derived['prompt_deltas']['full_minus_wrong_top1']:.1f}` percentage points in "
            f"top-1 task accuracy, and `+{100 * derived['prompt_deltas']['full_minus_wrong_recall4']:.1f}` "
            "percentage points in recall@4.",
            f"- Full beats blank on {paired['full_vs_blank']['wins']} of 25 tasks; paired nDCG 95% CI "
            f"`[{paired['full_vs_blank']['ci95_low']:.4f}, {paired['full_vs_blank']['ci95_high']:.4f}]`.",
            f"- Full beats wrong-prompt retrieval on {paired['full_vs_wrong']['wins']} of 25 tasks; paired 95% CI "
            f"`[{paired['full_vs_wrong']['ci95_low']:.4f}, {paired['full_vs_wrong']['ci95_high']:.4f}]`.",
            "",
            "This is the cleanest evidence that the early Pi0 representation actually contains and uses language. "
            "The effect was hidden in the first benchmark because every candidate within a task had essentially the "
            "same instruction.",
            "",
            f"![Pi0 prompt confusion]({image['pi0_confusion']})",
            "",
            "Prompt-gated DINO first filters candidates by exact instruction text and then ranks them visually. It "
            "matches task-gated DINO in aggregate. Even where prompts are duplicated across tasks, DINO resolves the "
            "remaining scene differences in this dataset.",
            "",
            "## Separating Task Selection from Phase Retrieval",
            "",
            f"![Cross-benchmark conditional RMSE]({image['cross_rmse']})",
            "",
            "For queries where a mixed-memory method retrieves at least one correct-task frame, its conditional action "
            "RMSE remains close to its task-specific RMSE:",
            "",
            "| Method | Same-task RMSE | Mixed conditional RMSE | Relative change | Mixed task recall@4 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_PAIRS:
        row = conditional[method]
        lines.append(
            f"| {SAME_NAMES[method]} | {row['same_task_rmse']:.4f} | "
            f"{row['mixed_conditional_rmse']:.4f} | {row['relative_change']:+.1%} | "
            f"{row['mixed_task_recall_at4']:.1%} |"
        )
    lines.extend(
        [
            "",
            "The conditional increases are generally only about 2-6%. This suggests that most global-index degradation "
            "comes from selecting the wrong task or diluting the top four with wrong-task frames, rather than losing the "
            "ability to identify useful phases once the correct task is reached.",
            "",
            "Conditional RMSE must be interpreted alongside recall because it excludes queries with no correct-task hit.",
            "",
            "## Per-Task Patterns",
            "",
            f"![Per-task heatmap]({image['per_task']})",
            "",
            f"DINO top+state is the best ungated method on {derived['mixed_task_winner_counts'].get('dino_top_state_global', 0)} "
            "of 25 tasks. Pi0 full wins five tasks, especially several visually ambiguous cabinet and container tasks.",
            "",
            "### Hardest Tasks for DINO Top+State",
            "",
            "| Task | Instruction | nDCG | Top-1 task | Recall@4 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['task_id']} | {row['prompt']} | {row['ndcg']:.3f} | "
        f"{row['top1_task_accuracy']:.1%} | {row['task_recall_at4']:.1%} |"
        for row in hardest_dino
    )
    lines.extend(
        [
            "",
            "### Hardest Tasks for Pi0 Full",
            "",
            "| Task | Instruction | nDCG | Top-1 task | Recall@4 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['task_id']} | {row['prompt']} | {row['ndcg']:.3f} | "
        f"{row['top1_task_accuracy']:.1%} | {row['task_recall_at4']:.1%} |"
        for row in hardest_pi0
    )
    lines.extend(
        [
            "",
            "The most frequent DINO confusions occur among visually similar cabinet/bowl tasks:",
            "",
            ", ".join(f"`{row['query_task']}→{row['retrieved_task']}` ({row['count']})" for row in dino_confusions)
            + ".",
            "",
            "Pi0's most frequent confusions include both visual and instruction-related pairs:",
            "",
            ", ".join(f"`{row['query_task']}→{row['retrieved_task']}` ({row['count']})" for row in pi0_confusions)
            + ".",
            "",
            "## Practical Interpretation",
            "",
            "1. **Known task or exact prompt:** use a prompt/task filter first, then retrieve phase using DINO+state. "
            "The exact combination prompt-gated DINO+state was not directly scored, but the two benchmarks strongly "
            "motivate testing it next.",
            "2. **Unknown task without reliable text matching:** DINO top+state is the safest current choice.",
            "3. **Learned multimodal retrieval:** Pi0 full clearly uses prompt information, but its early-fusion embedding "
            "does not outperform DINO+state overall.",
            "4. **Wrist image:** wrist features help consistently, but state supplies a slightly stronger and cheaper gain.",
            "",
            "## Limitations",
            "",
            "- These are offline retrieval benchmarks, not policy-success evaluations.",
            "- Future expert actions are used only for scoring and the reporting-only oracle.",
            "- The 25 tasks were selected because at least one prior policy succeeded on them; results may be optimistic "
            "relative to all LIBERO-90 tasks.",
            "- Mixed retrieval covers 25 tasks, not a full 90-task memory.",
            "- Prompt gating requires exact prompt equality and has not been tested against paraphrases or noisy language.",
            "- Conditional action RMSE excludes queries without a correct-task retrieval.",
            "- Pi0 uses the earliest fused prefix representation, not a later semantic or action-conditioned layer.",
            "- Cached retrieval latency does not include one-time feature extraction.",
            "",
            "## Reproducibility",
            "",
            f"- Normal config: `{same_manifest['normal_config']}`",
            f"- Normal checkpoint: `{same_manifest['normal_checkpoint']}`",
            f"- Source datasets: `{same_manifest['datasets'][0]}`, `{same_manifest['datasets'][1]}`",
            f"- Retrieval memory for tasks 20-29: `{same_manifest['retrieval_dirs'][0]}`",
            f"- Retrieval memory for tasks 30-69: `{same_manifest['retrieval_dirs'][1]}`",
            f"- Hybrid calibration: `{args.same_root / 'calibration.json'}`",
            f"- Shuffled-prompt assignment: `{args.mixed_root / 'shuffled_prompt_map_full.json'}`",
            f"- Task-specific manifest: `{args.same_root / 'manifest.json'}`",
            f"- Mixed-task manifest: `{args.mixed_root / 'manifest.json'}`",
            f"- Task-specific detailed results: `{args.same_root / 'full' / 'queries.jsonl'}`",
            f"- Mixed-task detailed results: `{args.mixed_root / 'full' / 'queries.jsonl'}`",
            f"- Task-specific report: `{same_manifest['outputs']['xlsx']}`",
            f"- Mixed-task report: `{mixed_manifest['outputs']['xlsx']}`",
            f"- Task-specific command: `{same_manifest['command']}`",
            f"- Mixed-task command: `{mixed_manifest['command']}`",
            "- Combined analysis generator: `scripts/export_libero90_retrieval_combined_analysis.py`",
            "",
            "The manifests record the checkpoint identity hashes, dataset metadata hashes, package versions, CUDA "
            "device, git commit, working-tree diff, cache sizes, and exact output paths.",
            "",
            "## Appendix: Included Task Prompts",
            "",
            "These are the exact prompt strings stored in the benchmark results. Tasks `20` and `44`, tasks `25` and "
            "`31`, and tasks `46` and `50` intentionally share prompt text. That duplication is why prompt gating is "
            "not equivalent in principle to task-ID gating.",
            "",
            "| Task ID | Exact prompt |",
            "|---:|---|",
        ]
    )
    lines.extend(f"| {task_id} | {prompts[task_id]} |" for task_id in sorted(prompts))
    return "\n".join(lines)


def plot_same_ranking(rows: dict[str, dict[str, str]], output: Path) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda row: num(row["best_of_4_action_rmse"]),
        reverse=True,
    )
    labels = [SAME_NAMES[row["method"]] for row in ordered]
    values = [num(row["best_of_4_action_rmse"]) for row in ordered]
    colors = [
        "#2f7d32"
        if row["method"] == "dino_top_state"
        else "#777777"
        if row["method"] in {"random", "oracle_action"}
        else "#3f78a8"
        for row in ordered
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Best-of-4 future-action RMSE (lower is better)")
    ax.set_title("Benchmark 1: task-specific retrieval")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_same_deltas(rows: dict[str, dict[str, str]], output: Path) -> None:
    methods = (
        "state",
        "dino_top_wrist",
        "dino_top_state",
        "dino_top_wrist_state",
        "pi0_fast_first_fusion",
        "pi0_fast_first_fusion_no_prompt",
    )
    means = [num(rows[method]["mean_delta_best_of_4_action_rmse"]) for method in methods]
    low = [num(rows[method]["ci95_low"]) for method in methods]
    high = [num(rows[method]["ci95_high"]) for method in methods]
    errors = np.asarray(
        [
            [mean - lower for mean, lower in zip(means, low, strict=True)],
            [upper - mean for mean, upper in zip(means, high, strict=True)],
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(methods))
    ax.errorbar(
        means,
        positions,
        xerr=errors,
        fmt="o",
        color="#2f5f8f",
        ecolor="#7796b5",
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(positions, [SAME_NAMES[method] for method in methods])
    ax.set_xlabel("Paired RMSE difference versus DINO top (negative is better)")
    ax.set_title("Task-bootstrap 95% confidence intervals")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mixed_ranking(rows: dict[str, dict[str, str]], output: Path) -> None:
    methods = (
        "random_global",
        "state_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_global",
        "dino_top_global",
        "dino_top_wrist_global",
        "dino_top_wrist_state_global",
        "dino_top_state_global",
        "prompt_gated_dino_top",
        "oracle_action_task",
    )
    values = [num(rows[method]["task_aware_ndcg_at4"]) for method in methods]
    colors = [
        "#2f7d32"
        if method == "dino_top_state_global"
        else "#7a4e9d"
        if method.startswith("pi0")
        else "#777777"
        if method in {"random_global", "oracle_action_task"}
        else "#3f78a8"
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh([MIXED_NAMES[method] for method in methods], values, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Task-aware nDCG@4 (higher is better)")
    ax.set_title("Benchmark 2: global mixed-task retrieval")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_selection_phase(rows: dict[str, dict[str, str]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    for method in PRACTICAL_MIXED:
        row = rows[method]
        x = num(row["task_recall_at4"])
        y = num(row["conditional_best_action_rmse"])
        size = 100 + 400 * num(row["task_aware_ndcg_at4"])
        ax.scatter(x, y, s=size, alpha=0.8)
        ax.annotate(
            MIXED_NAMES[method],
            (x, y),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Correct-task recall@4 (higher is better)")
    ax.set_ylabel("Conditional action RMSE (lower is better)")
    ax.set_title("Task identification versus within-task action matching")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_prompt_ablation(rows: dict[str, dict[str, str]], output: Path) -> None:
    methods = (
        "pi0_fast_first_fusion_global",
        "pi0_fast_first_fusion_no_prompt_global",
        "pi0_fast_first_fusion_shuffled_prompt_global",
    )
    metrics = (
        ("task_aware_ndcg_at4", "nDCG@4"),
        ("top1_task_accuracy", "Top-1 task"),
        ("task_recall_at4", "Recall@4"),
        ("useful_retrieval_at4", "Useful@4"),
    )
    positions = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ("#2f7d32", "#4c78a8", "#b65f5f")
    for index, (method, color) in enumerate(zip(methods, colors, strict=True)):
        values = [num(rows[method][metric]) for metric, _ in metrics]
        ax.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=MIXED_NAMES[method],
            color=color,
        )
    ax.set_xticks(positions, [label for _, label in metrics])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Pi0 prompt ablation in mixed memory")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_cross_rmse(
    same: dict[str, dict[str, str]],
    mixed: dict[str, dict[str, str]],
    output: Path,
) -> None:
    methods = tuple(METHOD_PAIRS)
    same_values = [num(same[method]["best_of_4_action_rmse"]) for method in methods]
    mixed_values = [num(mixed[METHOD_PAIRS[method]]["conditional_best_action_rmse"]) for method in methods]
    positions = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(positions - width / 2, same_values, width, label="Task-specific", color="#4c78a8")
    ax.bar(
        positions + width / 2,
        mixed_values,
        width,
        label="Mixed, conditional on correct-task hit",
        color="#f2a541",
    )
    ax.set_xticks(
        positions,
        [SAME_NAMES[method] for method in methods],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Best-of-4 action RMSE")
    ax.set_title("Within-task action quality is mostly preserved after a correct-task hit")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_per_task(rows: list[dict[str, str]], output: Path) -> None:
    methods = list(PRACTICAL_MIXED)
    tasks = sorted({int(row["task_id"]) for row in rows})
    lookup = {(row["method"], int(row["task_id"])): num(row["task_aware_ndcg_at4"]) for row in rows}
    matrix = np.asarray([[lookup[(method, task)] for task in tasks] for method in methods])
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(tasks)), tasks, rotation=90)
    ax.set_yticks(range(len(methods)), [MIXED_NAMES[method] for method in methods])
    ax.set_xlabel("LIBERO task")
    ax.set_title("Mixed-memory task-aware nDCG@4")
    fig.colorbar(image, ax=ax, label="nDCG@4")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pi0_confusion(rows: list[dict[str, str]], output: Path) -> None:
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
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(MIXED_NAMES[method])
        ax.set_xlabel("Retrieved task")
        ax.set_ylabel("Query task")
        ax.set_xticks(range(len(tasks)), tasks, rotation=90, fontsize=6)
        ax.set_yticks(range(len(tasks)), tasks, fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.75, label="Top-1 fraction")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def winner_counts(
    rows: list[dict[str, str]],
    *,
    metric: str,
    methods: set[str],
    maximize: bool,
) -> dict[str, int]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] in methods:
            grouped[int(row["task_id"])].append(row)
    counts: Counter[str] = Counter()
    for task_rows in grouped.values():
        best = (max if maximize else min)(num(row[metric]) for row in task_rows)
        for row in task_rows:
            if num(row[metric]) == best:
                counts[row["method"]] += 1
    return dict(counts)


def load_task_prompts(path: Path) -> dict[int, str]:
    prompts = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            prompts.setdefault(int(row["task_id"]), str(row["prompt"]))
    return prompts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def keyed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["method"]: row for row in rows}


def num(value: object) -> float:
    return float(value)


if __name__ == "__main__":
    main()
