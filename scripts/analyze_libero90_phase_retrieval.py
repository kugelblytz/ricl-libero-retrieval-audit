#!/usr/bin/env python3
"""Analyze when each task is retrieved during the final LIBERO-90 run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import textwrap
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITIONS = ("dino_global", "dino_state_global", "pi_global", "random_global")
LABELS = {
    "dino_global": "DINO",
    "dino_state_global": "DINO+State",
    "pi_global": "pi0-FF",
    "random_global": "Random",
}
COLORS = {
    "dino_global": "#4C78A8",
    "dino_state_global": "#238B45",
    "pi_global": "#CC6677",
    "random_global": "#8F8F8F",
}
PHASES = (
    "Approach / setup",
    "Contact / grasp / actuate",
    "Transport / manipulate",
    "Place / release / finish",
)
PHASE_SHORT = ("Approach", "Contact / grasp", "Transport", "Place / finish")
DEFAULT_PRIMARY_ROOT = Path("data/libero/videos/libero90_retrieval_final10_global25_disjoint_20260810_211205")
DEFAULT_EXTENSION_ROOT = Path("data/libero/videos/libero90_retrieval_extension15_global25_disjoint_20260811_145749")
DEFAULT_PRIMARY_PROTOCOL = Path("scripts/libero90_final10_protocol.json")
DEFAULT_EXTENSION_PROTOCOL = Path("scripts/libero90_extension15_protocol.json")
DEFAULT_POOL = Path("ricl_libero_preprocessing/libero90_solvable25_spread20/retrieval")
DEFAULT_OUTPUT = Path("data/libero/reports/libero90_closed_loop_retrieval/phase_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--primary-protocol", type=Path, default=DEFAULT_PRIMARY_PROTOCOL)
    parser.add_argument("--extension-protocol", type=Path, default=DEFAULT_EXTENSION_PROTOCOL)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _trial_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])


def _phase_index(step: int, elapsed_steps: int) -> int:
    """Map a policy step to a normalized rollout-time quartile."""
    return min(3, max(0, int(4 * int(step) / max(int(elapsed_steps), 1))))


def _instructions(protocols: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for protocol in protocols:
        for task_id, row in protocol["per_task"].items():
            result[int(task_id)] = str(row["instruction"])
    return result


def _candidate_phase_lookup(pool: Path) -> dict[tuple[int, int, int], int]:
    lookup: dict[tuple[int, int, int], int] = {}
    for task_dir in sorted(pool.glob("task_*"), key=lambda path: int(path.name.split("_")[-1])):
        task_id = int(task_dir.name.split("_")[-1])
        for episode_path in sorted(task_dir.glob("episode_*.npz"), key=lambda path: int(path.stem.split("_")[-1])):
            episode_id = int(episode_path.stem.split("_")[-1])
            with np.load(episode_path, allow_pickle=False) as episode:
                frame_indices = np.asarray(episode["frame_indices"], dtype=np.int32)
            for step_index, frame_index in enumerate(frame_indices):
                phase = min(3, int(4 * step_index / max(len(frame_indices), 1)))
                key = (task_id, episode_id, int(frame_index))
                if key in lookup:
                    raise ValueError(f"Duplicate candidate frame identity: {key}")
                lookup[key] = phase
    return lookup


def _load_slots(
    roots: list[Path],
    instructions: dict[int, str],
    candidate_phases: dict[tuple[int, int, int], int],
) -> tuple[pd.DataFrame, dict[str, dict[tuple[int, int, int], dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    canonical_by_condition: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for condition in CONDITIONS:
        canonical: dict[tuple[int, int, int], dict[str, Any]] = {}
        for root in roots:
            for row in _read_jsonl(root / condition / "retrieval_trace.jsonl"):
                if row.get("error") is not None:
                    raise ValueError(f"Infrastructure error in {condition} {_trial_key(row)}: {row['error']}")
                canonical[_trial_key(row)] = row
        if len(canonical) != 250:
            raise ValueError(f"Expected 250 canonical traces for {condition}, found {len(canonical)}")
        canonical_by_condition[condition] = canonical
        for (target_task, scene, trial), row in sorted(canonical.items()):
            trace = row["retrieval_trace"]
            by_step: dict[int, list[dict[str, Any]]] = {}
            for item in trace:
                by_step.setdefault(int(item["policy_step"]), []).append(item)
            if any(sorted(int(item["rank"]) for item in items) != [0, 1, 2, 3] for items in by_step.values()):
                raise ValueError(f"Expected ranks 0..3 at every query for {condition} {(target_task, scene, trial)}")
            for item in trace:
                source_task = int(item["retrieved_task_id"])
                source_key = (source_task, int(item["episode_id"]), int(item["frame_index"]))
                if source_key not in candidate_phases:
                    raise KeyError(f"Retrieved frame is absent from frozen memory: {source_key}")
                query_phase = _phase_index(int(item["policy_step"]), int(row["elapsed_steps"]))
                records.append(
                    {
                        "condition": condition,
                        "method_label": LABELS[condition],
                        "target_task": target_task,
                        "target_instruction": instructions[target_task],
                        "scene_setup_id": scene,
                        "trial_id": trial,
                        "success": bool(row["success"]),
                        "elapsed_steps": int(row["elapsed_steps"]),
                        "policy_step": int(item["policy_step"]),
                        "query_phase": query_phase,
                        "query_phase_label": PHASES[query_phase],
                        "rank": int(item["rank"]),
                        "source_task": source_task,
                        "source_instruction": instructions[source_task],
                        "source_phase": candidate_phases[source_key],
                        "source_phase_label": PHASES[candidate_phases[source_key]],
                        "exact_task": source_task == target_task,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No retrieval slots found")
    return frame, canonical_by_condition


TRIAL_PHASE = ["condition", "target_task", "scene_setup_id", "trial_id", "success", "query_phase"]


def _phase_exact_table(slots: pd.DataFrame) -> pd.DataFrame:
    trial = (
        slots.groupby(TRIAL_PHASE, observed=True)
        .agg(
            exact_fraction=("exact_task", "mean"),
            retrieved_slots=("exact_task", "size"),
        )
        .reset_index()
    )
    rows = []
    status_filters = {
        "all": pd.Series(data=True, index=trial.index),
        "success": trial["success"],
        "failure": ~trial["success"],
    }
    for status, mask in status_filters.items():
        selected_trial = trial[mask]
        selected_slots = slots if status == "all" else slots[slots["success"] == (status == "success")]
        per_task = (
            selected_trial.groupby(["condition", "query_phase", "target_task"], observed=True)
            .agg(
                task_scene_exact_fraction=("exact_fraction", "mean"),
                rollouts_with_phase=("trial_id", "size"),
            )
            .reset_index()
        )
        task_equal = (
            per_task.groupby(["condition", "query_phase"], observed=True)
            .agg(
                exact_task_fraction_equal_task_scene=("task_scene_exact_fraction", "mean"),
                task_count=("target_task", "nunique"),
                rollouts_with_phase=("rollouts_with_phase", "sum"),
            )
            .reset_index()
        )
        slot_equal = (
            selected_slots.groupby(["condition", "query_phase"], observed=True)
            .agg(
                exact_task_fraction_slot_weighted=("exact_task", "mean"),
                retrieved_slots=("exact_task", "size"),
            )
            .reset_index()
        )
        merged = task_equal.merge(slot_equal, on=["condition", "query_phase"], how="left")
        merged["status"] = status
        rows.append(merged)
    result = pd.concat(rows, ignore_index=True)
    result["method_label"] = result["condition"].map(LABELS)
    result["query_phase_label"] = result["query_phase"].map(dict(enumerate(PHASES)))
    return result[
        [
            "condition",
            "method_label",
            "query_phase",
            "query_phase_label",
            "status",
            "task_count",
            "rollouts_with_phase",
            "retrieved_slots",
            "exact_task_fraction_equal_task_scene",
            "exact_task_fraction_slot_weighted",
        ]
    ].sort_values(["condition", "status", "query_phase"])


def _balanced_distribution(
    slots: pd.DataFrame,
    value_column: str,
    values: list[int],
    output_name: str,
) -> pd.DataFrame:
    counts = slots.groupby([*TRIAL_PHASE, value_column], observed=True).size().unstack(fill_value=0)  # noqa: PD010
    counts = counts.reindex(columns=values, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.columns.name = value_column
    fractions = fractions.stack(future_stack=True).rename("rollout_fraction").reset_index()  # noqa: PD013
    per_task = (
        fractions.groupby(["condition", "query_phase", "target_task", value_column], observed=True)["rollout_fraction"]
        .mean()
        .reset_index(name="task_scene_fraction")
    )
    balanced = (
        per_task.groupby(["condition", "query_phase", value_column], observed=True)
        .agg(
            **{output_name: ("task_scene_fraction", "mean")},
            task_count=("target_task", "nunique"),
        )
        .reset_index()
    )
    raw = (
        slots.groupby(["condition", "query_phase", value_column], observed=True)
        .size()
        .reset_index(name="retrieved_slots")
    )
    raw_total = raw.groupby(["condition", "query_phase"], observed=True)["retrieved_slots"].transform("sum")
    raw["slot_weighted_fraction"] = raw["retrieved_slots"] / raw_total
    return balanced.merge(raw, on=["condition", "query_phase", value_column], how="left").fillna(
        {"retrieved_slots": 0, "slot_weighted_fraction": 0.0}
    )


def _phase_alignment_table(slots: pd.DataFrame) -> pd.DataFrame:
    result = _balanced_distribution(slots, "source_phase", list(range(4)), "equal_task_scene_fraction")
    result["method_label"] = result["condition"].map(LABELS)
    result["query_phase_label"] = result["query_phase"].map(dict(enumerate(PHASES)))
    result["source_phase_label"] = result["source_phase"].map(dict(enumerate(PHASES)))
    return result.sort_values(["condition", "query_phase", "source_phase"])


def _source_task_table(slots: pd.DataFrame, task_ids: list[int], instructions: dict[int, str]) -> pd.DataFrame:
    result = _balanced_distribution(slots, "source_task", task_ids, "equal_task_scene_fraction")
    result["method_label"] = result["condition"].map(LABELS)
    result["query_phase_label"] = result["query_phase"].map(dict(enumerate(PHASES)))
    result["source_instruction"] = result["source_task"].map(instructions)
    return result.sort_values(["condition", "query_phase", "source_task"])


def _route_table(slots: pd.DataFrame, task_ids: list[int], instructions: dict[int, str]) -> pd.DataFrame:
    counts = slots.groupby([*TRIAL_PHASE, "source_task"], observed=True).size().unstack(fill_value=0)  # noqa: PD010
    counts = counts.reindex(columns=task_ids, fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    fractions.columns.name = "source_task"
    fractions = fractions.stack(future_stack=True).rename("rollout_fraction").reset_index()  # noqa: PD013
    task_phase = (
        fractions.groupby(["condition", "query_phase", "target_task", "source_task"], observed=True)["rollout_fraction"]
        .mean()
        .reset_index(name="mean_scene_fraction")
    )
    raw = (
        slots.groupby(["condition", "query_phase", "target_task", "source_task"], observed=True)
        .size()
        .reset_index(name="retrieved_slots")
    )
    result = task_phase.merge(raw, on=["condition", "query_phase", "target_task", "source_task"], how="left").fillna(
        {"retrieved_slots": 0}
    )
    result["method_label"] = result["condition"].map(LABELS)
    result["query_phase_label"] = result["query_phase"].map(dict(enumerate(PHASES)))
    result["target_instruction"] = result["target_task"].map(instructions)
    result["source_instruction"] = result["source_task"].map(instructions)
    result["exact_task"] = result["target_task"] == result["source_task"]
    result["same_instruction"] = result["target_instruction"] == result["source_instruction"]
    return result.sort_values(["condition", "query_phase", "target_task", "source_task"])


def _plot_exact_task(phase_exact: pd.DataFrame, output: Path) -> None:
    fig, (ax_all, ax_outcome) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    x = np.arange(4)
    for condition in CONDITIONS:
        rows = phase_exact[(phase_exact["condition"] == condition) & (phase_exact["status"] == "all")].sort_values(
            "query_phase"
        )
        ax_all.plot(
            x,
            100 * rows["exact_task_fraction_equal_task_scene"],
            marker="o",
            lw=2.3,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    for condition in CONDITIONS[:3]:
        for status, linestyle, alpha in (("success", "-", 1.0), ("failure", "--", 0.72)):
            rows = phase_exact[(phase_exact["condition"] == condition) & (phase_exact["status"] == status)].sort_values(
                "query_phase"
            )
            ax_outcome.plot(
                x,
                100 * rows["exact_task_fraction_equal_task_scene"],
                marker="o",
                lw=2.0,
                ls=linestyle,
                alpha=alpha,
                color=COLORS[condition],
                label=f"{LABELS[condition]} — {status}",
            )
    for ax in (ax_all, ax_outcome):
        ax.set_xticks(x, PHASE_SHORT, rotation=14, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Exact-task retrieved slots (%)")
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    ax_all.set_title("A  All rollouts (equal task and scene weighting)")
    ax_outcome.set_title("B  Successful vs. failed rollouts")
    ax_all.legend(frameon=False, ncol=2, fontsize=9)
    ax_outcome.legend(frameon=False, fontsize=8.2, ncol=2)
    fig.suptitle("Exact-task retrieval across normalized policy phases", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_alignment(alignment: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.2, 4.1), sharex=True, sharey=True)
    image = None
    for ax, condition in zip(axes, CONDITIONS, strict=True):
        rows = alignment[alignment["condition"] == condition]
        matrix = rows.pivot(  # noqa: PD010
            index="query_phase", columns="source_phase", values="equal_task_scene_fraction"
        ).to_numpy()
        image = ax.imshow(100 * matrix, vmin=0, vmax=65, cmap="YlGnBu")
        for query_phase in range(4):
            for source_phase in range(4):
                value = 100 * matrix[query_phase, source_phase]
                ax.text(
                    source_phase,
                    query_phase,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color="white" if value > 38 else "#182234",
                )
        ax.set_title(LABELS[condition], fontweight="bold")
        ax.set_xticks(range(4), ("Early", "Contact", "Transport", "Finish"), rotation=35, ha="right")
        ax.set_yticks(range(4), ("Approach", "Contact", "Transport", "Finish"))
        ax.set_xlabel("Retrieved expert-frame phase")
    axes[0].set_ylabel("Policy query phase")
    assert image is not None
    fig.subplots_adjust(left=0.06, right=0.90, top=0.82, bottom=0.22, wspace=0.24)
    colorbar_axis = fig.add_axes([0.925, 0.22, 0.015, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Retrieved slots (%)")
    fig.suptitle("Policy phase → retrieved demonstration phase", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_source_heatmaps(source_tasks: pd.DataFrame, task_ids: list[int], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 9.0), sharex=True, sharey=True)
    vmax = max(0.12, float(source_tasks["equal_task_scene_fraction"].max()))
    image = None
    for ax, condition in zip(axes.flat, CONDITIONS, strict=True):
        rows = source_tasks[source_tasks["condition"] == condition]
        matrix = (
            rows.pivot(  # noqa: PD010
                index="source_task", columns="query_phase", values="equal_task_scene_fraction"
            )
            .reindex(task_ids)
            .to_numpy()
        )
        image = ax.imshow(100 * matrix, aspect="auto", vmin=0, vmax=100 * vmax, cmap="magma")
        ax.set_title(LABELS[condition], fontweight="bold")
        ax.set_yticks(range(len(task_ids)), task_ids, fontsize=7.5)
        ax.set_xticks(range(4), PHASE_SHORT, rotation=20, ha="right")
        ax.set_ylabel("Retrieved source task ID")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Share of retrieved slots (%)\n(equal task and scene weighting)")
    fig.suptitle("Which source tasks dominate each policy phase?", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.89, top=0.92, bottom=0.08, hspace=0.25, wspace=0.17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_example(
    roots: list[Path],
    canonical_by_condition: dict[str, dict[tuple[int, int, int], dict[str, Any]]],
    instructions: dict[int, str],
    output: Path,
) -> dict[str, Any]:
    key = (52, 25, 4)
    row = canonical_by_condition["dino_global"][key]
    root = next(
        root
        for root in roots
        if key in {_trial_key(candidate) for candidate in _read_jsonl(root / "dino_global" / "retrieval_trace.jsonl")}
    )
    videos = list((root / "dino_global" / "videos").glob("rollout_task_52_scene_025_trial_004_*_success.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one phase-example video, found {videos}")
    video_path = videos[0]
    reader = imageio.get_reader(video_path)
    frames = [np.asarray(frame) for frame in reader]
    reader.close()
    selected_indices = [min(len(frames) - 1, int(len(frames) * fraction)) for fraction in (0.125, 0.375, 0.625, 0.875)]
    top_sources = []
    for query_phase in range(4):
        counts = Counter(
            int(item["retrieved_task_id"])
            for item in row["retrieval_trace"]
            if _phase_index(int(item["policy_step"]), int(row["elapsed_steps"])) == query_phase
        )
        source_task, slots = counts.most_common(1)[0]
        top_sources.append(
            {
                "task_id": source_task,
                "instruction": instructions[source_task],
                "slots": slots,
                "phase_slots": sum(counts.values()),
            }
        )
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.25))
    for phase, (ax, frame_index, source) in enumerate(zip(axes, selected_indices, top_sources, strict=True)):
        ax.imshow(frames[frame_index])
        ax.axis("off")
        ax.set_title(PHASE_SHORT[phase], fontsize=10.5, fontweight="bold")
        ax.text(
            0.5,
            -0.07,
            textwrap.fill(f"Top source T{source['task_id']}: {source['instruction']}", width=38),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.5,
            wrap=True,
        )
    fig.suptitle(
        "Example successful rollout: T52 — pick up the milk and put it in the basket", fontsize=14, fontweight="bold"
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.2, wspace=0.03)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "condition": "dino_global",
        "task_id": 52,
        "scene_setup_id": 25,
        "trial_id": 4,
        "success": True,
        "video_path": str(video_path),
        "top_source_by_phase": top_sources,
    }


def _write_summary(
    output: Path,
    slots: pd.DataFrame,
    phase_exact: pd.DataFrame,
    alignment: pd.DataFrame,
    source_tasks: pd.DataFrame,
    routes: pd.DataFrame,
    example: dict[str, Any],
) -> None:
    methods = []
    for condition in CONDITIONS:
        exact_rows = phase_exact[
            (phase_exact["condition"] == condition) & (phase_exact["status"] == "all")
        ].sort_values("query_phase")
        outcome_exact = {}
        for status in ("success", "failure"):
            status_rows = phase_exact[
                (phase_exact["condition"] == condition) & (phase_exact["status"] == status)
            ].sort_values("query_phase")
            outcome_exact[status] = [float(value) for value in status_rows["exact_task_fraction_equal_task_scene"]]
        matrix_rows = alignment[alignment["condition"] == condition]
        matrix = matrix_rows.pivot(  # noqa: PD010
            index="query_phase", columns="source_phase", values="equal_task_scene_fraction"
        ).to_numpy()
        top_by_phase = []
        route_by_phase = []
        for phase in range(4):
            top = source_tasks[
                (source_tasks["condition"] == condition) & (source_tasks["query_phase"] == phase)
            ].nlargest(3, "equal_task_scene_fraction")
            top_by_phase.append(
                {
                    "phase": PHASES[phase],
                    "tasks": [
                        {
                            "task_id": int(row.source_task),
                            "instruction": row.source_instruction,
                            "fraction": float(row.equal_task_scene_fraction),
                        }
                        for row in top.itertuples()
                    ],
                }
            )
            top_routes = routes[
                (routes["condition"] == condition)
                & (routes["query_phase"] == phase)
                & (~routes["exact_task"])
                & (routes["retrieved_slots"] > 0)
            ].nlargest(3, "mean_scene_fraction")
            route_by_phase.append(
                {
                    "phase": PHASES[phase],
                    "routes": [
                        {
                            "target_task": int(row.target_task),
                            "target_instruction": row.target_instruction,
                            "source_task": int(row.source_task),
                            "source_instruction": row.source_instruction,
                            "mean_scene_fraction": float(row.mean_scene_fraction),
                        }
                        for row in top_routes.itertuples()
                    ],
                }
            )
        methods.append(
            {
                "condition": condition,
                "method_label": LABELS[condition],
                "retrieved_slots": int((slots["condition"] == condition).sum()),
                "exact_task_fraction_by_phase": [
                    float(value) for value in exact_rows["exact_task_fraction_equal_task_scene"]
                ],
                "outcome_exact_task_fraction_by_phase": outcome_exact,
                "query_to_demo_phase_matrix": matrix.tolist(),
                "top_source_tasks_by_phase": top_by_phase,
                "top_cross_task_routes_by_phase": route_by_phase,
            }
        )
    payload = {
        "analysis_status": "final_complete",
        "phase_definition": {
            "type": "normalized rollout-time proxy",
            "bins": ["[0%, 25%)", "[25%, 50%)", "[50%, 75%)", "[75%, 100%]"],
            "labels": list(PHASES),
            "query_assignment": "policy_step / elapsed_steps",
            "retrieved_frame_assignment": "cached expert-frame ordinal / cached episode length",
            "limitation": "Per-step query kinematics and contact flags were not saved. Approach/grasp/transport/place are transparent temporal-semantic labels, not event-detector annotations; for stove, drawer, and microwave tasks, grasp means contact or actuation.",
            "weighting": "Each rollout-phase is normalized first, then scenes are averaged within target task and tasks are averaged equally.",
        },
        "validation": {
            "conditions": 4,
            "rollouts_per_condition": 250,
            "total_rollouts": 1000,
            "retrieved_slots": int(len(slots)),
            "target_tasks": int(slots["target_task"].nunique()),
            "source_tasks": int(slots["source_task"].nunique()),
            "unmatched_retrieved_frames": 0,
        },
        "methods": methods,
        "example": example,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    protocols = [json.loads(args.primary_protocol.read_text()), json.loads(args.extension_protocol.read_text())]
    instructions = _instructions(protocols)
    task_ids = sorted(instructions)
    if len(task_ids) != 25:
        raise ValueError(f"Expected 25 tasks, found {task_ids}")
    candidate_phases = _candidate_phase_lookup(args.pool)
    slots, canonical = _load_slots([args.primary_root, args.extension_root], instructions, candidate_phases)
    phase_exact = _phase_exact_table(slots)
    alignment = _phase_alignment_table(slots)
    source_tasks = _source_task_table(slots, task_ids, instructions)
    routes = _route_table(slots, task_ids, instructions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    phase_exact.to_csv(args.output_dir / "phase_exact_task.csv", index=False)
    alignment.to_csv(args.output_dir / "phase_alignment.csv", index=False)
    source_tasks.to_csv(args.output_dir / "phase_source_tasks.csv", index=False)
    routes.to_csv(args.output_dir / "phase_routes.csv", index=False)
    _plot_exact_task(phase_exact, figures / "phase_exact_task.png")
    _plot_alignment(alignment, figures / "phase_alignment.png")
    _plot_source_heatmaps(source_tasks, task_ids, figures / "phase_source_task_heatmaps.png")
    example = _plot_example(
        [args.primary_root, args.extension_root], canonical, instructions, figures / "phase_example_t52.png"
    )
    _write_summary(args.output_dir / "summary.json", slots, phase_exact, alignment, source_tasks, routes, example)
    print(json.dumps({"analysis_status": "final_complete", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
