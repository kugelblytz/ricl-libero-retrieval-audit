#!/usr/bin/env python3
"""Create the concise final PDF for the unified LIBERO-90 retrieval run."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

DEFAULT_ANALYSIS_DIR = Path("data/libero/reports/libero90_closed_loop_retrieval")
DEFAULT_PHASE_DIR = DEFAULT_ANALYSIS_DIR / "phase_analysis"
DEFAULT_OUTPUT = Path("data/libero/reports/libero90_global_retrieval_final_run_20260817.pdf")

COLORS = {
    "ink": "#172033",
    "muted": "#5B6473",
    "rule": "#D8DEE9",
    "panel": "#F3F6FA",
    "accent": "#31688E",
    "success": "#197A55",
    "dino_global": "#4C78A8",
    "dino_state_global": "#238B45",
    "pi_global": "#CC6677",
    "random_global": "#9E9E9E",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--phase-dir", type=Path, default=DEFAULT_PHASE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


TASK_SHORT = {
    20: "stove on",
    21: "stove on + pan",
    24: "black bowl → bottom drawer",
    25: "black bowl → cabinet top",
    28: "close top drawer",
    29: "black bowl → top drawer",
    30: "black bowl → plate",
    31: "black bowl → cabinet top",
    34: "yellow mug → front of white mug",
    35: "open microwave",
    43: "white bowl → cabinet top",
    44: "stove on",
    46: "alphabet soup → basket",
    49: "tomato sauce → basket",
    50: "alphabet soup → basket",
    51: "butter → basket",
    52: "milk → basket",
    55: "alphabet soup → tray",
    56: "butter → tray",
    57: "cream cheese → tray",
    59: "tomato sauce → tray",
    60: "left black bowl → tray",
    61: "pudding → tray",
    68: "yellow mug → right plate",
    69: "pudding → left of plate",
}

TASK_TINY = {
    20: "stove",
    21: "stove+pan",
    24: "bowl→bottom dr.",
    25: "bowl→cab. top",
    28: "close drawer",
    29: "bowl→top dr.",
    30: "bowl→plate",
    31: "bowl→cab. top",
    34: "mug relation",
    35: "open microwave",
    43: "white bowl→top",
    44: "stove",
    46: "soup→basket",
    49: "tomato→basket",
    50: "soup→basket",
    51: "butter→basket",
    52: "milk→basket",
    55: "soup→tray",
    56: "butter→tray",
    57: "cheese→tray",
    59: "tomato→tray",
    60: "left bowl→tray",
    61: "pudding→tray",
    68: "mug→right plate",
    69: "pudding→left plate",
}


def _wrapped(text: str, width: int) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def _bullet(fig: plt.Figure, x: float, y: float, text: str, width: int = 58, size: float = 9.3) -> None:
    fig.text(x, y, "•", fontsize=size + 1.5, color=COLORS["accent"], va="top")
    fig.text(
        x + 0.018,
        y,
        _wrapped(text, width),
        fontsize=size,
        color=COLORS["ink"],
        va="top",
        linespacing=1.28,
    )


def _section(fig: plt.Figure, y: float, title: str) -> None:
    fig.text(0.06, y, title, fontsize=13.2, fontweight="bold", color=COLORS["ink"], va="top")
    fig.add_artist(plt.Line2D([0.06, 0.94], [y - 0.012, y - 0.012], color=COLORS["rule"], lw=1.0))


def _footer(fig: plt.Figure, page: int, source: str) -> None:
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.043, 0.043], color=COLORS["rule"], lw=0.8))
    fig.text(0.06, 0.025, source, fontsize=6.8, color=COLORS["muted"], va="center")
    fig.text(0.94, 0.025, str(page), fontsize=7.2, color=COLORS["muted"], ha="right", va="center")


def _format_p(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value:.2g}"
    return f"{value:.3f}"


def _page_one(summary: dict, analysis_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.06,
        0.965,
        "Global Retrieval in Closed-Loop LIBERO-90",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.06,
        0.928,
        f"Concise final report  •  25 tasks  •  completed 13 Aug 2026  •  "
        f"generated {datetime.now(UTC).date():%d %b %Y}",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )

    callout = FancyBboxPatch(
        (0.06, 0.846),
        0.88,
        0.058,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        transform=fig.transFigure,
        fc="#EAF5F0",
        ec="#B8DCCA",
        lw=1.0,
    )
    fig.add_artist(callout)
    fig.text(0.078, 0.885, "FINAL STATUS", fontsize=8.5, fontweight="bold", color=COLORS["success"], va="top")
    fig.text(
        0.078,
        0.865,
        "Complete: 1,000/1,000 canonical rollouts and retrieval traces validated; zero infrastructure errors.",
        fontsize=11.1,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )

    _section(fig, 0.814, "Frozen protocol")
    _bullet(
        fig,
        0.06,
        0.783,
        "25 baseline-RICL-capable tasks selected from LIBERO tasks 20-69. The estimand is conditional on this capability-screened subset, not full LIBERO-90 coverage.",
        width=50,
        size=8.7,
    )
    _bullet(
        fig,
        0.535,
        0.783,
        "One shared global memory: 20 demonstrations per task, 500 episodes and 69,990 candidate frames. Every method retrieves four frames from the same memory.",
        width=48,
        size=8.7,
    )
    _bullet(
        fig,
        0.06,
        0.717,
        "Ten frozen final scenes per task, disjoint from both memory and capability-screening scenes. All selectors receive the same 250 paired task-scene identities.",
        width=50,
        size=8.7,
    )
    _bullet(
        fig,
        0.535,
        0.717,
        "Fixed pi0-FAST RICL consumer. DINO uses top-camera DINOv2; DINO+State adds normalized proprioception; pi0-FF uses top/wrist images, state and prompt; random is uniform. Five-step replanning; no interpolation.",
        width=48,
        size=8.35,
    )

    _section(fig, 0.647, "Final pooled results")
    closed = {row["condition"]: row for row in summary["closed_loop"]}
    offline = {row["condition"]: row for row in summary["offline_global"]}
    expert = {row["condition"]: row for row in summary["matched_expert"]}
    order = ("dino_global", "dino_state_global", "pi_global", "random_global")
    table_rows = []
    for condition in order:
        row = closed[condition]
        exp = expert.get(condition)
        table_rows.append(
            [
                row["method_label"],
                f'{row["successes"]}/250',
                f'{100 * row["success_rate"]:.1f}%',
                f'[{100 * row["success_ci95_low"]:.1f}, {100 * row["success_ci95_high"]:.1f}]',
                f'{100 * row["correct_task_fraction"]:.1f}%',
                "—" if exp is None else f'{100 * exp["correct_task_fraction"]:.1f}%',
                f'{offline[condition]["global_task_aware_ndcg_at4"]:.4f}',
            ]
        )
    ax = fig.add_axes([0.06, 0.428, 0.88, 0.19])
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Selector",
            "Successes",
            "Success",
            "Task-hier. 95% CI",
            "On-policy\nexact task",
            "Expert\nexact task",
            "Offline\nnDCG@4",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.17, 0.12, 0.105, 0.19, 0.14, 0.14, 0.135],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)  # noqa: FBT003 - matplotlib requires a positional flag.
    table.set_fontsize(8.8)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F5F7FA" if row_idx % 2 else "#E9EEF4")
            if col_idx == 0:
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(COLORS[order[row_idx - 1]])

    _section(fig, 0.393, "What the completed run supports")
    _bullet(
        fig,
        0.06,
        0.362,
        "No informed selector is a closed-loop winner: pi0-FF (40.4%), DINO (40.0%) and DINO+State (36.8%) are separated by only 3.6 points, and every informed-method paired interval crosses zero.",
        width=112,
        size=9.5,
    )
    _bullet(
        fig,
        0.06,
        0.311,
        "Offline ordering does not transfer to control: DINO+State has the best expert-query nDCG@4 (0.7897; better than DINO on all 25 tasks) but the lowest informed closed-loop rate.",
        width=112,
        size=9.5,
    )
    _bullet(
        fig,
        0.06,
        0.260,
        "Random context reaches 26.0%. Under the pre-specified two-part control rule, only pi0-FF is established above random: +14.4 pp, task-bootstrap CI [2.8, 26.0], Holm-adjusted McNemar p=0.000218.",
        width=112,
        size=9.5,
    )
    _bullet(
        fig,
        0.06,
        0.209,
        "Correct-task retrieval falls sharply on policy trajectories. Early exact-task retrieval is positively associated with eventual success for all informed selectors, but this association is not causal evidence.",
        width=112,
        size=9.5,
    )

    execution = FancyBboxPatch(
        (0.06, 0.078),
        0.88,
        0.082,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        transform=fig.transFigure,
        fc=COLORS["panel"],
        ec=COLORS["rule"],
        lw=0.9,
    )
    fig.add_artist(execution)
    fig.text(0.078, 0.143, "EXECUTION NOTE", fontsize=8.3, fontweight="bold", color=COLORS["muted"], va="top")
    fig.text(
        0.078,
        0.124,
        _wrapped(
            "The study ran first on a deterministic ten-task subset (10-12 Aug) and then on all remaining 15 capable tasks (12-13 Aug). The extension was chosen after some first-batch outcomes were visible, but no remaining capable task was selected or excluded by outcome; code and protocol were unchanged. This report aggregates all 25 tasks once.",
            132,
        ),
        fontsize=8.35,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )
    _footer(fig, 1, f"Analysis: {analysis_dir / 'summary.json'}")
    return fig


def _page_two(summary: dict, analysis_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.06, 0.965, "Diagnostics and paired evidence", fontsize=20, fontweight="bold", color=COLORS["ink"], va="top"
    )
    fig.text(
        0.06,
        0.931,
        "Expert-query retrieval, closed-loop control, matched-scene shift, and taskwise transfer",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )

    image_path = analysis_dir / "figures" / "main_four_panel.png"
    ax_img = fig.add_axes([0.055, 0.492, 0.89, 0.407])
    ax_img.imshow(mpimg.imread(image_path))
    ax_img.axis("off")
    fig.text(
        0.06,
        0.474,
        "A: held-out expert nDCG@4.  B: closed-loop success with task-hierarchical 95% CIs.  C: exact-task retrieval on matched expert and policy scenes.  D: taskwise offline deltas do not predict online deltas.",
        fontsize=7.7,
        color=COLORS["muted"],
        va="top",
    )

    _section(fig, 0.438, "Paired comparisons")
    comparisons = {row["comparison"]: row for row in summary["paired_comparisons"]}
    names = [
        "DINO+State minus DINO",
        "pi0-FF minus DINO+State",
        "pi0-FF minus DINO",
        "DINO minus random",
        "DINO+State minus random",
        "pi0-FF minus random",
    ]
    decisions = {
        "DINO+State minus DINO": "Unresolved",
        "pi0-FF minus DINO+State": "Unresolved",
        "pi0-FF minus DINO": "Descriptive; unresolved",
        "DINO minus random": "Control rule not met",
        "DINO+State minus random": "Control rule not met",
        "pi0-FF minus random": "Outperforms random",
    }
    rows = []
    for name in names:
        row = comparisons[name]
        rows.append(
            [
                name.replace(" minus ", " - "),
                f'{row["success_rate_difference_pp"]:+.1f}',
                f'[{row["difference_ci95_low_pp"]:.1f}, {row["difference_ci95_high_pp"]:.1f}]',
                _format_p(row["holm_adjusted_p"]),
                decisions[name],
            ]
        )
    ax = fig.add_axes([0.06, 0.229, 0.88, 0.18])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Comparison", "Δ success\n(pp)", "Task-hier. 95% CI\n(pp)", "Holm p", "Decision"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.30, 0.12, 0.21, 0.12, 0.25],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)  # noqa: FBT003 - matplotlib requires a positional flag.
    table.set_fontsize(8.3)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F5F7FA" if row_idx % 2 else "#E9EEF4")
            if row_idx == 6 and col_idx == 4:
                cell.get_text().set_color(COLORS["success"])
                cell.get_text().set_fontweight("bold")

    _section(fig, 0.198, "Interpretive checks")
    _bullet(
        fig,
        0.06,
        0.168,
        "Matched expert → on-policy exact-task retrieval: DINO 87.2% → 51.8%, DINO+State 87.9% → 48.9%, pi0-FF 83.1% → 38.7%. These are descriptive distribution shifts, not causal effects.",
        width=115,
        size=8.9,
    )
    _bullet(
        fig,
        0.06,
        0.126,
        "Within tasks, the first ten retrievals are more often from the exact task in successful than failed rollouts by +17.1 pp (DINO), +23.4 pp (DINO+State), and +12.9 pp (pi0-FF); all exploratory task-bootstrap CIs exclude zero.",
        width=115,
        size=8.9,
    )
    _bullet(
        fig,
        0.06,
        0.084,
        "Exact-task retrieval is not the whole story: DINO succeeds 10/10 on task 20 while every one of its 816 retrieved slots comes from task 21, showing useful cross-task transfer and a limitation of exact-task metrics.",
        width=115,
        size=8.9,
    )
    _footer(
        fig,
        2,
        "Statistical unit: task  •  10,000 bootstrap repetitions  •  exact paired McNemar tests with Holm correction",
    )
    return fig


def _phase_method(phase_summary: dict, condition: str) -> dict:
    return next(row for row in phase_summary["methods"] if row["condition"] == condition)


def _page_three(phase_summary: dict, phase_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(0.06, 0.965, "When are tasks retrieved?", fontsize=20, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(
        0.06,
        0.931,
        "A phase audit over 254,132 retrieved slots from all 1,000 rollouts",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )

    caveat = FancyBboxPatch(
        (0.06, 0.845),
        0.88,
        0.057,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        transform=fig.transFigure,
        fc="#FFF7E8",
        ec="#E9CF9A",
        lw=0.9,
    )
    fig.add_artist(caveat)
    fig.text(0.078, 0.885, "HOW PHASE IS DEFINED", fontsize=8.2, fontweight="bold", color="#8B6515", va="top")
    fig.text(
        0.078,
        0.866,
        _wrapped(
            "The run did not save query-state contact flags. Policy phases are therefore normalized rollout-time quartiles, not event-detector labels. 'Grasp' means contact/actuation for stove, drawer and microwave tasks. Retrieved demonstration phase is measured from the cached expert frame's exact ordinal position.",
            133,
        ),
        fontsize=8.35,
        color=COLORS["ink"],
        va="top",
        linespacing=1.2,
    )

    phase_names = (
        "Approach / setup",
        "Contact / grasp / actuate",
        "Transport / manipulate",
        "Place / release / finish",
    )
    phase_ranges = ("0-25%", "25-50%", "50-75%", "75-100%")
    phase_colors = ("#EAF2FB", "#EAF6EE", "#FFF1F3", "#F2EDFA")
    for index, (name, phase_range, color) in enumerate(zip(phase_names, phase_ranges, phase_colors, strict=True)):
        x = 0.06 + index * 0.223
        box = FancyBboxPatch(
            (x, 0.774),
            0.205,
            0.048,
            boxstyle="round,pad=0.005,rounding_size=0.005",
            transform=fig.transFigure,
            fc=color,
            ec=COLORS["rule"],
            lw=0.8,
        )
        fig.add_artist(box)
        fig.text(
            x + 0.1025,
            0.807,
            phase_range,
            fontsize=7.6,
            fontweight="bold",
            color=COLORS["muted"],
            ha="center",
            va="top",
        )
        fig.text(x + 0.1025, 0.789, name, fontsize=8.0, fontweight="bold", color=COLORS["ink"], ha="center", va="top")

    _section(fig, 0.749, "A concrete rollout example")
    example_axis = fig.add_axes([0.06, 0.568, 0.88, 0.16])
    example_axis.imshow(mpimg.imread(phase_dir / "figures" / "phase_example_t52.png"))
    example_axis.axis("off")
    fig.text(
        0.06,
        0.552,
        "DINO succeeds on T52 (milk → basket) while its dominant source moves T50 → T49 → T46 → T52: it first reuses other basket-placement tasks and becomes exact-task dominant only at placement.",
        fontsize=7.8,
        color=COLORS["muted"],
        va="top",
    )

    _section(fig, 0.515, "Correct-task retrieval through the rollout")
    exact_axis = fig.add_axes([0.055, 0.168, 0.89, 0.322])
    exact_axis.imshow(mpimg.imread(phase_dir / "figures" / "phase_exact_task.png"))
    exact_axis.axis("off")

    dino = _phase_method(phase_summary, "dino_global")
    dino_state = _phase_method(phase_summary, "dino_state_global")
    pi = _phase_method(phase_summary, "pi_global")
    finding = FancyBboxPatch(
        (0.06, 0.064),
        0.88,
        0.071,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        transform=fig.transFigure,
        fc=COLORS["panel"],
        ec=COLORS["rule"],
        lw=0.8,
    )
    fig.add_artist(finding)
    fig.text(
        0.078,
        0.118,
        "PHASE RESULT",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.078,
        0.099,
        _wrapped(
            f"Across all rollouts, exact-task retrieval rises from {100*dino['exact_task_fraction_by_phase'][0]:.1f}% to {100*dino['exact_task_fraction_by_phase'][3]:.1f}% for DINO and {100*dino_state['exact_task_fraction_by_phase'][0]:.1f}% to {100*dino_state['exact_task_fraction_by_phase'][3]:.1f}% for DINO+State, but falls from {100*pi['exact_task_fraction_by_phase'][0]:.1f}% to {100*pi['exact_task_fraction_by_phase'][3]:.1f}% for pi0-FF. Successful and failed trajectories separate strongly after approach; this is association, not proof that retrieval caused success.",
            133,
        ),
        fontsize=8.25,
        color=COLORS["ink"],
        va="top",
        linespacing=1.23,
    )
    _footer(fig, 3, f"Phase artifacts: {phase_dir}")
    return fig


def _page_four(phase_summary: dict, phase_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.06, 0.965, "Does retrieval follow task phase?", fontsize=20, fontweight="bold", color=COLORS["ink"], va="top"
    )
    fig.text(
        0.06,
        0.931,
        "Policy query quartile versus the exact temporal quartile of every retrieved expert frame",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.06,
        0.890,
        _wrapped(
            "Each row below sums to 100%. A phase-sensitive retriever should move mass from early demonstration frames toward late frames as the rollout progresses. Random stays near 25% in every cell and provides the calibration baseline.",
            125,
        ),
        fontsize=9.0,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )

    alignment_axis = fig.add_axes([0.055, 0.588, 0.89, 0.246])
    alignment_axis.imshow(mpimg.imread(phase_dir / "figures" / "phase_alignment.png"))
    alignment_axis.axis("off")

    _section(fig, 0.552, "Diagonal phase match")
    table_rows = []
    interpretations = {
        "dino_global": "Strong early and finish alignment",
        "dino_state_global": "Strongest finish alignment",
        "pi_global": "Strong early; diffuse later",
        "random_global": "Uniform by construction",
    }
    for condition in ("dino_global", "dino_state_global", "pi_global", "random_global"):
        method = _phase_method(phase_summary, condition)
        matrix = method["query_to_demo_phase_matrix"]
        table_rows.append(
            [
                method["method_label"],
                f"{100*matrix[0][0]:.1f}%",
                f"{100*matrix[1][1]:.1f}%",
                f"{100*matrix[2][2]:.1f}%",
                f"{100*matrix[3][3]:.1f}%",
                interpretations[condition],
            ]
        )
    ax = fig.add_axes([0.06, 0.354, 0.88, 0.17])
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Selector",
            "Approach\n→ early",
            "Contact\n→ contact",
            "Transport\n→ transport",
            "Finish\n→ finish",
            "Pattern",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.16, 0.13, 0.13, 0.14, 0.13, 0.31],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)  # noqa: FBT003 - matplotlib requires a positional flag.
    table.set_fontsize(8.3)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#F5F7FA" if row_idx % 2 else "#E9EEF4")
            if col_idx == 0:
                condition = ("dino_global", "dino_state_global", "pi_global", "random_global")[row_idx - 1]
                cell.get_text().set_color(COLORS[condition])
                cell.get_text().set_fontweight("bold")

    _section(fig, 0.319, "Interpretation")
    _bullet(
        fig,
        0.06,
        0.286,
        "All informed selectors recognize early expert frames during approach. DINO+State is most concentrated on late expert frames during finish (57.9%), consistent with state helping temporal phase matching.",
        width=113,
        size=9.0,
    )
    _bullet(
        fig,
        0.06,
        0.229,
        "DINO and DINO+State often retrieve finish frames already during the transport quartile (33.8% and 37.1%). The temporal boundary between carrying and placement is therefore not sharp in the expert demonstrations.",
        width=113,
        size=9.0,
    )
    _bullet(
        fig,
        0.06,
        0.172,
        "pi0-FF has the strongest approach and contact diagonal cells, but its finish queries are less concentrated on finish frames (39.0%). Even so, it has the highest numerical success rate—phase alignment alone is not sufficient to rank control performance.",
        width=113,
        size=9.0,
    )
    _bullet(
        fig,
        0.06,
        0.105,
        "These matrices measure temporal alignment, not action correctness: two tasks can be in the same phase while requiring different objects, receptacles, or goals.",
        width=113,
        size=9.0,
    )
    _footer(fig, 4, "Rows are equal-task/equal-scene averages; columns are cached expert-episode temporal quartiles")
    return fig


def _top_source_cell(method: dict, phase: int) -> str:
    tasks = method["top_source_tasks_by_phase"][phase]["tasks"][:2]
    return "\n".join(f"T{row['task_id']} {TASK_TINY[row['task_id']]} ({100*row['fraction']:.1f}%)" for row in tasks)


def _page_five(phase_summary: dict, phase_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.06,
        0.965,
        "Which source tasks are retrieved when?",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.06,
        0.931,
        "Task identities selected during approach, contact/grasp, transport and finish",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )

    heatmap_axis = fig.add_axes([0.09, 0.482, 0.82, 0.415])
    heatmap_axis.imshow(mpimg.imread(phase_dir / "figures" / "phase_source_task_heatmaps.png"))
    heatmap_axis.axis("off")

    _section(fig, 0.451, "Two most frequent source tasks per phase")
    methods = [
        _phase_method(phase_summary, condition)
        for condition in ("dino_global", "dino_state_global", "pi_global", "random_global")
    ]
    rows = [[method["method_label"], *[_top_source_cell(method, phase) for phase in range(4)]] for method in methods]
    ax = fig.add_axes([0.06, 0.167, 0.88, 0.252])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Selector", "Approach / setup", "Contact / grasp", "Transport", "Place / finish"],
        cellLoc="left",
        colLoc="center",
        colWidths=[0.14, 0.215, 0.215, 0.215, 0.215],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)  # noqa: FBT003 - matplotlib requires a positional flag.
    table.set_fontsize(7.25)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor("#F5F7FA" if row_idx % 2 else "#E9EEF4")
            if col_idx == 0:
                condition = ("dino_global", "dino_state_global", "pi_global", "random_global")[row_idx - 1]
                cell.get_text().set_color(COLORS[condition])
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_ha("center")

    fig.text(
        0.06,
        0.137,
        _wrapped(
            "DINO-family retrieval stays distributed across several semantically related task memories. pi0-FF increasingly concentrates on T61 (pudding → tray) and T21 (stove + pan) after approach, matching its decline in exact-task retrieval. Random's T21/T44 preference is not semantic: uniform frame sampling favors tasks with longer cached episodes.",
            131,
        ),
        fontsize=8.45,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )
    fig.text(
        0.06,
        0.077,
        "Percentages normalize each rollout-phase first, average the ten scenes within each target task, then weight all 25 target tasks equally.",
        fontsize=7.5,
        color=COLORS["muted"],
        va="top",
    )
    _footer(
        fig,
        5,
        "The full 25 source-task x 4 phase matrices are shown above; exact values are saved in phase_source_tasks.csv",
    )
    return fig


def _route_text(route: dict) -> str:
    target = int(route["target_task"])
    source = int(route["source_task"])
    return (
        f"T{target} {TASK_TINY[target]} → T{source} {TASK_TINY[source]}\n"
        f"{100*route['mean_scene_fraction']:.0f}% of target-phase slots"
    )


def _route_table(fig: plt.Figure, method: dict, y: float, height: float) -> None:
    fig.text(
        0.06,
        y + height + 0.013,
        method["method_label"],
        fontsize=11.2,
        fontweight="bold",
        color=COLORS[method["condition"]],
        va="bottom",
    )
    rows = []
    for phase_index, phase_row in enumerate(method["top_cross_task_routes_by_phase"]):
        routes = phase_row["routes"][:2]
        rows.append(
            [
                ("Approach", "Contact / grasp", "Transport", "Place / finish")[phase_index],
                _route_text(routes[0]),
                _route_text(routes[1]),
            ]
        )
    ax = fig.add_axes([0.06, y, 0.88, height])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Policy phase", "Dominant cross-task route", "Second route"],
        cellLoc="left",
        colLoc="center",
        colWidths=[0.16, 0.42, 0.42],
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)  # noqa: FBT003 - matplotlib requires a positional flag.
    table.set_fontsize(6.8)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row_idx == 0:
            cell.set_facecolor(COLORS["ink"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor("#F5F7FA" if row_idx % 2 else "#E9EEF4")
            if col_idx == 0:
                cell.get_text().set_fontweight("bold")


def _page_six(phase_summary: dict) -> plt.Figure:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    fig.text(
        0.06,
        0.965,
        "Dominant cross-task retrieval routes",
        fontsize=20,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.06,
        0.931,
        "For each target task and phase: which wrong-task memory accounts for the largest share?",
        fontsize=9.4,
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.06,
        0.890,
        _wrapped(
            "Routes below exclude exact-task retrieval. Percentages are the mean fraction of that target task's phase-specific selections across its ten frozen scenes; they expose systematic semantic transfer and systematic confusion that global averages hide.",
            126,
        ),
        fontsize=8.9,
        color=COLORS["ink"],
        va="top",
        linespacing=1.25,
    )

    _route_table(fig, _phase_method(phase_summary, "dino_global"), 0.635, 0.175)
    _route_table(fig, _phase_method(phase_summary, "dino_state_global"), 0.405, 0.175)
    _route_table(fig, _phase_method(phase_summary, "pi_global"), 0.175, 0.175)

    note = FancyBboxPatch(
        (0.06, 0.066),
        0.88,
        0.076,
        boxstyle="round,pad=0.008,rounding_size=0.006",
        transform=fig.transFigure,
        fc=COLORS["panel"],
        ec=COLORS["rule"],
        lw=0.8,
    )
    fig.add_artist(note)
    fig.text(0.078, 0.128, "HOW TO READ THESE ROUTES", fontsize=8.2, fontweight="bold", color=COLORS["muted"], va="top")
    fig.text(
        0.078,
        0.108,
        _wrapped(
            "Some transfer is useful: T20 retrieves T21 for 100% of DINO-family slots in every phase and still succeeds 10/10 under DINO. Other routes preserve phase but change the goal, such as T24 bottom-drawer placement → T25 cabinet-top placement. pi0-FF's repeated later-phase routing into T61 explains part of its exact-task drop. Random has no route above 12%, as expected.",
            132,
        ),
        fontsize=8.15,
        color=COLORS["ink"],
        va="top",
        linespacing=1.2,
    )
    _footer(
        fig,
        6,
        "Cross-task transfer can be useful or harmful; exact-task accuracy and temporal-phase match must be interpreted together",
    )
    return fig


def main() -> None:
    args = _parse_args()
    summary_path = args.analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("analysis_status") != "final_complete":
        raise SystemExit(f"Refusing to publish non-final analysis: {summary.get('analysis_status')}")
    for row in summary["closed_loop"]:
        if not row.get("complete") or row.get("valid_result_rollouts") != 250 or row.get("valid_trace_rollouts") != 250:
            raise SystemExit(f"Incomplete condition: {row['condition']}")
    if sum(row.get("infrastructure_error_rows", 0) for row in summary["closed_loop"]):
        raise SystemExit("Infrastructure errors are present; investigate before publishing")
    phase_summary_path = args.phase_dir / "summary.json"
    phase_summary = json.loads(phase_summary_path.read_text())
    if phase_summary.get("analysis_status") != "final_complete":
        raise SystemExit(f"Refusing to publish non-final phase analysis: {phase_summary.get('analysis_status')}")
    phase_validation = phase_summary.get("validation", {})
    if phase_validation.get("total_rollouts") != 1000 or phase_validation.get("unmatched_retrieved_frames") != 0:
        raise SystemExit(f"Phase analysis validation failed: {phase_validation}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Global Retrieval in Closed-Loop LIBERO-90 — Final Run Report",
        "Author": "RICL/OpenPI experiment pipeline",
        "Subject": "Final pooled and phase-resolved analysis of 1,000 retrieval-conditioned closed-loop rollouts",
        "Keywords": "LIBERO-90, retrieval, DINOv2, pi0-FAST, closed-loop control",
    }
    with PdfPages(args.output, metadata=metadata) as pdf:
        page_one = _page_one(summary, args.analysis_dir)
        pdf.savefig(page_one, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_one)
        page_two = _page_two(summary, args.analysis_dir)
        pdf.savefig(page_two, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_two)
        page_three = _page_three(phase_summary, args.phase_dir)
        pdf.savefig(page_three, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_three)
        page_four = _page_four(phase_summary, args.phase_dir)
        pdf.savefig(page_four, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_four)
        page_five = _page_five(phase_summary, args.phase_dir)
        pdf.savefig(page_five, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_five)
        page_six = _page_six(phase_summary)
        pdf.savefig(page_six, bbox_inches="tight", pad_inches=0.05)
        plt.close(page_six)
    print(args.output)


if __name__ == "__main__":
    main()
