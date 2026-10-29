from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).with_name("summarize_libero_pi_vs_dino.py")
SPEC = importlib.util.spec_from_file_location("summarize_libero_pi_vs_dino", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_module
SPEC.loader.exec_module(summary_module)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summary_uses_latest_error_free_paired_trials(tmp_path: Path) -> None:
    conditions = ["dino_global", "dino_state_global", "pi_global", "random_global"]
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"conditions": conditions, "expected_trials_per_condition": 1})
    )
    base = {
        "task_id": 21,
        "trial_id": 0,
        "scene_setup_id": 3,
        "elapsed_steps": 100,
        "timings_ms": {"request_total": {"mean": 10.0}},
    }
    method_rows = {
        "dino_global": [{**base, "success": False, "error": None}],
        "dino_state_global": [
            {**base, "success": False, "error": "disconnect"},
            {**base, "success": True, "error": None},
        ],
        "pi_global": [{**base, "success": False, "error": None}],
        "random_global": [{**base, "success": False, "error": None}],
    }
    trace = [
        {
            **base,
            "success": True,
            "error": None,
            "retrieval_trace": [
                {"rank": 0, "retrieved_task_id": 24},
                {"rank": 1, "retrieved_task_id": 21},
            ],
        }
    ]
    for condition in conditions:
        _write_rows(tmp_path / condition / "results.jsonl", method_rows[condition])
        _write_rows(tmp_path / condition / "retrieval_trace.jsonl", trace)

    summary = summary_module.summarize(tmp_path, bootstrap_repetitions=100)

    assert summary["conditions"][1]["valid_trials"] == 1
    assert summary["conditions"][1]["infrastructure_error_rows"] == 1
    assert summary["conditions"][1]["cross_task_retrieval_rate"] == 0.5
    assert summary["paired_comparisons"][0]["left_only_successes"] == 1
    assert summary["paired_comparisons"][0]["mcnemar_exact_p"] == 1.0
    assert len(summary["paired_comparisons"]) == 5
    assert all("holm_adjusted_p" in row for row in summary["paired_comparisons"])
    assert len(summary["per_task"]) == 4

    offline_path = tmp_path / "offline_per_task.csv"
    offline_path.write_text(
        "method,task_id,task_aware_ndcg_at4\n"
        "dino_top_global,21,0.7\n"
        "dino_top_global,22,0.5\n"
        "dino_top_state_global,21,0.8\n"
        "dino_top_state_global,22,0.6\n"
        "pi0_fast_first_fusion_global,21,0.6\n"
        "pi0_fast_first_fusion_global,22,0.4\n"
    )
    summary_module._write_final_analysis(tmp_path, summary, offline_path, 10, 7)  # noqa: SLF001

    assert (tmp_path / "offline_ndcg_summary.csv").is_file()
    assert (tmp_path / "offline_online_correlations.csv").is_file()
    assert (tmp_path / "figures" / "global_offline_ndcg.png").is_file()
    assert (tmp_path / "figures" / "closed_loop_success.png").is_file()
    assert (tmp_path / "figures" / "per_task_closed_loop_heatmap.png").is_file()
    assert (tmp_path / "figures" / "offline_vs_online_paired_delta.png").is_file()
    assert (tmp_path / "figures" / "correct_task_retrieval_vs_success.png").is_file()
    offline_summary = list(csv.DictReader((tmp_path / "offline_ndcg_summary.csv").open()))
    assert {int(row["task_count"]) for row in offline_summary} == {2}
    assert float(offline_summary[0]["mean_task_aware_ndcg_at4"]) == 0.6
