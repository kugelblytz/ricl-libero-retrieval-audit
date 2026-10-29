from __future__ import annotations

from collections import defaultdict
import argparse
import datetime as dt
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape


RUNS = (
    {
        "key": "first10_ricl20",
        "label": "First10 RICL20 leftout10",
        "task_range": "20-29",
        "protocol": "RICL matched, 20 retrieval demos/task, 10 held-out scenes/task",
        "path": Path("data/libero/videos/libero90_leftout10/retrieval_trace.jsonl"),
        "kind": "ricl_trace",
        "expected": 100,
        "status": "FINAL",
    },
    {
        "key": "first10_ricl40",
        "label": "First10 RICL40 leftout10",
        "task_range": "20-29",
        "protocol": "RICL matched, 40 retrieval demos/task, 10 held-out scenes/task",
        "path": Path("data/libero/videos/libero90_leftout10_spread40_20260701_133248/retrieval_trace.jsonl"),
        "kind": "ricl_trace",
        "expected": 100,
        "status": "FINAL",
    },
    {
        "key": "first10_normal",
        "label": "First10 Normal leftout10",
        "task_range": "20-29",
        "protocol": "Normal Pi0-FAST, no retrieval context; matched held-out scene selection",
        "path": Path("data/libero/videos/libero90_normal_leftout10_spread40_cuda_fp32_20260712_112443/results.jsonl"),
        "kind": "results",
        "expected": 100,
        "status": "FINAL",
    },
    {
        "key": "first10_ricl49_diag",
        "label": "First10 RICL49 leftout1 diagnostic",
        "task_range": "20-29",
        "protocol": "RICL diagnostic only: 49 retrieval demos/task, 1 held-out scene/task",
        "path": Path("data/libero/videos/libero90_leftout1_spread49_20260701_133248/retrieval_trace.jsonl"),
        "kind": "ricl_trace",
        "expected": 10,
        "status": "DIAGNOSTIC",
    },
    {
        "key": "current30_ricl20",
        "label": "Current30to69 RICL20",
        "task_range": "30-69",
        "protocol": "RICL matched, 20 retrieval demos/task, 10 held-out scenes/task",
        "path": Path("data/libero/videos/libero90_30to69_ricl20_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"),
        "kind": "results",
        "expected": 400,
        "status": "FINAL",
    },
    {
        "key": "current30_normal_partial",
        "label": "Current30to69 Normal",
        "task_range": "30-69",
        "protocol": "Normal Pi0-FAST, no retrieval context; same selected scene setups",
        "path": Path("data/libero/videos/libero90_30to69_normal_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"),
        "kind": "results",
        "expected": 400,
        "status": "PARTIAL",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export preliminary LIBERO-90 RICL-vs-normal overview to XLSX.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/libero/reports/libero90_pre_overview_ricl20_vs_normal_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        ),
    )
    args = parser.parse_args()

    exported_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    runs = [load_run(run) for run in RUNS]

    workbook = {
        "Summary": summary_rows(runs, exported_at),
        "PerTask": per_task_rows(runs),
        "Current30to69": current_rows(runs),
        "First10_20to29": first10_rows(runs),
        "Artifacts": artifacts_rows(runs, args.output, exported_at),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output, workbook)
    print(f"Wrote {args.output}")
    for run in runs:
        print(
            f"{run['label']}: {run['successes']}/{run['rollouts']} "
            f"({rate_text(run)}) status={run['status']} path={run['path']}"
        )


def load_run(config: dict) -> dict:
    path = config["path"]
    if not path.exists():
        return {
            **config,
            "path": str(path),
            "rollouts": 0,
            "successes": 0,
            "rate": None,
            "per_task": {},
            "per_scene": {},
            "missing": True,
        }

    if config["kind"] == "ricl_trace":
        records = load_ricl_trace(path)
    else:
        records = load_results(path)

    per_task: dict[int, dict] = defaultdict(empty_task)
    per_scene = {}
    for row in records:
        task_id = int(row["task_id"])
        scene_id = int(row["scene_setup_id"])
        success = bool(row["success"])
        stats = per_task[task_id]
        stats["prompt"] = stats["prompt"] or row.get("prompt", "")
        stats["rollouts"] += 1
        stats["successes"] += int(success)
        per_scene[(task_id, scene_id)] = {
            "success": success,
            "elapsed_steps": row.get("elapsed_steps", ""),
            "prompt": row.get("prompt", ""),
        }

    rollouts = sum(stats["rollouts"] for stats in per_task.values())
    successes = sum(stats["successes"] for stats in per_task.values())
    status = config["status"]
    if rollouts and rollouts < config["expected"] and status == "FINAL":
        status = "PARTIAL"
    return {
        **config,
        "path": str(path),
        "rollouts": rollouts,
        "successes": successes,
        "rate": None if rollouts == 0 else successes / rollouts,
        "per_task": dict(per_task),
        "per_scene": per_scene,
        "status": status,
        "missing": False,
    }


def load_ricl_trace(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event_type") != "rollout_summary":
                continue
            records.append(
                {
                    "task_id": row["task_id"],
                    "scene_setup_id": row.get("initial_state_index", row.get("scene_setup_number", "")),
                    "success": row["success"],
                    "elapsed_steps": row.get("final_timestep", ""),
                    "prompt": row.get("task_description", ""),
                }
            )
    return records


def load_results(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {
            "task_id": row["task_id"],
            "scene_setup_id": row["scene_setup_id"],
            "success": row["success"],
            "elapsed_steps": row.get("elapsed_steps", ""),
            "prompt": row.get("task_language", ""),
        }
        for row in rows
    ]


def summary_rows(runs: list[dict], exported_at: str) -> list[list[object]]:
    rows = [
        ["Exported at", exported_at, "", "", "", "", "", "", ""],
        ["Run", "Task range", "Status", "Successes", "Rollouts", "Expected rollouts", "Success rate", "Protocol", "Artifact"],
    ]
    for run in runs:
        rows.append(
            [
                run["label"],
                run["task_range"],
                run["status"],
                run["successes"],
                run["rollouts"],
                run["expected"],
                rate_text(run),
                run["protocol"],
                run["path"],
            ]
        )
    return rows


def per_task_rows(runs: list[dict]) -> list[list[object]]:
    rows = [["Task", "Prompt", *[run["label"] for run in runs]]]
    for task_id in range(20, 70):
        prompt = first_prompt(runs, task_id)
        row = [task_id, prompt]
        for run in runs:
            stats = run["per_task"].get(task_id)
            row.append("" if stats is None else ratio_text(stats))
        rows.append(row)
    return rows


def current_rows(runs: list[dict]) -> list[list[object]]:
    ricl = run_by_key(runs, "current30_ricl20")
    normal = run_by_key(runs, "current30_normal_partial")
    rows = [["Task", "Prompt", "RICL20", "Normal", "Normal status", "Delta normal-ricl"]]
    for task_id in range(30, 70):
        ricl_stats = ricl["per_task"].get(task_id, empty_task())
        normal_stats = normal["per_task"].get(task_id, empty_task())
        prompt = normal_stats["prompt"] or ricl_stats["prompt"]
        normal_status = "PARTIAL" if normal_stats["rollouts"] < 10 else "COMPLETE"
        rows.append(
            [
                task_id,
                prompt,
                ratio_text(ricl_stats),
                ratio_text(normal_stats),
                normal_status,
                normal_stats["successes"] - ricl_stats["successes"],
            ]
        )
    return rows


def first10_rows(runs: list[dict]) -> list[list[object]]:
    ordered = [
        run_by_key(runs, "first10_ricl20"),
        run_by_key(runs, "first10_ricl40"),
        run_by_key(runs, "first10_normal"),
        run_by_key(runs, "first10_ricl49_diag"),
    ]
    rows = [["Task", "Prompt", *[run["label"] for run in ordered]]]
    for task_id in range(20, 30):
        row = [task_id, first_prompt(ordered, task_id)]
        for run in ordered:
            stats = run["per_task"].get(task_id)
            row.append("" if stats is None else ratio_text(stats))
        rows.append(row)
    return rows


def artifacts_rows(runs: list[dict], output: Path, exported_at: str) -> list[list[object]]:
    rows = [
        ["Key", "Value"],
        ["Exported at", exported_at],
        ["Output XLSX", str(output)],
        ["Note", "PRELIMINARY: current30to69 normal run is still in progress unless status shows FINAL."],
    ]
    for run in runs:
        rows.append([run["label"], run["path"]])
    return rows


def run_by_key(runs: list[dict], key: str) -> dict:
    return next(run for run in runs if run["key"] == key)


def first_prompt(runs: list[dict], task_id: int) -> str:
    for run in runs:
        stats = run["per_task"].get(task_id)
        if stats and stats["prompt"]:
            return stats["prompt"]
    return ""


def empty_task() -> dict:
    return {"prompt": "", "successes": 0, "rollouts": 0}


def ratio_text(stats: dict) -> str:
    return f"{stats['successes']}/{stats['rollouts']}"


def rate_text(run: dict) -> str:
    return "" if run["rate"] is None else f"{100.0 * run['rate']:.1f}%"


def write_xlsx(path: Path, sheets: dict[str, list[list[object]]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("xl/workbook.xml", workbook_xml(list(sheets)))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        zf.writestr("xl/styles.xml", styles_xml())
        for idx, rows in enumerate(sheets.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def content_types_xml(sheet_count: int) -> str:
    sheet_overrides = "\n".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
{sheet_overrides}
</Types>'''


def root_rels_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = "\n".join(
        f'<sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>{sheets}</sheets>
</workbook>'''


def workbook_rels_xml(sheet_count: int) -> str:
    relationships = "\n".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, sheet_count + 1)
    )
    relationships += (
        f'\n<Relationship Id="rId{sheet_count + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{relationships}
</Relationships>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="1"><fill><patternFill patternType="none"/></fill></fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>'''


def worksheet_xml(rows: list[list[object]]) -> str:
    row_xml = []
    for row_idx, row in enumerate(rows, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{column_name(col_idx)}{row_idx}"
            cells.append(cell_xml(ref, value))
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetData>{"".join(row_xml)}</sheetData>
</worksheet>'''


def cell_xml(ref: str, value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


if __name__ == "__main__":
    main()
