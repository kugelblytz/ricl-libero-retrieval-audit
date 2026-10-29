from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape


STYLE_DEFAULT = 0
STYLE_WINNER = 1
STYLE_LOSER = 2
STYLE_TIE = 3
STYLE_SKIPPED = 4


@dataclass(frozen=True)
class StyledCell:
    value: object
    style: int = STYLE_DEFAULT


RUN_DEFS = (
    {
        "key": "nointerp",
        "label": "RICL20 no interpolation",
        "task_range": "20-69 solvable subset",
        "protocol": "RICL, 20 retrieval demos/task, action interpolation disabled",
        "kind": "results",
    },
    {
        "key": "first10_ricl20",
        "label": "20-29 RICL20 interpolated",
        "task_range": "20-29",
        "protocol": "Existing RICL20 baseline, interpolation enabled, 10 held-out scenes/task",
        "kind": "ricl_trace",
    },
    {
        "key": "first10_ricl40",
        "label": "20-29 RICL40 interpolated",
        "task_range": "20-29",
        "protocol": "Existing RICL40 baseline, interpolation enabled, 10 held-out scenes/task",
        "kind": "ricl_trace",
    },
    {
        "key": "first10_normal",
        "label": "20-29 Normal reference",
        "task_range": "20-29",
        "protocol": "Existing normal baseline; leftout10_spread40 scene-selection reference",
        "kind": "results",
    },
    {
        "key": "later_ricl20",
        "label": "30-69 RICL20 interpolated",
        "task_range": "30-69",
        "protocol": "Existing RICL20 baseline, interpolation enabled, 10 held-out scenes/task",
        "kind": "results",
    },
    {
        "key": "later_normal",
        "label": "30-69 Normal",
        "task_range": "30-69",
        "protocol": "Existing normal baseline, no retrieval context, same selected scene setups",
        "kind": "results",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO-90 RICL20 no-interpolation solvable-task comparison.")
    parser.add_argument(
        "--nointerp-results",
        type=Path,
        default=Path("data/libero/videos/libero90_20to69_ricl20_nointerp_solvable_seeded20_20260719/results.jsonl"),
    )
    parser.add_argument("--first10-ricl20", type=Path, default=Path("data/libero/videos/libero90_leftout10/retrieval_trace.jsonl"))
    parser.add_argument(
        "--first10-ricl40",
        type=Path,
        default=Path("data/libero/videos/libero90_leftout10_spread40_20260701_133248/retrieval_trace.jsonl"),
    )
    parser.add_argument(
        "--first10-normal",
        type=Path,
        default=Path("data/libero/videos/libero90_normal_leftout10_spread40_cuda_fp32_20260712_112443/results.jsonl"),
    )
    parser.add_argument(
        "--later-ricl20",
        type=Path,
        default=Path("data/libero/videos/libero90_30to69_ricl20_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"),
    )
    parser.add_argument(
        "--later-normal",
        type=Path,
        default=Path("data/libero/videos/libero90_30to69_normal_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/libero/reports/libero90_20to69_ricl20_nointerp_solvable_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        ),
    )
    parser.add_argument(
        "--included-task-ids",
        default="20,21,24,25,28,29,30,31,34,35,43,44,46,49,50,51,52,55,56,57,59,60,61,68,69",
    )
    parser.add_argument(
        "--excluded-task-ids",
        default="22,23,26,27,32,33,36,37,38,39,40,41,42,45,47,48,53,54,58,62,63,64,65,66,67",
    )
    args = parser.parse_args()

    paths = {
        "nointerp": args.nointerp_results,
        "first10_ricl20": args.first10_ricl20,
        "first10_ricl40": args.first10_ricl40,
        "first10_normal": args.first10_normal,
        "later_ricl20": args.later_ricl20,
        "later_normal": args.later_normal,
    }
    included = parse_task_ids(args.included_task_ids)
    excluded = parse_task_ids(args.excluded_task_ids)
    exported_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    runs = [load_run(run_def, paths[run_def["key"]]) for run_def in RUN_DEFS]

    workbook = {
        "Summary": summary_sheet(runs, included, excluded, exported_at),
        "PerTask": per_task_sheet(runs, included, excluded),
        "NoInterpDetails": nointerp_details_sheet(run_by_key(runs, "nointerp"), included),
        "BaselineSources": baseline_sources_sheet(runs),
        "Artifacts": artifacts_sheet(args, exported_at),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output, workbook)
    print(f"Wrote {args.output}")
    for run in runs:
        print(f"{run['label']}: {run['successes']}/{run['rollouts']} ({percent(run['successes'], run['rollouts'])}) path={run['path']}")


def parse_task_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def load_run(run_def: dict, path: Path) -> dict:
    if not path.exists():
        return {
            **run_def,
            "path": str(path),
            "missing": True,
            "rollouts": 0,
            "successes": 0,
            "per_task": {},
            "rows": [],
        }
    rows = load_ricl_trace(path) if run_def["kind"] == "ricl_trace" else load_results(path)
    per_task: dict[int, dict] = defaultdict(empty_task)
    for row in rows:
        task_id = int(row["task_id"])
        stats = per_task[task_id]
        stats["prompt"] = stats["prompt"] or row.get("prompt", "")
        stats["rollouts"] += 1
        stats["successes"] += int(bool(row.get("success")))
        stats["elapsed_steps"].append(row.get("elapsed_steps", ""))
    return {
        **run_def,
        "path": str(path),
        "missing": False,
        "rollouts": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
        "per_task": dict(per_task),
        "rows": rows,
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
                    "trial_id": row.get("trial_id", row.get("rollout_idx", "")),
                    "success": row["success"],
                    "elapsed_steps": row.get("final_timestep", ""),
                    "prompt": row.get("task_description", ""),
                    "video_path": row.get("video_path", ""),
                }
            )
    return records


def load_results(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {
            "task_id": row["task_id"],
            "scene_setup_id": row["scene_setup_id"],
            "trial_id": row.get("trial_id", ""),
            "success": row["success"],
            "elapsed_steps": row.get("elapsed_steps", ""),
            "prompt": row.get("task_language", ""),
            "video_path": row.get("video_path", ""),
            "retrieval_mode": row.get("retrieval_mode", ""),
            "error": row.get("error", ""),
        }
        for row in rows
    ]


def empty_task() -> dict:
    return {"prompt": "", "rollouts": 0, "successes": 0, "elapsed_steps": []}


def summary_sheet(runs: list[dict], included: list[int], excluded: list[int], exported_at: str) -> list[list[object]]:
    rows = [
        ["Exported at", exported_at],
        ["Included task count", len(included)],
        ["Excluded task count", len(excluded)],
        ["Expected no-interp rollouts", len(included) * 10],
        [],
        ["Run", "Task range", "Successes", "Rollouts", "Success rate", "Protocol", "Artifact"],
    ]
    for run in runs:
        rows.append(
            [
                run["label"],
                run["task_range"],
                run["successes"],
                run["rollouts"],
                percent(run["successes"], run["rollouts"]),
                run["protocol"],
                run["path"],
            ]
        )
    rows.extend(
        [
            [],
            ["Included tasks", ",".join(str(task_id) for task_id in included)],
            ["Excluded zero-baseline tasks", ",".join(str(task_id) for task_id in excluded)],
            ["Color legend", "Green = best task success count, red = worst, yellow = tie among available runs, gray = intentionally skipped"],
        ]
    )
    return rows


def per_task_sheet(runs: list[dict], included: list[int], excluded: list[int]) -> list[list[object]]:
    rows = [["Task", "Status", "Prompt", *[run["label"] for run in runs], "Best run(s)"]]
    included_set = set(included)
    excluded_set = set(excluded)
    for task_id in range(20, 70):
        prompt = first_prompt(runs, task_id)
        stats = [run["per_task"].get(task_id) for run in runs]
        styles = task_styles(stats)
        status = "INCLUDED" if task_id in included_set else "EXCLUDED_ZERO_BASELINE" if task_id in excluded_set else "OUT_OF_SCOPE"
        status_cell = StyledCell(status, STYLE_SKIPPED if status != "INCLUDED" else STYLE_DEFAULT)
        row = [task_id, status_cell, prompt]
        for run, task_stats, style in zip(runs, stats, styles, strict=True):
            if task_id not in included_set and run["key"] == "nointerp":
                row.append(StyledCell("", STYLE_SKIPPED))
            elif task_stats is None:
                row.append("")
            else:
                row.append(StyledCell(ratio_text(task_stats), style))
        row.append(best_runs(runs, stats))
        rows.append(row)
    return rows


def nointerp_details_sheet(nointerp: dict, included: list[int]) -> list[list[object]]:
    rows = [["Task", "Scene setup", "Trial", "Prompt", "Success", "Elapsed steps", "Retrieval mode", "Error", "Video"]]
    included_set = set(included)
    for row in sorted(nointerp["rows"], key=lambda item: (int(item["task_id"]), int(item["scene_setup_id"]), int(item.get("trial_id") or 0))):
        if int(row["task_id"]) not in included_set:
            continue
        rows.append(
            [
                row["task_id"],
                row["scene_setup_id"],
                row.get("trial_id", ""),
                row.get("prompt", ""),
                "TRUE" if row.get("success") else "FALSE",
                row.get("elapsed_steps", ""),
                row.get("retrieval_mode", ""),
                row.get("error", ""),
                row.get("video_path", ""),
            ]
        )
    return rows


def baseline_sources_sheet(runs: list[dict]) -> list[list[object]]:
    rows = [["Run", "Task range", "Rollouts", "Successes", "Notes", "Artifact"]]
    for run in runs:
        rows.append([run["label"], run["task_range"], run["rollouts"], run["successes"], run["protocol"], run["path"]])
    return rows


def artifacts_sheet(args: argparse.Namespace, exported_at: str) -> list[list[object]]:
    return [
        ["Key", "Value"],
        ["Exported at", exported_at],
        ["No-interp results", str(args.nointerp_results)],
        ["Output XLSX", str(args.output)],
        ["20-29 RICL20 baseline", str(args.first10_ricl20)],
        ["20-29 RICL40 baseline", str(args.first10_ricl40)],
        ["20-29 normal reference", str(args.first10_normal)],
        ["30-69 RICL20 baseline", str(args.later_ricl20)],
        ["30-69 normal baseline", str(args.later_normal)],
        ["Normal 20-29 note", "leftout10_spread40 reference; scene IDs differ from RICL20 spread20 first-10 run."],
    ]


def run_by_key(runs: list[dict], key: str) -> dict:
    return next(run for run in runs if run["key"] == key)


def first_prompt(runs: list[dict], task_id: int) -> str:
    for run in runs:
        stats = run["per_task"].get(task_id)
        if stats and stats.get("prompt"):
            return stats["prompt"]
    return ""


def task_styles(stats: list[dict | None]) -> list[int]:
    values = [None if item is None or item["rollouts"] == 0 else item["successes"] for item in stats]
    present_values = [value for value in values if value is not None]
    if not present_values:
        return [STYLE_DEFAULT] * len(stats)
    high = max(present_values)
    low = min(present_values)
    if high == low:
        return [STYLE_TIE if value is not None else STYLE_DEFAULT for value in values]
    styles = []
    for value in values:
        if value is None:
            styles.append(STYLE_DEFAULT)
        elif value == high:
            styles.append(STYLE_WINNER)
        elif value == low:
            styles.append(STYLE_LOSER)
        else:
            styles.append(STYLE_DEFAULT)
    return styles


def best_runs(runs: list[dict], stats: list[dict | None]) -> str:
    candidates = [(run["label"], item["successes"]) for run, item in zip(runs, stats, strict=True) if item and item["rollouts"]]
    if not candidates:
        return ""
    best = max(value for _, value in candidates)
    return ", ".join(label for label, value in candidates if value == best)


def ratio_text(stats: dict) -> str:
    return f"{stats['successes']}/{stats['rollouts']}"


def percent(successes: int, rollouts: int) -> str:
    return "" if rollouts == 0 else f"{100.0 * successes / rollouts:.1f}%"


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
<fills count="6">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFFFC7CE"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFFFEB9C"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFD9E1F2"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="5">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/>
<xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1"/>
<xf numFmtId="0" fontId="0" fillId="4" borderId="0" xfId="0" applyFill="1"/>
<xf numFmtId="0" fontId="0" fillId="5" borderId="0" xfId="0" applyFill="1"/>
</cellXfs>
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
    style = STYLE_DEFAULT
    if isinstance(value, StyledCell):
        style = value.style
        value = value.value
    style_attr = "" if style == STYLE_DEFAULT else f' s="{style}"'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}"{style_attr} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def column_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


if __name__ == "__main__":
    main()
