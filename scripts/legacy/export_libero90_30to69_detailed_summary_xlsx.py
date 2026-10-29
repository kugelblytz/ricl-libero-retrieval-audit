from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape


DEFAULT_RICL_RESULTS = Path(
    "data/libero/videos/libero90_30to69_ricl20_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"
)
DEFAULT_NORMAL_RESULTS = Path(
    "data/libero/videos/libero90_30to69_normal_leftout10_ricl20_cachefix_20260713_232910/results.jsonl"
)

STYLE_DEFAULT = 0
STYLE_WINNER = 1
STYLE_LOSER = 2
STYLE_TIE = 3


@dataclass(frozen=True)
class StyledCell:
    value: object
    style: int = STYLE_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a detailed LIBERO-90 30-69 RICL20 vs normal XLSX summary.")
    parser.add_argument("--ricl-results", type=Path, default=DEFAULT_RICL_RESULTS)
    parser.add_argument("--normal-results", type=Path, default=DEFAULT_NORMAL_RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/libero/reports/libero90_30to69_ricl20_vs_normal_detailed_"
            f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        ),
    )
    args = parser.parse_args()

    ricl_rows = load_rows(args.ricl_results)
    normal_rows = load_rows(args.normal_results)
    paired = pair_rows(ricl_rows, normal_rows)
    exported_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    workbook = {
        "Summary": summary_sheet(ricl_rows, normal_rows, paired, args, exported_at),
        "PerTask": per_task_sheet(paired),
        "RolloutPairs": rollout_pairs_sheet(paired),
        "OutcomeMatrix": outcome_matrix_sheet(paired),
        "Artifacts": artifacts_sheet(args, exported_at),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(args.output, workbook)
    print(f"Wrote {args.output}")
    print_summary(ricl_rows, normal_rows, paired)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda row: (int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])))


def pair_rows(ricl_rows: list[dict], normal_rows: list[dict]) -> list[dict]:
    ricl_by_key = {(int(row["task_id"]), int(row["scene_setup_id"])): row for row in ricl_rows}
    normal_by_key = {(int(row["task_id"]), int(row["scene_setup_id"])): row for row in normal_rows}
    keys = sorted(set(ricl_by_key) | set(normal_by_key))
    pairs = []
    for task_id, scene_id in keys:
        ricl = ricl_by_key.get((task_id, scene_id))
        normal = normal_by_key.get((task_id, scene_id))
        prompt = ""
        if normal is not None:
            prompt = normal.get("task_language", "")
        if not prompt and ricl is not None:
            prompt = ricl.get("task_language", "")
        pairs.append(
            {
                "task_id": task_id,
                "scene_setup_id": scene_id,
                "prompt": prompt,
                "ricl": ricl,
                "normal": normal,
            }
        )
    return pairs


def summary_sheet(ricl_rows: list[dict], normal_rows: list[dict], paired: list[dict], args: argparse.Namespace, exported_at: str):
    ricl_successes = count_successes(ricl_rows)
    normal_successes = count_successes(normal_rows)
    both_success = sum(1 for pair in paired if success(pair["ricl"]) and success(pair["normal"]))
    ricl_only = sum(1 for pair in paired if success(pair["ricl"]) and not success(pair["normal"]))
    normal_only = sum(1 for pair in paired if not success(pair["ricl"]) and success(pair["normal"]))
    both_fail = sum(1 for pair in paired if present(pair["ricl"]) and present(pair["normal"]) and not success(pair["ricl"]) and not success(pair["normal"]))

    return [
        ["Exported at", exported_at],
        ["Task range", "30-69"],
        ["Matched rollout pairs", len([p for p in paired if present(p["ricl"]) and present(p["normal"])])],
        [],
        ["Metric", "RICL20", "Normal", "Delta RICL-Normal"],
        ["Rollouts", len(ricl_rows), len(normal_rows), len(ricl_rows) - len(normal_rows)],
        ["Successes", ricl_successes, normal_successes, ricl_successes - normal_successes],
        ["Success rate", percent(ricl_successes, len(ricl_rows)), percent(normal_successes, len(normal_rows)), pct_points(ricl_successes, len(ricl_rows), normal_successes, len(normal_rows))],
        ["Timeout-ish rollouts >=400 steps", count_timeouts(ricl_rows), count_timeouts(normal_rows), count_timeouts(ricl_rows) - count_timeouts(normal_rows)],
        ["Mean elapsed steps", mean_steps(ricl_rows), mean_steps(normal_rows), mean_steps(ricl_rows) - mean_steps(normal_rows)],
        [],
        ["Paired outcome", "Count"],
        ["Both succeeded", both_success],
        ["RICL succeeded, normal failed", ricl_only],
        ["Normal succeeded, RICL failed", normal_only],
        ["Both failed", both_fail],
        [],
        ["RICL results", str(args.ricl_results)],
        ["Normal results", str(args.normal_results)],
        ["Notes", "RICL20 uses 20 same-task retrieval demos/task; normal uses no retrieval context. Both use the same task IDs and scene setup selection."],
    ]


def per_task_sheet(paired: list[dict]) -> list[list[object]]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for pair in paired:
        by_task[pair["task_id"]].append(pair)

    rows = [
        [
            "Task",
            "Prompt",
            "RICL successes",
            "RICL rollouts",
            "RICL rate",
            "Normal successes",
            "Normal rollouts",
            "Normal rate",
            "Delta successes RICL-Normal",
            "Both success",
            "RICL only",
            "Normal only",
            "Both fail",
            "RICL mean steps",
            "Normal mean steps",
        ]
    ]
    for task_id in range(30, 70):
        task_pairs = by_task.get(task_id, [])
        prompt = next((pair["prompt"] for pair in task_pairs if pair["prompt"]), "")
        ricl = [pair["ricl"] for pair in task_pairs if present(pair["ricl"])]
        normal = [pair["normal"] for pair in task_pairs if present(pair["normal"])]
        ricl_successes = count_successes(ricl)
        normal_successes = count_successes(normal)
        ricl_style, normal_style, delta_style = comparison_styles(ricl_successes, normal_successes)
        rows.append(
            [
                task_id,
                prompt,
                StyledCell(ricl_successes, ricl_style),
                len(ricl),
                StyledCell(percent(ricl_successes, len(ricl)), ricl_style),
                StyledCell(normal_successes, normal_style),
                len(normal),
                StyledCell(percent(normal_successes, len(normal)), normal_style),
                StyledCell(ricl_successes - normal_successes, delta_style),
                sum(1 for pair in task_pairs if success(pair["ricl"]) and success(pair["normal"])),
                sum(1 for pair in task_pairs if success(pair["ricl"]) and not success(pair["normal"])),
                sum(1 for pair in task_pairs if not success(pair["ricl"]) and success(pair["normal"])),
                sum(
                    1
                    for pair in task_pairs
                    if present(pair["ricl"]) and present(pair["normal"]) and not success(pair["ricl"]) and not success(pair["normal"])
                ),
                mean_steps(ricl),
                mean_steps(normal),
            ]
        )
    return rows


def rollout_pairs_sheet(paired: list[dict]) -> list[list[object]]:
    rows = [
        [
            "Task",
            "Scene setup",
            "Prompt",
            "RICL success",
            "Normal success",
            "Outcome",
            "RICL steps",
            "Normal steps",
            "Step delta RICL-Normal",
            "RICL video",
            "Normal video",
        ]
    ]
    for pair in paired:
        ricl = pair["ricl"]
        normal = pair["normal"]
        ricl_steps = elapsed_steps(ricl)
        normal_steps = elapsed_steps(normal)
        rows.append(
            [
                pair["task_id"],
                pair["scene_setup_id"],
                pair["prompt"],
                bool_text(success(ricl)) if present(ricl) else "",
                bool_text(success(normal)) if present(normal) else "",
                paired_outcome(ricl, normal),
                ricl_steps,
                normal_steps,
                "" if ricl_steps == "" or normal_steps == "" else ricl_steps - normal_steps,
                "" if ricl is None else ricl.get("video_path", ""),
                "" if normal is None else normal.get("video_path", ""),
            ]
        )
    return rows


def outcome_matrix_sheet(paired: list[dict]) -> list[list[object]]:
    rows = [["Outcome", "Count", "Percent of matched pairs"]]
    matched = [pair for pair in paired if present(pair["ricl"]) and present(pair["normal"])]
    outcomes = [
        ("Both succeeded", lambda p: success(p["ricl"]) and success(p["normal"])),
        ("RICL only", lambda p: success(p["ricl"]) and not success(p["normal"])),
        ("Normal only", lambda p: not success(p["ricl"]) and success(p["normal"])),
        ("Both failed", lambda p: not success(p["ricl"]) and not success(p["normal"])),
    ]
    for label, predicate in outcomes:
        count = sum(1 for pair in matched if predicate(pair))
        rows.append([label, count, percent(count, len(matched))])
    return rows


def artifacts_sheet(args: argparse.Namespace, exported_at: str) -> list[list[object]]:
    return [
        ["Key", "Value"],
        ["Exported at", exported_at],
        ["RICL results", str(args.ricl_results)],
        ["Normal results", str(args.normal_results)],
        ["Output XLSX", str(args.output)],
        ["Comparison", "Normal Pi0-FAST vs RICL20 Pi0-FAST on LIBERO-90 tasks 30-69, 10 scene setups/task."],
        ["Color legend", "Green = better task success count, red = worse task success count, yellow = tie."],
    ]


def print_summary(ricl_rows: list[dict], normal_rows: list[dict], paired: list[dict]) -> None:
    ricl_successes = count_successes(ricl_rows)
    normal_successes = count_successes(normal_rows)
    print(f"RICL20: {ricl_successes}/{len(ricl_rows)} ({percent(ricl_successes, len(ricl_rows))})")
    print(f"Normal: {normal_successes}/{len(normal_rows)} ({percent(normal_successes, len(normal_rows))})")
    print(f"Matched pairs: {sum(1 for pair in paired if present(pair['ricl']) and present(pair['normal']))}")


def present(row: dict | None) -> bool:
    return row is not None


def success(row: dict | None) -> bool:
    return bool(row and row.get("success"))


def elapsed_steps(row: dict | None) -> int | str:
    if row is None:
        return ""
    return int(row.get("elapsed_steps", 0))


def count_successes(rows: list[dict]) -> int:
    return sum(int(bool(row.get("success"))) for row in rows)


def count_timeouts(rows: list[dict]) -> int:
    return sum(1 for row in rows if int(row.get("elapsed_steps", 0)) >= 400)


def mean_steps(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return round(sum(int(row.get("elapsed_steps", 0)) for row in rows) / len(rows), 2)


def percent(numerator: int, denominator: int) -> str:
    return "" if denominator == 0 else f"{100.0 * numerator / denominator:.1f}%"


def pct_points(a_num: int, a_den: int, b_num: int, b_den: int) -> str:
    if a_den == 0 or b_den == 0:
        return ""
    return f"{100.0 * (a_num / a_den - b_num / b_den):.1f} pp"


def comparison_styles(ricl_successes: int, normal_successes: int) -> tuple[int, int, int]:
    if ricl_successes > normal_successes:
        return STYLE_WINNER, STYLE_LOSER, STYLE_WINNER
    if normal_successes > ricl_successes:
        return STYLE_LOSER, STYLE_WINNER, STYLE_LOSER
    return STYLE_TIE, STYLE_TIE, STYLE_TIE


def bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def paired_outcome(ricl: dict | None, normal: dict | None) -> str:
    if ricl is None:
        return "Missing RICL"
    if normal is None:
        return "Missing normal"
    if success(ricl) and success(normal):
        return "Both success"
    if success(ricl) and not success(normal):
        return "RICL only"
    if not success(ricl) and success(normal):
        return "Normal only"
    return "Both fail"


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
<fills count="5">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFC6EFCE"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFFFC7CE"/><bgColor indexed="64"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFFFEB9C"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="4">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/>
<xf numFmtId="0" fontId="0" fillId="3" borderId="0" xfId="0" applyFill="1"/>
<xf numFmtId="0" fontId="0" fillId="4" borderId="0" xfId="0" applyFill="1"/>
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
