from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LIBERO-90 task 30-69 RICL-vs-normal comparison to XLSX.")
    parser.add_argument("--ricl-results", type=Path, required=True)
    parser.add_argument("--normal-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    ricl = load_results("RICL20 tasks 30-69", args.ricl_results, "RICL Pi0-FAST with 20 retrieval demos/task.")
    normal = load_results("Normal tasks 30-69", args.normal_results, "Normal Pi0-FAST, no retrieval context.")
    runs = [ricl, normal]

    summary_rows = [
        ["Run", "Successes", "Rollouts", "Success rate", "Artifact", "Notes"],
        *[
            [
                run["label"],
                run["successes"],
                run["rollouts"],
                f"{100.0 * run['rate']:.1f}%" if run["rate"] is not None else "",
                run["path"],
                run["notes"],
            ]
            for run in runs
        ],
    ]

    per_task_rows = [["Task", "Prompt", "RICL20", "Normal", "Delta normal-ricl"]]
    for task_id in range(30, 70):
        ricl_stats = ricl["per_task"].get(task_id, empty_task())
        normal_stats = normal["per_task"].get(task_id, empty_task())
        prompt = normal_stats["prompt"] or ricl_stats["prompt"]
        per_task_rows.append(
            [
                task_id,
                prompt,
                ratio_text(ricl_stats),
                ratio_text(normal_stats),
                normal_stats["successes"] - ricl_stats["successes"],
            ]
        )

    scene_rows = [["Task", "Scene setup", "RICL success", "Normal success", "RICL steps", "Normal steps"]]
    scene_keys = sorted(set(ricl["per_scene"]) | set(normal["per_scene"]))
    for task_id, scene_setup_id in scene_keys:
        ricl_scene = ricl["per_scene"].get((task_id, scene_setup_id), {})
        normal_scene = normal["per_scene"].get((task_id, scene_setup_id), {})
        scene_rows.append(
            [
                task_id,
                scene_setup_id,
                bool_text(ricl_scene.get("success")),
                bool_text(normal_scene.get("success")),
                ricl_scene.get("elapsed_steps", ""),
                normal_scene.get("elapsed_steps", ""),
            ]
        )

    artifacts_rows = [
        ["Key", "Value"],
        ["RICL results", str(args.ricl_results)],
        ["Normal results", str(args.normal_results)],
        ["Output XLSX", str(args.output)],
    ]
    if args.manifest is not None:
        artifacts_rows.append(["Manifest", str(args.manifest)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(
        args.output,
        {
            "Summary": summary_rows,
            "PerTask": per_task_rows,
            "PerScene": scene_rows,
            "Artifacts": artifacts_rows,
        },
    )
    print(f"Wrote {args.output}")


def load_results(label: str, path: Path, notes: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    per_task: dict[int, dict] = defaultdict(empty_task)
    per_scene = {}
    for row in rows:
        task_id = int(row["task_id"])
        scene_setup_id = int(row["scene_setup_id"])
        success = bool(row["success"])
        stats = per_task[task_id]
        stats["prompt"] = stats["prompt"] or row.get("task_language", "")
        stats["rollouts"] += 1
        stats["successes"] += int(success)
        per_scene[(task_id, scene_setup_id)] = {
            "success": success,
            "elapsed_steps": row.get("elapsed_steps", ""),
        }

    successes = sum(stats["successes"] for stats in per_task.values())
    rollouts = sum(stats["rollouts"] for stats in per_task.values())
    return {
        "label": label,
        "path": str(path),
        "notes": notes,
        "successes": successes,
        "rollouts": rollouts,
        "rate": None if rollouts == 0 else successes / rollouts,
        "per_task": dict(per_task),
        "per_scene": per_scene,
    }


def empty_task() -> dict:
    return {"prompt": "", "successes": 0, "rollouts": 0}


def ratio_text(stats: dict) -> str:
    return f"{stats['successes']}/{stats['rollouts']}"


def bool_text(value: object) -> str:
    if value == "":
        return ""
    if value is None:
        return ""
    return "TRUE" if bool(value) else "FALSE"


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
