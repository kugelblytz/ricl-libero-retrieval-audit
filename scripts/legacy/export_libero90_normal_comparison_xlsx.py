from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape


OUTPUT_XLSX = Path("data/libero/reports/libero90_ricl_vs_normal_comparison.xlsx")

RICL_RUNS = (
    (
        "RICL 20mem leftout10",
        Path("data/libero/videos/libero90_leftout10/retrieval_trace.jsonl"),
        "RICL matched retrieval, 20 demos/task, 10 held-out scenes/task",
    ),
    (
        "RICL 40mem leftout10",
        Path("data/libero/videos/libero90_leftout10_spread40_20260701_133248/retrieval_trace.jsonl"),
        "RICL matched retrieval, 40 demos/task, 10 held-out scenes/task",
    ),
    (
        "RICL 49mem leftout1",
        Path("data/libero/videos/libero90_leftout1_spread49_20260701_133248/retrieval_trace.jsonl"),
        "RICL matched retrieval, 49 demos/task, 1 held-out scene/task",
    ),
    (
        "RICL mismatch successful tasks",
        Path("data/libero/videos/libero90_successful_tasks_mismatch_20260703_102406/retrieval_trace.jsonl"),
        "RICL mismatched retrieval diagnostic on successful-at-least-once tasks",
    ),
)


def main() -> None:
    runs = []
    for label, path, notes in RICL_RUNS:
        runs.append(load_ricl_trace(label, path, notes))

    normal_path = find_normal_results()
    if normal_path is None:
        runs.append(
            {
                "label": "Normal matched leftout10",
                "path": "",
                "notes": "PENDING: normal checkpoint/results.jsonl not present locally yet",
                "successes": None,
                "rollouts": None,
                "rate": None,
                "per_task": {},
            }
        )
    else:
        runs.append(load_normal_results("Normal matched leftout10", normal_path))

    summary_rows = [
        ["Run", "Successes", "Rollouts", "Success rate", "Artifact", "Notes"],
        *[
            [
                run["label"],
                "" if run["successes"] is None else run["successes"],
                "" if run["rollouts"] is None else run["rollouts"],
                "" if run["rate"] is None else f"{100.0 * run['rate']:.1f}%",
                str(run["path"]),
                run["notes"],
            ]
            for run in runs
        ],
    ]

    per_task_rows = [["Task", *[run["label"] for run in runs]]]
    for task_id in range(20, 30):
        row = [task_id]
        for run in runs:
            stats = run["per_task"].get(task_id)
            row.append("" if stats is None else f"{stats['successes']}/{stats['rollouts']}")
        per_task_rows.append(row)

    workbook = {
        "Summary": summary_rows,
        "PerTask": per_task_rows,
    }
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(OUTPUT_XLSX, workbook)
    print(f"Wrote {OUTPUT_XLSX}")
    if normal_path is None:
        print("Normal results are pending because no libero90_normal*/results.jsonl was found.")


def find_normal_results() -> Path | None:
    candidates = sorted(
        Path("data/libero/videos").glob("libero90_normal*/results.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_ricl_trace(label: str, path: Path, notes: str) -> dict:
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "notes": f"MISSING: {notes}",
            "successes": None,
            "rollouts": None,
            "rate": None,
            "per_task": {},
        }

    rollouts = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event_type") == "rollout_summary":
                rollouts.append(record)
    return summarize_rollouts(label, path, notes, rollouts, task_key="task_id", success_key="success")


def load_normal_results(label: str, path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return summarize_rollouts(
        label,
        path,
        "Normal Pi0-FAST, no retrieval context; same held-out scene selection.",
        rows,
        task_key="task_id",
        success_key="success",
    )


def summarize_rollouts(label: str, path: Path, notes: str, rollouts: list[dict], *, task_key: str, success_key: str) -> dict:
    per_task: dict[int, dict[str, int]] = defaultdict(lambda: {"successes": 0, "rollouts": 0})
    for rollout in rollouts:
        task_id = int(rollout[task_key])
        per_task[task_id]["rollouts"] += 1
        per_task[task_id]["successes"] += int(bool(rollout[success_key]))

    successes = sum(stats["successes"] for stats in per_task.values())
    rollout_count = sum(stats["rollouts"] for stats in per_task.values())
    return {
        "label": label,
        "path": str(path),
        "notes": notes,
        "successes": successes,
        "rollouts": rollout_count,
        "rate": None if rollout_count == 0 else successes / rollout_count,
        "per_task": dict(per_task),
    }


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
    if isinstance(value, int | float) and not isinstance(value, bool):
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
