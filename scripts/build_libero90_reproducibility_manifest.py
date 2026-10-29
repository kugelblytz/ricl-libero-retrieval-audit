#!/usr/bin/env python3
"""Build or verify the closed-loop LIBERO-90 artifact checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_OUTPUT = Path("data/libero/reports/libero90_closed_loop_retrieval/artifact_manifest.json")
RUN_ROOTS = (
    Path("data/libero/videos/libero90_pi_vs_dino_context_full_20260810"),
    Path("data/libero/videos/libero90_retrieval_final10_global25_disjoint_20260810_211205"),
    Path("data/libero/videos/libero90_retrieval_extension15_global25_disjoint_20260811_145749"),
)
REPORT_ROOT = Path("data/libero/reports/libero90_closed_loop_retrieval")
FINAL_REPORT = Path("data/libero/reports/libero90_global_retrieval_final_run_20260817.pdf")
ALLOWED_ROOT_SUFFIXES = {".csv", ".json"}
ALLOWED_FIGURE_SUFFIXES = {".pdf", ".png"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if the existing manifest is not current.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_artifacts(root: Path) -> set[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing run root: {root}")
    paths = {
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix in ALLOWED_ROOT_SUFFIXES
    }
    paths.update(root.glob("*/results.jsonl"))
    paths.update(root.glob("*/retrieval_trace.jsonl"))
    paths.update(root.glob("*/*_resources.csv"))
    figures = root / "figures"
    if figures.is_dir():
        paths.update(path for path in figures.iterdir() if path.suffix in ALLOWED_FIGURE_SUFFIXES)
    return paths


def _report_artifacts(output: Path) -> set[Path]:
    if not REPORT_ROOT.is_dir():
        raise FileNotFoundError(f"Missing report root: {REPORT_ROOT}")
    paths = {
        path
        for path in REPORT_ROOT.rglob("*")
        if path.is_file()
        and path != output
        and path.name != "README.md"
        and path.suffix in ALLOWED_ROOT_SUFFIXES | ALLOWED_FIGURE_SUFFIXES
    }
    if not FINAL_REPORT.is_file():
        raise FileNotFoundError(f"Missing final report: {FINAL_REPORT}")
    paths.add(FINAL_REPORT)
    return paths


def _validate_run_statuses() -> None:
    expected = (
        (RUN_ROOTS[0], "incomplete", 80),
        (RUN_ROOTS[1], "complete", 400),
        (RUN_ROOTS[2], "complete", 600),
    )
    for root, status, total_trials in expected:
        manifest = json.loads((root / "run_manifest.json").read_text())
        if manifest.get("status") != status:
            raise ValueError(f"Unexpected status for {root}: {manifest.get('status')!r}")
        if int(manifest.get("expected_total_trials", -1)) != total_trials:
            raise ValueError(f"Unexpected trial count for {root}: {manifest.get('expected_total_trials')!r}")


def _payload(output: Path) -> dict[str, object]:
    _validate_run_statuses()
    paths: set[Path] = set()
    for root in RUN_ROOTS:
        paths.update(_run_artifacts(root))
    paths.update(_report_artifacts(output))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")
    artifacts = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    ]
    return {
        "schema_version": 1,
        "description": "Committed LIBERO-90 closed-loop retrieval result and report artifacts.",
        "excluded_classes": [
            "rollout videos",
            "runtime logs (preserved separately in the historical artifact archive)",
            "feature caches",
            "datasets and retrieval pools",
            "model assets and checkpoints",
            "PID and tmux state",
        ],
        "run_statuses": {
            RUN_ROOTS[0].name: "incomplete",
            RUN_ROOTS[1].name: "complete (400/400 canonical trials)",
            RUN_ROOTS[2].name: "complete (600/600 canonical trials)",
        },
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in artifacts),
        "artifacts": artifacts,
    }


def main() -> None:
    args = _parse_args()
    payload = _payload(args.output)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"Missing manifest: {args.output}")
        if args.output.read_text() != rendered:
            raise SystemExit(f"Artifact manifest is stale: {args.output}")
        print(f"Verified {payload['artifact_count']} artifacts ({payload['total_size_bytes']} bytes)")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)
    print(f"Wrote {args.output}: {payload['artifact_count']} artifacts ({payload['total_size_bytes']} bytes)")


if __name__ == "__main__":
    main()
