#!/usr/bin/env python3
"""Archive and checksum historical LIBERO-90 result artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

VIDEO_ROOT = Path("data/libero/videos")
TOP_LEVEL_LOG_ROOT = Path("data/libero/logs")
OUTPUT_ROOT = Path("data/libero/reports/libero90_historical_artifacts")
ARCHIVE_ROOT = OUTPUT_ROOT / "logs"
DEFAULT_MANIFEST = OUTPUT_ROOT / "artifact_manifest.json"
EXCLUDED_SUFFIXES = {".mp4", ".npy", ".npz"}
INCLUDED_RESULT_SUFFIXES = {".csv", ".json", ".jsonl", ".png"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="Verify archives and manifest without writing.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_artifacts() -> tuple[list[Path], list[Path]]:
    if not VIDEO_ROOT.is_dir():
        raise FileNotFoundError(f"Missing historical run root: {VIDEO_ROOT}")
    results: list[Path] = []
    logs: list[Path] = []
    for path in VIDEO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix in EXCLUDED_SUFFIXES:
            continue
        if path.suffix == ".log":
            logs.append(path)
        elif path.suffix in INCLUDED_RESULT_SUFFIXES:
            results.append(path)
        else:
            raise ValueError(f"Unclassified historical artifact: {path}")
    return sorted(results), sorted(logs)


def _top_level_experiment_logs() -> list[Path]:
    if not TOP_LEVEL_LOG_ROOT.is_dir():
        return []
    paths = set(TOP_LEVEL_LOG_ROOT.glob("libero90*.log"))
    paths.update(TOP_LEVEL_LOG_ROOT.glob("ricl*.log"))
    return sorted(path for path in paths if "watchdog" not in path.name)


def _archive_path(source: Path) -> Path:
    return ARCHIVE_ROOT / source.with_suffix(source.suffix + ".gz")


def _write_gzip(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with (
        source.open("rb") as source_handle,
        temporary.open("wb") as raw_output,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed,
    ):
        shutil.copyfileobj(source_handle, compressed, length=1024 * 1024)
    temporary.replace(destination)


def _manifest_payload(results: list[Path], logs: list[Path]) -> dict[str, object]:
    result_entries = [
        {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in results
    ]
    log_entries = []
    for source in logs:
        archive = _archive_path(source)
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archived log: {archive}")
        log_entries.append(
            {
                "source_path": source.as_posix(),
                "source_size_bytes": source.stat().st_size,
                "source_sha256": _sha256(source),
                "archive_path": archive.as_posix(),
                "archive_size_bytes": archive.stat().st_size,
                "archive_sha256": _sha256(archive),
                "compression": "gzip (mtime=0, original filename omitted)",
            }
        )
    return {
        "schema_version": 1,
        "description": "Historical LIBERO-90 results and deterministic log archives.",
        "result_artifact_count": len(result_entries),
        "result_artifact_size_bytes": sum(item["size_bytes"] for item in result_entries),
        "log_archive_count": len(log_entries),
        "log_source_size_bytes": sum(item["source_size_bytes"] for item in log_entries),
        "log_archive_size_bytes": sum(item["archive_size_bytes"] for item in log_entries),
        "excluded_classes": [
            "rollout videos",
            "feature caches",
            "datasets and retrieval pools",
            "model assets and checkpoints",
            "Codex rescue transcripts",
            "crontab snapshots",
            "watchdog, PID, lock, and tmux state",
        ],
        "result_artifacts": result_entries,
        "log_archives": log_entries,
    }


def main() -> None:
    args = _parse_args()
    results, video_logs = _video_artifacts()
    logs = sorted(video_logs + _top_level_experiment_logs())
    if not args.check:
        for source in logs:
            _write_gzip(source, _archive_path(source))
    payload = _manifest_payload(results, logs)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.manifest.is_file() or args.manifest.read_text() != rendered:
            raise SystemExit(f"Historical artifact manifest is stale: {args.manifest}")
        print(
            f"Verified {len(results)} results and {len(logs)} log archives "
            f"({payload['log_archive_size_bytes']} compressed bytes)"
        )
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(rendered)
    print(
        f"Wrote {args.manifest}: {len(results)} results and {len(logs)} log archives "
        f"({payload['log_archive_size_bytes']} compressed bytes)"
    )


if __name__ == "__main__":
    main()
