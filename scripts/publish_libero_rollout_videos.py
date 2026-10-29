#!/usr/bin/env python3
"""Inventory, verify, and publish LIBERO rollout videos to a private Hub dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_VIDEOS_ROOT = Path("data/libero/videos")
DEFAULT_OUTPUT_DIR = Path("data/libero/reports/libero90_rollout_video_archive")
DEFAULT_REPO_ID = "kugelblytz/ricl-libero-rollout-videos"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Verify that local metadata matches the videos.")
    action.add_argument("--upload", action="store_true", help="Create/update the private dataset and upload videos.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(videos_root: Path) -> list[dict[str, object]]:
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Missing videos root: {videos_root}")
    videos = sorted(path for path in videos_root.rglob("*.mp4") if path.is_file())
    if not videos:
        raise ValueError(f"No MP4 files found under {videos_root}")
    rows = []
    for path in videos:
        relative_path = path.relative_to(videos_root)
        rows.append(
            {
                "path": relative_path.as_posix(),
                "run": relative_path.parts[0] if len(relative_path.parts) > 1 else "_root",
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _summary(rows: list[dict[str, object]], repo_id: str) -> dict[str, object]:
    runs: dict[str, dict[str, int]] = {}
    for row in rows:
        run = str(row["run"])
        run_summary = runs.setdefault(run, {"video_count": 0, "size_bytes": 0})
        run_summary["video_count"] += 1
        run_summary["size_bytes"] += int(row["size_bytes"])
    return {
        "schema_version": 1,
        "description": "Private archive of RICL/OpenPI LIBERO rollout videos.",
        "source_root": DEFAULT_VIDEOS_ROOT.as_posix(),
        "source_repository": "https://github.com/kugelblytz/ricl_libero",
        "huggingface_dataset": f"https://huggingface.co/datasets/{repo_id}",
        "video_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "run_count": len(runs),
        "runs": dict(sorted(runs.items())),
    }


def _readme(summary: dict[str, object]) -> str:
    return f"""---
pretty_name: RICL LIBERO Rollout Videos
tags:
- robotics
- libero
- vision-language-action
- retrieval
---

# RICL LIBERO rollout videos

Private video archive for the LIBERO experiments in
[`kugelblytz/ricl_libero`](https://github.com/kugelblytz/ricl_libero). It
contains {summary['video_count']:,} MP4 rollout videos across
{summary['run_count']} run groups ({int(summary['total_size_bytes']) / 2**20:.1f}
MiB total).

The directory layout is preserved relative to `data/libero/videos`.
`video_manifest.jsonl` records each file's relative path, byte size, run group,
and SHA-256 checksum. `summary.json` provides archive-level and per-run counts.

These are simulator rollout outputs, not the LIBERO demonstration dataset used
for training or retrieval memory construction. Result JSONL files, retrieval
traces, protocols, reports, and analysis code remain versioned in the source
Git repository.

## Verification

From the source repository checkout:

```bash
uv run python scripts/publish_libero_rollout_videos.py --check
```

The repository is private because rollout archives are large operational
artifacts and are not part of the public benchmark release.
"""


def _render(rows: list[dict[str, object]], repo_id: str) -> dict[str, str]:
    summary = _summary(rows, repo_id)
    return {
        "README.md": _readme(summary),
        "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "video_manifest.jsonl": "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    }


def _write_or_check(output_dir: Path, rendered: dict[str, str], *, check: bool) -> None:
    if check:
        stale = [
            name
            for name, content in rendered.items()
            if not (output_dir / name).is_file() or (output_dir / name).read_text() != content
        ]
        if stale:
            raise SystemExit(f"Rollout video archive metadata is stale: {', '.join(stale)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        (output_dir / name).write_text(content)


def _upload(args: argparse.Namespace, expected_video_count: int) -> None:
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(args.repo_id, repo_type="dataset", private=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.output_dir,
        allow_patterns=["README.md", "summary.json", "video_manifest.jsonl"],
        commit_message="Add rollout video archive metadata",
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.videos_root,
        allow_patterns=["*.mp4", "**/*.mp4"],
        commit_message="Upload LIBERO rollout videos",
    )
    remote_videos = [path for path in api.list_repo_files(args.repo_id, repo_type="dataset") if path.endswith(".mp4")]
    if len(remote_videos) != expected_video_count:
        raise RuntimeError(f"Remote video count mismatch: expected {expected_video_count}, found {len(remote_videos)}")
    print(f"Verified {len(remote_videos)} videos at https://huggingface.co/datasets/{args.repo_id}")


def main() -> None:
    args = _parse_args()
    rows = _inventory(args.videos_root)
    rendered = _render(rows, args.repo_id)
    _write_or_check(args.output_dir, rendered, check=args.check)
    total_size = sum(int(row["size_bytes"]) for row in rows)
    print(f"Inventoried {len(rows)} videos ({total_size} bytes) across {len({row['run'] for row in rows})} runs")
    if args.upload:
        _upload(args, len(rows))


if __name__ == "__main__":
    main()
