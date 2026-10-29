from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import publish_libero_rollout_videos as archive


def test_inventory_and_render_are_deterministic(tmp_path):
    videos = tmp_path / "videos"
    first = videos / "run_b" / "second.mp4"
    second = videos / "run_a" / "first.mp4"
    root_video = videos / "root.mp4"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"second-video")
    second.write_bytes(b"first-video")
    root_video.write_bytes(b"root-video")

    rows = archive._inventory(videos)  # noqa: SLF001

    assert [row["path"] for row in rows] == ["root.mp4", "run_a/first.mp4", "run_b/second.mp4"]
    assert rows[0]["run"] == "_root"
    assert rows[1]["sha256"] == hashlib.sha256(b"first-video").hexdigest()
    rendered = archive._render(rows, "owner/private-videos")  # noqa: SLF001
    summary = json.loads(rendered["summary.json"])
    assert summary["video_count"] == 3
    assert summary["run_count"] == 3
    assert summary["total_size_bytes"] == len(b"root-video") + len(b"first-video") + len(b"second-video")
    assert len(rendered["video_manifest.jsonl"].splitlines()) == 3


def test_metadata_check_detects_stale_files(tmp_path):
    output = tmp_path / "metadata"
    rendered = {"README.md": "archive\n", "summary.json": "{}\n"}
    archive._write_or_check(output, rendered, check=False)  # noqa: SLF001
    archive._write_or_check(output, rendered, check=True)  # noqa: SLF001

    (output / "summary.json").write_text("stale\n")
    with pytest.raises(SystemExit, match="summary.json"):
        archive._write_or_check(output, rendered, check=True)  # noqa: SLF001
