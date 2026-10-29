from __future__ import annotations

import argparse
from pathlib import Path

from openpi.policies import libero_dino_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export LIBERO episode DINO embeddings as one memory-mapped feature matrix."
    )
    parser.add_argument("--demos-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = libero_dino_feature_cache.build_cache(args.demos_dir, overwrite=args.overwrite)
    gib = manifest["frame_count"] * manifest["feature_dim"] * 4 / 2**30
    print(
        f"Wrote {manifest['frame_count']} x {manifest['feature_dim']} float32 DINO features "
        f"({gib:.2f} GiB) under {args.demos_dir / libero_dino_feature_cache.CACHE_DIRNAME}"
    )


if __name__ == "__main__":
    main()
