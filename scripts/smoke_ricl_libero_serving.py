from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("USE_TF", "0")

from scripts.serve_policy_ricl import Args
from scripts.serve_policy_ricl import Checkpoint
from scripts.serve_policy_ricl import create_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test RICL-LIBERO serving policy selection.")
    parser.add_argument("--config-name", default="pi0_fast_libero_ricl")
    parser.add_argument("--checkpoint-dir", default="checkpoints/pi0_fast_libero_ricl/smoke_libero_ricl/0")
    parser.add_argument("--demos-dir", default="ricl_libero_preprocessing/retrieval")
    args = parser.parse_args()

    policy = create_policy(
        Args(
            policy=Checkpoint(
                config=args.config_name,
                dir=args.checkpoint_dir,
                demos_dir=args.demos_dir,
            )
        )
    )
    assert type(policy).__name__ == "RiclLiberoPolicy"
    print(type(policy).__name__)


if __name__ == "__main__":
    main()
