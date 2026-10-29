from __future__ import annotations

import argparse
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("USE_TF", "0")

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the RICL-LIBERO data loader.")
    parser.add_argument("--config-name", default="pi0_fast_libero_ricl")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-norm-stats", action="store_true")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        train_config = dataclasses.replace(train_config, batch_size=args.batch_size)

    loader = _data_loader.create_data_loader(
        train_config,
        sharding=None,
        skip_norm_stats=args.skip_norm_stats,
        shuffle=False,
        num_batches=1,
        num_workers=args.num_workers,
    )
    if not args.skip_norm_stats and loader.data_config().norm_stats is None:
        raise ValueError("Expected RICL-LIBERO norm stats. Run `uv run python scripts/setup_ricl_libero.py` first.")

    observation, actions = next(iter(loader))
    expected_action_shape = (train_config.model.action_horizon, train_config.model.action_dim)
    expected_distance_shape = (train_config.model.num_retrieved_observations + 1, 1)

    assert actions.shape[-2:] == expected_action_shape, actions.shape
    assert observation.exp_lamda_distances is not None
    assert observation.exp_lamda_distances.shape[-2:] == expected_distance_shape

    print(
        "RICL-LIBERO data loader OK",
        f"query_state={observation.query_state.shape}",
        f"actions={actions.shape}",
        f"exp_lamda_distances={observation.exp_lamda_distances.shape}",
    )


if __name__ == "__main__":
    main()
