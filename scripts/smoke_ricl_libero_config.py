from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("USE_TF", "0")

from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the RICL-LIBERO train config.")
    parser.add_argument("--config-name", default="pi0_fast_libero_ricl")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    assert "libero" in train_config.name
    assert "ricl" in train_config.name
    assert train_config.model.action_dim == 7
    assert train_config.model.action_horizon == 10
    assert train_config.model.num_retrieved_observations == 4
    assert train_config.batch_size == 16

    print(
        train_config.name,
        train_config.model.action_dim,
        train_config.model.action_horizon,
        train_config.model.num_retrieved_observations,
        train_config.batch_size,
    )


if __name__ == "__main__":
    main()
