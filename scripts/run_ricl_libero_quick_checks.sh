#!/usr/bin/env bash
set -euo pipefail

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export USE_TF="${USE_TF:-0}"

RICL_LIBERO_FILES=(
  src/openpi/training/ricl_libero_dataset.py
  scripts/setup_ricl_libero.py
  scripts/smoke_ricl_libero_config.py
  scripts/validate_ricl_libero_artifacts.py
  scripts/smoke_ricl_libero_data_loader.py
  scripts/smoke_ricl_libero_serving.py
  src/openpi/training/ricl_libero_dataset_test.py
  src/openpi/policies/ricl_libero_policy_test.py
)

git diff --check
uv run ruff check "${RICL_LIBERO_FILES[@]}"
python -m py_compile "${RICL_LIBERO_FILES[@]}"
uv run pytest \
  src/openpi/training/ricl_libero_dataset_test.py \
  src/openpi/policies/ricl_libero_policy_test.py \
  -q
uv run python scripts/smoke_ricl_libero_config.py
uv run python scripts/smoke_ricl_libero_config.py --config-name pi0_fast_libero90_ricl
