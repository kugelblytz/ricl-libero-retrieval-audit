#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODE="${MODE:-all}"
BASE_ROOT="${BASE_ROOT:-data/libero/retrieval_benchmark/libero90_solvable25_offline_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999}"
CONFIG_NAME="${CONFIG_NAME:-pi0_fast_libero90_normal}"
REPORT_XLSX="${REPORT_XLSX:-data/libero/reports/libero90_solvable25_mixed_global_v1.xlsx}"
MANIFEST_JSON="${MANIFEST_JSON:-data/libero/reports/libero90_solvable25_mixed_global_v1.manifest.json}"
PLOTS_DIR="${PLOTS_DIR:-data/libero/reports/libero90_solvable25_mixed_global_v1_plots}"
PI0_BATCH_SIZE="${PI0_BATCH_SIZE:-4}"
CUDA_QUERY_BATCH_SIZE="${CUDA_QUERY_BATCH_SIZE:-256}"
RESUME="${RESUME:-1}"

log() {
  printf '[libero-mixed-retrieval] %s\n' "$*"
}

run_stage() {
  local stage="$1"
  local split="$2"
  shift 2
  log "Running stage=${stage} split=${split}"
  uv run python scripts/benchmark_libero90_mixed_retrieval.py \
    "$stage" \
    --split "$split" \
    --base-root "$BASE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-name "$CONFIG_NAME" \
    --pi0-batch-size "$PI0_BATCH_SIZE" \
    --cuda-query-batch-size "$CUDA_QUERY_BATCH_SIZE" \
    "$@"
}

if [[ "$JAX_PLATFORMS" != "cuda" ]]; then
  log "ERROR: JAX_PLATFORMS must be cuda, got ${JAX_PLATFORMS}"
  exit 1
fi
if [[ "$OPENPI_POLICY_FLOAT32_FOR_CUDA" != "1" ]]; then
  log "ERROR: OPENPI_POLICY_FLOAT32_FOR_CUDA must be 1"
  exit 1
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "all" ]]; then
  log "ERROR: MODE must be smoke, full, or all"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT" "$(dirname "$REPORT_XLSX")" "$PLOTS_DIR"
resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi

if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  run_stage validate smoke
  run_stage extract-shuffled-pi0 smoke "${resume_args[@]}"
  run_stage build-distances smoke "${resume_args[@]}"
  run_stage score smoke "${resume_args[@]}"
  test -s "$OUTPUT_ROOT/smoke/validation.json"
  log "Smoke validation passed"
fi

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
  if [[ ! -s "$OUTPUT_ROOT/smoke/validation.json" ]]; then
    log "ERROR: full mode requires $OUTPUT_ROOT/smoke/validation.json"
    exit 1
  fi
  run_stage validate full
  run_stage extract-shuffled-pi0 full "${resume_args[@]}"
  run_stage build-distances full "${resume_args[@]}"
  run_stage score full "${resume_args[@]}"
  uv run python scripts/export_libero90_mixed_retrieval_benchmark.py \
    --benchmark-root "$OUTPUT_ROOT" \
    --manifest "$MANIFEST_JSON" \
    --output "$REPORT_XLSX" \
    --plots-dir "$PLOTS_DIR"
  log "Full mixed-task benchmark complete"
  log "Report: $REPORT_XLSX"
  log "Plots: $PLOTS_DIR"
  log "Manifest: $MANIFEST_JSON"
fi
