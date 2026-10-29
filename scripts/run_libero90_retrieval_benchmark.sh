#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODE="${MODE:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/libero/retrieval_benchmark/libero90_solvable25_offline_v1}"
CALIBRATION_SOURCE_DIR="${CALIBRATION_SOURCE_DIR:-${OUTPUT_ROOT}/calibration_source}"
export LIBERO_CALIBRATION_RETRIEVAL_DIR="${CALIBRATION_SOURCE_DIR}/retrieval"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999}"
CONFIG_NAME="${CONFIG_NAME:-pi0_fast_libero90_normal}"
REPORT_XLSX="${REPORT_XLSX:-data/libero/reports/libero90_solvable25_offline_v1.xlsx}"
MANIFEST_JSON="${MANIFEST_JSON:-data/libero/reports/libero90_solvable25_offline_v1.manifest.json}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-64}"
PI0_BATCH_SIZE="${PI0_BATCH_SIZE:-4}"
RESUME="${RESUME:-1}"

log() {
  printf '[libero-retrieval-benchmark] %s\n' "$*"
}

run_benchmark() {
  local stage="$1"
  local split="$2"
  shift 2
  log "Running stage=${stage} split=${split}"
  uv run python scripts/benchmark_libero90_retrieval.py \
    "$stage" \
    --split "$split" \
    --output-root "$OUTPUT_ROOT" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --config-name "$CONFIG_NAME" \
    --dino-batch-size "$DINO_BATCH_SIZE" \
    --pi0-batch-size "$PI0_BATCH_SIZE" \
    "$@"
}

calibration_memory_complete() {
  local task_ids="$1"
  local task_id task_dir count
  local metadata_path="${LIBERO_CALIBRATION_RETRIEVAL_DIR}/retrieval_metadata.json"
  if [[ ! -f "$metadata_path" ]] || ! python - "$metadata_path" <<'PY'
import json
import pathlib
import sys

metadata = json.loads(pathlib.Path(sys.argv[1]).read_text())
raise SystemExit(0 if metadata.get("task_id_field") == "libero_global_task_id" else 1)
PY
  then
    return 1
  fi
  read -ra task_array <<<"$task_ids"
  for task_id in "${task_array[@]}"; do
    task_dir="${LIBERO_CALIBRATION_RETRIEVAL_DIR}/task_$(printf '%02d' "$task_id")"
    count=0
    if [[ -d "$task_dir" ]]; then
      count=$(find "$task_dir" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)
    fi
    if [[ "$count" != "20" ]]; then
      return 1
    fi
  done
}

prepare_calibration_memory() {
  local task_ids="$1"
  if calibration_memory_complete "$task_ids"; then
    log "Calibration memory is complete for tasks: ${task_ids}"
    return
  fi
  log "Building deterministic global-task calibration memory for tasks: ${task_ids}"
  read -ra task_array <<<"$task_ids"
  uv run python scripts/setup_ricl_libero.py \
    --repo-id local/libero90_30tasks \
    --task-indices "${task_array[@]}" \
    --task-id-field libero_global_task_id \
    --num-retrieved-observations 4 \
    --retrieval-key image \
    --embedding-type 64PATCHES \
    --embedding-batch-size "$DINO_BATCH_SIZE" \
    --max-episodes-per-task 20 \
    --episode-selection seeded \
    --seed 0 \
    --preprocessing-dir "$CALIBRATION_SOURCE_DIR" \
    --retrieval-dir "$LIBERO_CALIBRATION_RETRIEVAL_DIR" \
    --assets-dir "$CALIBRATION_SOURCE_DIR/assets/libero" \
    --artifact-role retrieval \
    --max-distance-source assets/pi0_fast_libero90_ricl/libero/max_distance.json \
    --local-files-only
  if ! calibration_memory_complete "$task_ids"; then
    log "ERROR: calibration memory validation failed after preparation"
    exit 1
  fi
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

mkdir -p "$OUTPUT_ROOT" "$(dirname "$REPORT_XLSX")" "$(dirname "$MANIFEST_JSON")"
run_benchmark validate full

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi

if [[ "$MODE" == "smoke" || "$MODE" == "all" ]]; then
  # The smoke calibration uses task 0 only. It validates the calibration path cheaply;
  # the full run recalibrates the final weights over all tasks 0-19.
  prepare_calibration_memory "0"
  run_benchmark extract-dino calibration --task-ids 0 "${resume_args[@]}"
  run_benchmark calibrate calibration \
    --task-ids 0 \
    --calibration-output "$OUTPUT_ROOT/calibration_smoke.json" \
    "${resume_args[@]}"
  run_benchmark extract-dino smoke "${resume_args[@]}"
  run_benchmark extract-pi0 smoke "${resume_args[@]}"
  run_benchmark score smoke \
    --calibration-path "$OUTPUT_ROOT/calibration_smoke.json" \
    "${resume_args[@]}"
  test -s "$OUTPUT_ROOT/smoke/validation.json"
  log "Smoke test passed"
fi

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
  if [[ "$MODE" == "full" && ! -s "$OUTPUT_ROOT/smoke/validation.json" ]]; then
    log "ERROR: full mode requires a completed smoke validation at $OUTPUT_ROOT/smoke/validation.json"
    exit 1
  fi
  prepare_calibration_memory "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
  if [[ "$RESUME" == "1" && -s "$OUTPUT_ROOT/calibration.json" ]]; then
    log "Calibration result is complete; reusing $OUTPUT_ROOT/calibration.json"
  else
    run_benchmark extract-dino calibration "${resume_args[@]}"
    run_benchmark calibrate calibration "${resume_args[@]}"
  fi
  run_benchmark extract-dino full "${resume_args[@]}"
  run_benchmark extract-pi0 full "${resume_args[@]}"
  run_benchmark score full "${resume_args[@]}"
  uv run python scripts/export_libero90_retrieval_benchmark_xlsx.py \
    --benchmark-root "$OUTPUT_ROOT" \
    --manifest "$MANIFEST_JSON" \
    --output "$REPORT_XLSX"
  log "Full benchmark complete"
  log "Report: ${REPORT_XLSX}"
  log "Manifest: ${MANIFEST_JSON}"
fi
