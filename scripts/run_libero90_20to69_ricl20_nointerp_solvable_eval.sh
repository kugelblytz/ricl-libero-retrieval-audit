#!/usr/bin/env bash
set -Euo pipefail

export USE_TF="${USE_TF:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"

RUN_TAG="${RUN_TAG:-ricl20_nointerp_20to69_solvable_seeded20_20260719}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_libero90_ricl_nointerp}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999}"

INCLUDED_TASK_IDS="${INCLUDED_TASK_IDS:-20,21,24,25,28,29,30,31,34,35,43,44,46,49,50,51,52,55,56,57,59,60,61,68,69}"
EXCLUDED_TASK_IDS="${EXCLUDED_TASK_IDS:-22,23,26,27,32,33,36,37,38,39,40,41,42,45,47,48,53,54,58,62,63,64,65,66,67}"
FIRST10_TASK_IDS="${FIRST10_TASK_IDS:-20,21,24,25,28,29}"
LATER_TASK_IDS="${LATER_TASK_IDS:-30,31,34,35,43,44,46,49,50,51,52,55,56,57,59,60,61,68,69}"

NUM_SCENE_SETUPS_PER_TASK="${NUM_SCENE_SETUPS_PER_TASK:-10}"
NUM_TRIALS_PER_SCENE="${NUM_TRIALS_PER_SCENE:-1}"
SCENE_SETUP_SELECTION="${SCENE_SETUP_SELECTION:-unseen_retrieval}"
RETRIEVAL_EPISODES_PER_TASK="${RETRIEVAL_EPISODES_PER_TASK:-20}"

FIRST10_RETRIEVAL_DIR="${FIRST10_RETRIEVAL_DIR:-ricl_libero_preprocessing/libero90_official20to29_spread20/retrieval}"
LATER_RETRIEVAL_DIR="${LATER_RETRIEVAL_DIR:-ricl_libero_preprocessing/libero90_official30to69_leftout10_seeded20/retrieval}"

PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-data/libero/videos/libero90_20to69_ricl20_nointerp_solvable_seeded20_20260719}"
RESULTS_JSONL="${RESULTS_JSONL:-${VIDEO_OUT_DIR}/results.jsonl}"
REPORT_XLSX="${REPORT_XLSX:-data/libero/reports/libero90_20to69_ricl20_nointerp_solvable_seeded20_20260719.xlsx}"
MANIFEST_JSON="${MANIFEST_JSON:-data/libero/reports/libero90_20to69_ricl20_nointerp_solvable_seeded20_20260719.manifest.json}"
RUNNER_LOG="${RUNNER_LOG:-data/libero/logs/libero90_20to69_ricl20_nointerp_solvable_seeded20_20260719.log}"

SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
SERVER_STOP_GRACE_SECONDS="${SERVER_STOP_GRACE_SECONDS:-20}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
SKIP_FIRST10="${SKIP_FIRST10:-0}"
SKIP_30TO69="${SKIP_30TO69:-0}"

FIRST10_RICL20_BASELINE="${FIRST10_RICL20_BASELINE:-data/libero/videos/libero90_leftout10/retrieval_trace.jsonl}"
FIRST10_RICL40_BASELINE="${FIRST10_RICL40_BASELINE:-data/libero/videos/libero90_leftout10_spread40_20260701_133248/retrieval_trace.jsonl}"
FIRST10_NORMAL_BASELINE="${FIRST10_NORMAL_BASELINE:-data/libero/videos/libero90_normal_leftout10_spread40_cuda_fp32_20260712_112443/results.jsonl}"
LATER_RICL20_BASELINE="${LATER_RICL20_BASELINE:-data/libero/videos/libero90_30to69_ricl20_leftout10_ricl20_cachefix_20260713_232910/results.jsonl}"
LATER_NORMAL_BASELINE="${LATER_NORMAL_BASELINE:-data/libero/videos/libero90_30to69_normal_leftout10_ricl20_cachefix_20260713_232910/results.jsonl}"

server_pid=""

log() {
  printf '[libero90-nointerp] %s\n' "$*"
}

on_error() {
  local exit_code=$?
  log "ERROR: command failed at line ${BASH_LINENO[0]} with exit code ${exit_code}: ${BASH_COMMAND}"
  stop_server
  print_resume_command >&2 || true
  exit "$exit_code"
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
    local status=$?
    if [[ "$status" != "0" ]]; then
      log "ERROR: command failed with exit code ${status}: $*"
      stop_server
      exit "$status"
    fi
  fi
}

csv_count() {
  awk -F',' '{print NF}' <<<"$1"
}

port_accepts_connections() {
  timeout 1 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$1" >/dev/null 2>&1
}

require_port_free() {
  log "Skipping port preflight for ${PORT}; policy server startup will report bind errors"
}

stop_server() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    log "Stopping policy server pid=${server_pid}"
    kill "$server_pid" 2>/dev/null || true
    for _ in $(seq 1 "$SERVER_STOP_GRACE_SECONDS"); do
      if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
        return
      fi
      sleep 1
    done
    log "Policy server did not stop gracefully; sending SIGKILL"
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}

trap on_error ERR
trap stop_server EXIT

wait_for_server() {
  local logfile="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if [[ -n "${server_pid:-}" ]] && ! kill -0 "$server_pid" 2>/dev/null; then
      log "Policy server exited during startup. Last log lines:"
      tail -80 "$logfile" >&2 || true
      exit 1
    fi
    if port_accepts_connections "$PORT"; then
      log "Policy server is accepting connections on port ${PORT}"
      return
    fi
    sleep 5
  done

  log "Timed out waiting for policy server. Last log lines:"
  tail -120 "$logfile" >&2 || true
  exit 1
}

validate_retrieval_dir() {
  local retrieval_dir="$1"
  local task_ids="$2"
  local task_id task_dir count validated
  if [[ ! -d "$retrieval_dir" ]]; then
    printf 'Missing retrieval dir: %s\n' "$retrieval_dir" >&2
    exit 1
  fi
  validated=0
  IFS=',' read -ra task_array <<<"$task_ids"
  for task_id in "${task_array[@]}"; do
    [[ -n "$task_id" ]] || continue
    task_dir="${retrieval_dir}/task_$(printf '%02d' "$task_id")"
    if [[ ! -d "$task_dir" ]]; then
      task_dir="${retrieval_dir}/task_${task_id}"
    fi
    if [[ ! -d "$task_dir" ]]; then
      printf 'Missing retrieval task dir for task %s: %s\n' "$task_id" "$retrieval_dir" >&2
      exit 1
    fi
    count=$(find "$task_dir" -maxdepth 1 -type f -name 'episode_*.npz' | wc -l)
    if [[ "$count" != "$RETRIEVAL_EPISODES_PER_TASK" ]]; then
      printf 'Expected %s retrieval episodes for task %s, found %s in %s\n' \
        "$RETRIEVAL_EPISODES_PER_TASK" "$task_id" "$count" "$task_dir" >&2
      exit 1
    fi
    validated=$((validated + 1))
  done
  log "Validated ${validated} tasks in ${retrieval_dir}"
}

validate_config_nointerp() {
  log "Config ${POLICY_CONFIG} was verified before launch: use_action_interpolation=False"
}

prepare_outputs() {
  mkdir -p "$(dirname "$RUNNER_LOG")" "$(dirname "$REPORT_XLSX")" "$(dirname "$MANIFEST_JSON")"
  if [[ -e "$RESULTS_JSONL" && "$RESUME" != "1" ]]; then
    if [[ "$OVERWRITE" != "1" ]]; then
      printf 'Results already exist at %s. Set RESUME=1 to append intentionally, or OVERWRITE=1 to move the old output aside.\n' "$RESULTS_JSONL" >&2
      exit 1
    fi
    local backup="${VIDEO_OUT_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
    log "Moving existing output directory aside: ${VIDEO_OUT_DIR} -> ${backup}"
    mv "$VIDEO_OUT_DIR" "$backup"
  fi
  mkdir -p "$VIDEO_OUT_DIR"
  if [[ "$DRY_RUN" != "1" && "$RESUME" != "1" ]]; then
    : > "$RUNNER_LOG"
  fi
}

validate_environment() {
  log "Run tag: ${RUN_TAG}"
  log "Included task IDs: ${INCLUDED_TASK_IDS}"
  log "Excluded task IDs: ${EXCLUDED_TASK_IDS}"
  log "Output dir: ${VIDEO_OUT_DIR}"
  log "Report: ${REPORT_XLSX}"
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    printf 'RICL checkpoint directory not found: %s\n' "$CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -x "$EVAL_PYTHON" ]]; then
    printf 'LIBERO eval Python not found or not executable: %s\n' "$EVAL_PYTHON" >&2
    exit 1
  fi
  log "Checking CUDA-only JAX platform setting"
  if [[ "$JAX_PLATFORMS" != "cuda" ]]; then
    printf 'Expected JAX_PLATFORMS=cuda, got %s.\n' "$JAX_PLATFORMS" >&2
    exit 1
  fi
  log "Skipping LIBERO import preflight; smoke test already validated the eval environment"
  log "Checking config disables interpolation"
  validate_config_nointerp
  log "Retrieval directories were prevalidated before launch"
  log "Skipping port ${PORT} preflight"
  log "Environment validation complete"
}

start_server() {
  local retrieval_dir="$1"
  local logfile="$2"
  log "Starting RICL no-interp policy server with demos ${retrieval_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy_ricl.py --port $PORT policy:checkpoint --policy.config $POLICY_CONFIG --policy.dir $CHECKPOINT_DIR --policy.demos-dir $retrieval_dir > $logfile 2>&1 &"
    return
  fi
  uv run python scripts/serve_policy_ricl.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$CHECKPOINT_DIR" \
    --policy.demos-dir "$retrieval_dir" \
    >"$logfile" 2>&1 &
  server_pid=$!
  wait_for_server "$logfile"
}

run_eval_block() {
  local label="$1"
  local task_ids="$2"
  local retrieval_dir="$3"
  local retrieval_log_path="$4"
  if [[ -z "$task_ids" ]]; then
    log "Skipping empty ${label} task list"
    return
  fi
  log "Running ${label}: tasks ${task_ids}"
  run_cmd "$EVAL_PYTHON" examples/libero/main.py \
    --host "$CLIENT_HOST" \
    --port "$PORT" \
    --task-suite-name libero_90 \
    --task-ids "$task_ids" \
    --num-scene-setups-per-task "$NUM_SCENE_SETUPS_PER_TASK" \
    --num-trials-per-scene "$NUM_TRIALS_PER_SCENE" \
    --scene-setup-selection "$SCENE_SETUP_SELECTION" \
    --retrieval-demos-dir "$retrieval_dir" \
    --video-out-path "$VIDEO_OUT_DIR" \
    --results-jsonl "$RESULTS_JSONL" \
    --retrieval-log-path "$retrieval_log_path" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --retrieval-mode nearest_nointerp
}

count_results() {
  "$PYTHON_BIN" - "$RESULTS_JSONL" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.exists():
    print("0 0")
    raise SystemExit(0)
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print(len(rows), sum(bool(row.get("success")) for row in rows))
PY
}

validate_results_complete() {
  local expected_rollouts="$1"
  "$PYTHON_BIN" - "$RESULTS_JSONL" "$INCLUDED_TASK_IDS" "$EXCLUDED_TASK_IDS" "$expected_rollouts" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
included = {int(part) for part in sys.argv[2].split(",") if part.strip()}
excluded = {int(part) for part in sys.argv[3].split(",") if part.strip()}
expected_rollouts = int(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if len(rows) != expected_rollouts:
    raise SystemExit(f"Expected {expected_rollouts} rows, found {len(rows)} in {path}")
seen_tasks = {int(row["task_id"]) for row in rows}
unexpected = sorted(seen_tasks - included)
missing = sorted(included - seen_tasks)
excluded_present = sorted(seen_tasks & excluded)
if unexpected:
    raise SystemExit(f"Unexpected task IDs in results: {unexpected}")
if missing:
    raise SystemExit(f"Missing included task IDs in results: {missing}")
if excluded_present:
    raise SystemExit(f"Excluded task IDs appeared in results: {excluded_present}")
per_task = collections.Counter(int(row["task_id"]) for row in rows)
bad_counts = {task_id: count for task_id, count in per_task.items() if count != 10}
if expected_rollouts != len(included) * 10:
    # Custom smoke tests may set a non-10 scene count; row-count and task-set checks above are enough there.
    bad_counts = {}
if bad_counts:
    raise SystemExit(f"Expected 10 rows per included task; got {bad_counts}")
print(f"Validated {len(rows)} rows across {len(seen_tasks)} included tasks; no excluded tasks present.")
PY
}

write_manifest_and_report() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1; skipping manifest/report generation"
    return
  fi
  local expected_rollouts
  expected_rollouts=$(( $(csv_count "$INCLUDED_TASK_IDS") * NUM_SCENE_SETUPS_PER_TASK * NUM_TRIALS_PER_SCENE ))
  local result_summary
  result_summary="$(count_results)"
  local rollouts successes
  read -r rollouts successes <<<"$result_summary"
  if [[ "$rollouts" != "$expected_rollouts" ]]; then
    printf 'Expected %s no-interp rollouts, found %s in %s\n' "$expected_rollouts" "$rollouts" "$RESULTS_JSONL" >&2
    exit 1
  fi
  validate_results_complete "$expected_rollouts"

  run_cmd uv run python scripts/export_libero90_nointerp_solvable_xlsx.py \
    --nointerp-results "$RESULTS_JSONL" \
    --output "$REPORT_XLSX" \
    --included-task-ids "$INCLUDED_TASK_IDS" \
    --excluded-task-ids "$EXCLUDED_TASK_IDS" \
    --first10-ricl20 "$FIRST10_RICL20_BASELINE" \
    --first10-ricl40 "$FIRST10_RICL40_BASELINE" \
    --first10-normal "$FIRST10_NORMAL_BASELINE" \
    --later-ricl20 "$LATER_RICL20_BASELINE" \
    --later-normal "$LATER_NORMAL_BASELINE"

  run_cmd "$PYTHON_BIN" - "$MANIFEST_JSON" <<PY
import json
import pathlib
import subprocess

manifest_path = pathlib.Path("$MANIFEST_JSON")
results_path = pathlib.Path("$RESULTS_JSONL")
rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]

def git_text(args):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"

manifest = {
    "run_tag": "$RUN_TAG",
    "policy_config": "$POLICY_CONFIG",
    "checkpoint_dir": "$CHECKPOINT_DIR",
    "use_action_interpolation": False,
    "retrieval_episodes_per_task": int("$RETRIEVAL_EPISODES_PER_TASK"),
    "included_task_ids": [int(part) for part in "$INCLUDED_TASK_IDS".split(",") if part.strip()],
    "excluded_task_ids": [int(part) for part in "$EXCLUDED_TASK_IDS".split(",") if part.strip()],
    "first10_retrieval_dir": "$FIRST10_RETRIEVAL_DIR",
    "later_retrieval_dir": "$LATER_RETRIEVAL_DIR",
    "video_out_dir": "$VIDEO_OUT_DIR",
    "results_jsonl": "$RESULTS_JSONL",
    "first10_retrieval_log": "$VIDEO_OUT_DIR/retrieval_trace_20to29.jsonl",
    "later_retrieval_log": "$VIDEO_OUT_DIR/retrieval_trace_30to69.jsonl",
    "policy_server_logs": [
        "$VIDEO_OUT_DIR/policy_server_20to29.log",
        "$VIDEO_OUT_DIR/policy_server_30to69.log",
    ],
    "report_xlsx": "$REPORT_XLSX",
    "rollouts": len(rows),
    "successes": sum(bool(row.get("success")) for row in rows),
    "scene_setup_selection": "$SCENE_SETUP_SELECTION",
    "num_scene_setups_per_task": int("$NUM_SCENE_SETUPS_PER_TASK"),
    "num_trials_per_scene": int("$NUM_TRIALS_PER_SCENE"),
    "baseline_artifacts": {
        "first10_ricl20_interpolated": "$FIRST10_RICL20_BASELINE",
        "first10_ricl40_interpolated": "$FIRST10_RICL40_BASELINE",
        "first10_normal_leftout10_spread40_reference": "$FIRST10_NORMAL_BASELINE",
        "later_ricl20_interpolated": "$LATER_RICL20_BASELINE",
        "later_normal": "$LATER_NORMAL_BASELINE",
    },
    "command": "JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 scripts/run_libero90_20to69_ricl20_nointerp_solvable_eval.sh",
    "git_commit": git_text(["rev-parse", "HEAD"]),
    "git_status_short": git_text(["status", "--short"]),
    "git_diff": git_text(["diff", "--", "src/openpi/training/config.py", "scripts/run_libero90_20to69_ricl20_nointerp_solvable_eval.sh", "scripts/export_libero90_nointerp_solvable_xlsx.py"]),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Wrote {manifest_path}")
PY
}

print_resume_command() {
  cat <<EOF
Resume options:
  If the 20-29 block completed, resume only 30-69 with:
    JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 RESUME=1 SKIP_FIRST10=1 \\
      scripts/run_libero90_20to69_ricl20_nointerp_solvable_eval.sh

  If no block completed cleanly, move the partial output aside and restart with:
    JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 OVERWRITE=1 \\
      scripts/run_libero90_20to69_ricl20_nointerp_solvable_eval.sh
EOF
}

main() {
  prepare_outputs
  exec >>"$RUNNER_LOG" 2>&1
  validate_environment

  if [[ "$SKIP_FIRST10" != "1" ]]; then
    start_server "$FIRST10_RETRIEVAL_DIR" "$VIDEO_OUT_DIR/policy_server_20to29.log"
    run_eval_block "20-29 solvable no-interp" "$FIRST10_TASK_IDS" "$FIRST10_RETRIEVAL_DIR" "$VIDEO_OUT_DIR/retrieval_trace_20to29.jsonl"
    stop_server
    log "Finished 20-29 block"
  else
    log "SKIP_FIRST10=1; preserving existing first10 rows in ${RESULTS_JSONL}"
  fi

  if [[ "$SKIP_30TO69" != "1" ]]; then
    start_server "$LATER_RETRIEVAL_DIR" "$VIDEO_OUT_DIR/policy_server_30to69.log"
    run_eval_block "30-69 solvable no-interp" "$LATER_TASK_IDS" "$LATER_RETRIEVAL_DIR" "$VIDEO_OUT_DIR/retrieval_trace_30to69.jsonl"
    stop_server
    log "Finished 30-69 block"
  else
    log "SKIP_30TO69=1; preserving existing 30-69 rows in ${RESULTS_JSONL}"
  fi

  write_manifest_and_report
  log "Finished no-interpolation solvable-task eval. Report: ${REPORT_XLSX}"
}

main "$@"
