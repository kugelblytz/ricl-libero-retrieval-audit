#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"

TASK_IDS="${TASK_IDS:-21,24}"
NUM_SCENE_SETUPS_PER_TASK="${NUM_SCENE_SETUPS_PER_TASK:-10}"
CONDITIONS="${CONDITIONS:-dino_state_task dino_state_global pi_task pi_global}"

POOL_DIR="${POOL_DIR:-ricl_libero_preprocessing/libero90_successful6_spread20/retrieval}"
POOL_SOURCE_DIR="${POOL_SOURCE_DIR:-ricl_libero_preprocessing/libero90_official20to29_spread20/retrieval}"
POOL_TASK_IDS="${POOL_TASK_IDS:-20,21,24,25,28,29}"
SCENE_SETUP_SELECTION="${SCENE_SETUP_SELECTION:-unseen_retrieval}"
SCENE_SETUP_MANIFEST_PATH="${SCENE_SETUP_MANIFEST_PATH:-}"
PI_FEATURE_DIR="${PI_FEATURE_DIR:-data/libero/retrieval_benchmark/libero90_solvable25_offline_v1/cache/candidate/pi0_fast_first_fusion}"
PI_RETRIEVER_CONFIG="${PI_RETRIEVER_CONFIG:-pi0_fast_libero90_normal}"
PI_RETRIEVER_CHECKPOINT="${PI_RETRIEVER_CHECKPOINT:-checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999}"
PI_RETRIEVER_DTYPE="${PI_RETRIEVER_DTYPE:-float32}"
STATE_NORM_CHECKPOINT="${STATE_NORM_CHECKPOINT:-${PI_RETRIEVER_CHECKPOINT}}"
DINO_STATE_BETA="${DINO_STATE_BETA:-1.0}"
RETRIEVAL_SEARCH_DEVICE="${RETRIEVAL_SEARCH_DEVICE:-auto}"
RETRIEVAL_DISTANCE_CHUNK_SIZE="${RETRIEVAL_DISTANCE_CHUNK_SIZE:-256}"

POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_libero90_ricl_nointerp}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/libero/videos/libero90_pi_vs_dino_context_${RUN_TAG}}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-600}"
SERVER_STOP_GRACE_SECONDS="${SERVER_STOP_GRACE_SECONDS:-30}"
INFERENCE_TIMEOUT_SECONDS="${INFERENCE_TIMEOUT_SECONDS:-180}"
MAX_ROLLOUT_SECONDS="${MAX_ROLLOUT_SECONDS:-900}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-43200}"
MAX_CONDITION_RESTARTS="${MAX_CONDITION_RESTARTS:-2}"
MAX_VRAM_MIB="${MAX_VRAM_MIB:-30000}"
MIN_HOST_AVAILABLE_MIB="${MIN_HOST_AVAILABLE_MIB:-3072}"
RESOURCE_SAMPLE_SECONDS="${RESOURCE_SAMPLE_SECONDS:-5}"
DRY_RUN="${DRY_RUN:-0}"

server_pid=""
resource_monitor_pid=""
run_started_epoch="$(date +%s)"
run_deadline_epoch="$((run_started_epoch + MAX_RUN_SECONDS))"

log() {
  printf '[pi-vs-dino] %s\n' "$*"
}

port_accepts_connections() {
  python - "$PORT" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

stop_server() {
  if [[ -n "${resource_monitor_pid:-}" ]] && kill -0 "$resource_monitor_pid" 2>/dev/null; then
    kill "$resource_monitor_pid" 2>/dev/null || true
    wait "$resource_monitor_pid" 2>/dev/null || true
    resource_monitor_pid=""
  fi
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
    kill -9 "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}

on_error() {
  local exit_code=$?
  log "ERROR at line ${BASH_LINENO[0]}: ${BASH_COMMAND} (exit ${exit_code})"
  stop_server
  if [[ -f "$OUTPUT_ROOT/run_manifest.json" ]]; then
    update_manifest_status "failed" "command failed with exit ${exit_code}" || true
  fi
  exit "$exit_code"
}

trap on_error ERR
trap stop_server EXIT

wait_for_server() {
  local logfile="$1"
  local deadline=$((SECONDS + SERVER_START_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      log "Policy server exited during startup"
      tail -120 "$logfile" >&2 || true
      exit 1
    fi
    if port_accepts_connections; then
      log "Policy server is ready on port ${PORT}"
      return
    fi
    sleep 5
  done
  log "Timed out waiting for policy server"
  tail -120 "$logfile" >&2 || true
  exit 1
}

remaining_run_seconds() {
  local remaining=$((run_deadline_epoch - $(date +%s)))
  (( remaining > 0 )) || return 1
  printf '%d\n' "$remaining"
}

monitor_resources() {
  local output_path="$1"
  printf 'unix_time,gpu_used_mib,gpu_free_mib,host_available_mib\n' >"$output_path"
  while [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; do
    local gpu_values
    local host_available
    gpu_values="$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    host_available="$(awk '/MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
    printf '%s,%s,%s\n' "$(date +%s)" "$gpu_values" "$host_available" >>"$output_path"
    sleep "$RESOURCE_SAMPLE_SECONDS"
  done
}

validate_resources() {
  local resource_path="$1"
  python - "$resource_path" "$MAX_VRAM_MIB" "$MIN_HOST_AVAILABLE_MIB" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing resource log: {path}")
rows = list(csv.DictReader(path.open()))
if not rows:
    raise SystemExit(f"Resource log has no samples: {path}")
peak_gpu = max(int(row["gpu_used_mib"]) for row in rows)
minimum_host = min(int(row["host_available_mib"]) for row in rows)
print(f"resource_peak_gpu_mib={peak_gpu} resource_min_host_available_mib={minimum_host}")
if peak_gpu > int(sys.argv[2]):
    raise SystemExit(f"Peak GPU memory {peak_gpu} MiB exceeds limit {sys.argv[2]} MiB")
if minimum_host < int(sys.argv[3]):
    raise SystemExit(f"Host available memory {minimum_host} MiB is below limit {sys.argv[3]} MiB")
PY
}

condition_method() {
  case "$1" in
    dino_global) printf 'dino\n' ;;
    dino_state_task|dino_state_global) printf 'dino_state\n' ;;
    pi_task|pi_global) printf 'pi0_first_fusion\n' ;;
    random_global) printf 'random\n' ;;
    *) printf 'Unknown condition: %s\n' "$1" >&2; exit 1 ;;
  esac
}

condition_scope() {
  case "$1" in
    dino_state_task|pi_task) printf 'task\n' ;;
    dino_global|dino_state_global|pi_global|random_global) printf 'global\n' ;;
    *) printf 'Unknown condition: %s\n' "$1" >&2; exit 1 ;;
  esac
}

validate_pool() {
  python - "$POOL_DIR" "$POOL_TASK_IDS" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = [int(value) for value in sys.argv[2].split(",")]
manifest = json.loads((root / "pool_manifest.json").read_text())
if manifest["task_ids"] != expected:
    raise SystemExit(f"Pool tasks differ: {manifest['task_ids']} != {expected}")
if manifest["num_episodes"] != len(expected) * 20:
    raise SystemExit(f"Expected {len(expected) * 20} episodes, found {manifest['num_episodes']}")
for task_id in expected:
    paths = list((root / f"task_{task_id}").glob("episode_*.npz"))
    if len(paths) != 20:
        raise SystemExit(f"Task {task_id} has {len(paths)} episodes; expected 20")
PY
}

ensure_dino_feature_cache() {
  local cache_dir="$POOL_DIR/dino_feature_cache"
  local features="$cache_dir/features.npy"
  local manifest="$cache_dir/manifest.json"
  if [[ -f "$features" && -f "$manifest" ]]; then
    log "Using memory-mapped DINO feature cache at ${cache_dir}"
    return
  fi
  if [[ -e "$features" || -e "$manifest" ]]; then
    log "Incomplete DINO feature cache at ${cache_dir}; remove or regenerate it explicitly"
    exit 1
  fi
  log "Building memory-mapped DINO feature cache at ${cache_dir}"
  uv run python scripts/export_libero_dino_feature_cache.py --demos-dir "$POOL_DIR"
}

preflight() {
  [[ "$PI_RETRIEVER_DTYPE" == "float32" ]] || {
    log "Pi cache equivalence requires PI_RETRIEVER_DTYPE=float32"
    exit 1
  }
  [[ "$RETRIEVAL_SEARCH_DEVICE" =~ ^(auto|cpu|cuda)$ ]] || {
    log "RETRIEVAL_SEARCH_DEVICE must be auto, cpu, or cuda"
    exit 1
  }
  [[ "$POLICY_CONFIG" == *nointerp* ]] || {
    log "Final comparative runs require a no-interpolation policy config; got ${POLICY_CONFIG}"
    exit 1
  }
  if [[ "$SCENE_SETUP_SELECTION" == "manifest" ]]; then
    [[ -f "$SCENE_SETUP_MANIFEST_PATH" ]] || {
      log "Manifest scene selection requires SCENE_SETUP_MANIFEST_PATH; missing: ${SCENE_SETUP_MANIFEST_PATH:-<unset>}"
      exit 1
    }
  elif [[ -n "$SCENE_SETUP_MANIFEST_PATH" ]]; then
    log "SCENE_SETUP_MANIFEST_PATH is set but SCENE_SETUP_SELECTION=${SCENE_SETUP_SELECTION}"
    exit 1
  fi
  declare -A seen_conditions=()
  for condition in $CONDITIONS; do
    condition_method "$condition" >/dev/null
    condition_scope "$condition" >/dev/null
    [[ -z "${seen_conditions[$condition]:-}" ]] || {
      log "Duplicate condition: ${condition}"
      exit 1
    }
    seen_conditions[$condition]=1
  done
  for positive_value in \
    "$NUM_SCENE_SETUPS_PER_TASK" "$RETRIEVAL_DISTANCE_CHUNK_SIZE" "$INFERENCE_TIMEOUT_SECONDS" \
    "$MAX_ROLLOUT_SECONDS" "$MAX_RUN_SECONDS"; do
    [[ "$positive_value" =~ ^[0-9]+$ ]] && (( positive_value > 0 )) || {
      log "Expected positive integer runtime limits, got ${positive_value}"
      exit 1
    }
  done
  if [[ ! -f "$POOL_DIR/pool_manifest.json" ]]; then
    log "Building successful-task episode pool"
    uv run python scripts/make_libero_retrieval_pool.py \
      --source-dir "$POOL_SOURCE_DIR" \
      --output-dir "$POOL_DIR" \
      --task-ids "$POOL_TASK_IDS" \
      --materialization hardlink
  fi
  validate_pool
  ensure_dino_feature_cache
  [[ -d "$CHECKPOINT_DIR" ]] || { log "Missing RICL checkpoint: $CHECKPOINT_DIR"; exit 1; }
  [[ -d "$PI_RETRIEVER_CHECKPOINT" ]] || { log "Missing Pi retriever checkpoint: $PI_RETRIEVER_CHECKPOINT"; exit 1; }
  [[ -d "$PI_FEATURE_DIR" ]] || { log "Missing Pi candidate features: $PI_FEATURE_DIR"; exit 1; }
  [[ -x "$EVAL_PYTHON" ]] || { log "Missing LIBERO eval Python: $EVAL_PYTHON"; exit 1; }
  if port_accepts_connections; then
    log "Port ${PORT} is already in use"
    exit 1
  fi
  command -v nvidia-smi >/dev/null || { log "nvidia-smi is required for resource safeguards"; exit 1; }
  "$EVAL_PYTHON" - <<'PY'
import robosuite  # noqa: F401
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
PY
  for task_id in ${POOL_TASK_IDS//,/ }; do
    [[ -f "$PI_FEATURE_DIR/task_$(printf '%02d' "$task_id").npy" ]] || {
      log "Missing Pi feature cache for task ${task_id}"
      exit 1
    }
  done
  if [[ "$DRY_RUN" != "1" && " $CONDITIONS " == *" pi_"* ]]; then
    mkdir -p "$OUTPUT_ROOT"
    log "Verifying pruned Pi query encoder against the full-model feature cache"
    timeout --signal=TERM --kill-after=30s 300 \
      uv run python scripts/check_pi_first_fusion_equivalence.py \
        --pool-dir "$POOL_DIR" \
        --feature-dir "$PI_FEATURE_DIR" \
        --checkpoint-dir "$PI_RETRIEVER_CHECKPOINT" \
        --config "$PI_RETRIEVER_CONFIG" \
        --task-ids "$POOL_TASK_IDS" \
        --output-json "$OUTPUT_ROOT/pi_first_fusion_equivalence.json"
  fi
}

start_server() {
  local condition="$1"
  local method
  local scope
  local logfile="$2"
  method="$(condition_method "$condition")"
  scope="$(condition_scope "$condition")"
  log "Starting ${condition}: method=${method}, scope=${scope}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return
  fi
  uv run python scripts/serve_policy_ricl.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$CHECKPOINT_DIR" \
    --policy.demos-dir "$POOL_DIR" \
    --policy.retrieval-method "$method" \
    --policy.retrieval-scope "$scope" \
    --policy.pi-feature-dir "$PI_FEATURE_DIR" \
    --policy.pi-retriever-config "$PI_RETRIEVER_CONFIG" \
    --policy.pi-retriever-checkpoint "$PI_RETRIEVER_CHECKPOINT" \
    --policy.pi-retriever-dtype "$PI_RETRIEVER_DTYPE" \
    --policy.state-norm-checkpoint "$STATE_NORM_CHECKPOINT" \
    --policy.dino-state-beta "$DINO_STATE_BETA" \
    --policy.retrieval-search-device "$RETRIEVAL_SEARCH_DEVICE" \
    --policy.retrieval-distance-chunk-size "$RETRIEVAL_DISTANCE_CHUNK_SIZE" \
    >"$logfile" 2>&1 &
  server_pid=$!
  monitor_resources "${logfile%.log}_resources.csv" &
  resource_monitor_pid=$!
  wait_for_server "$logfile"
}

validate_condition_results() {
  local condition="$1"
  local condition_dir="$OUTPUT_ROOT/$condition"
  local expected_method
  local expected_scope
  expected_method="$(condition_method "$condition")"
  expected_scope="$(condition_scope "$condition")"
  python - \
    "$condition_dir" "$condition" "$TASK_IDS" "$NUM_SCENE_SETUPS_PER_TASK" \
    "$expected_method" "$expected_scope" "$POOL_TASK_IDS" <<'PY'
import json
from pathlib import Path
import sys

condition_dir = Path(sys.argv[1])
condition = sys.argv[2]
task_ids = [int(value) for value in sys.argv[3].split(",")]
expected = len(task_ids) * int(sys.argv[4])
expected_method = sys.argv[5]
expected_scope = sys.argv[6]
expected_pool_tasks = len([value for value in sys.argv[7].split(",") if value])

def valid_rows(path):
    if not path.is_file():
        return {}
    rows = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = (
                str(row["retrieval_mode"]),
                int(row["task_id"]),
                int(row["scene_setup_id"]),
                int(row["trial_id"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise SystemExit(f"Malformed row {path}:{line_number}: {error}") from error
        if row.get("error") is None:
            rows[key] = row
    return rows

results = valid_rows(condition_dir / "results.jsonl")
traces = valid_rows(condition_dir / "retrieval_trace.jsonl")
if len(results) != expected:
    raise SystemExit(f"{condition}: expected {expected} valid results, found {len(results)}")
if set(results) != set(traces):
    raise SystemExit(f"{condition}: valid result and retrieval-trace identities differ")
for key, row in traces.items():
    if key[0] != condition or key[1] not in task_ids:
        raise SystemExit(f"{condition}: unexpected trial identity {key}")
    if not row.get("retrieval_trace"):
        raise SystemExit(f"{condition}: empty retrieval trace for {key}")
    for item in row["retrieval_trace"]:
        if item.get("retrieval_method") != expected_method:
            raise SystemExit(
                f"{condition}: retrieval method {item.get('retrieval_method')!r} != {expected_method!r}"
            )
        if item.get("retrieval_scope") != expected_scope:
            raise SystemExit(
                f"{condition}: retrieval scope {item.get('retrieval_scope')!r} != {expected_scope!r}"
            )
        # Runs produced before candidate-pool provenance was added do not have
        # this field.  Accept those legacy rows when resuming, but validate the
        # value whenever it is present.  The final-experiment wrapper performs
        # a separate strict check that requires the field for all new rows.
        reported_pool_tasks = item.get("candidate_pool_num_tasks")
        if (
            expected_scope == "global"
            and reported_pool_tasks is not None
            and reported_pool_tasks != expected_pool_tasks
        ):
            raise SystemExit(
                f"{condition}: candidate pool reports {reported_pool_tasks!r} tasks; "
                f"expected {expected_pool_tasks}"
            )
print(f"{condition}: validated {expected} canonical trials")
PY
}

run_condition() {
  local condition="$1"
  local condition_dir="$OUTPUT_ROOT/$condition"
  local max_attempts=$((MAX_CONDITION_RESTARTS + 1))
  local scene_setup_args=(--scene-setup-selection "$SCENE_SETUP_SELECTION")
  if [[ -n "$SCENE_SETUP_MANIFEST_PATH" ]]; then
    scene_setup_args+=(--scene-setup-manifest-path "$SCENE_SETUP_MANIFEST_PATH")
  fi
  mkdir -p "$condition_dir"
  if [[ "$DRY_RUN" != "1" ]] && validate_condition_results "$condition" >/dev/null 2>&1; then
    log "Skipping already complete condition ${condition}"
    return
  fi

  for attempt in $(seq 1 "$max_attempts"); do
    local remaining
    if ! remaining="$(remaining_run_seconds)"; then
      update_manifest_status "incomplete" "run reached ${MAX_RUN_SECONDS}s ceiling"
      log "Run ceiling reached before ${condition} completed"
      return 124
    fi
    log "Condition ${condition}, attempt ${attempt}/${max_attempts}, remaining budget ${remaining}s"
    start_server "$condition" "$condition_dir/policy_server_attempt_${attempt}.log"
    if [[ "$DRY_RUN" == "1" ]]; then
      stop_server
      log "Dry-run validated ${condition} launch arguments"
      return
    fi

    local eval_status=0
    timeout --signal=TERM --kill-after=30s "$remaining" \
      "$EVAL_PYTHON" examples/libero/main.py \
        --host "$CLIENT_HOST" \
        --port "$PORT" \
        --task-suite-name libero_90 \
        --task-ids "$TASK_IDS" \
        --num-scene-setups-per-task "$NUM_SCENE_SETUPS_PER_TASK" \
        --num-trials-per-scene 1 \
        "${scene_setup_args[@]}" \
        --retrieval-demos-dir "$POOL_DIR" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --retrieval-mode "$condition" \
        --video-out-path "$condition_dir/videos" \
        --results-jsonl "$condition_dir/results.jsonl" \
        --retrieval-log-path "$condition_dir/retrieval_trace.jsonl" \
        --inference-timeout-seconds "$INFERENCE_TIMEOUT_SECONDS" \
        --max-rollout-seconds "$MAX_ROLLOUT_SECONDS" \
        --resume \
        --fail-on-error \
        >>"$condition_dir/eval.log" 2>&1 || eval_status=$?
    stop_server

    local resource_path="$condition_dir/policy_server_attempt_${attempt}_resources.csv"
    if ! validate_resources "$resource_path" >>"$condition_dir/eval.log" 2>&1; then
      update_manifest_status "failed" "resource limit exceeded in ${condition}"
      return 1
    fi
    if (( eval_status == 124 )); then
      update_manifest_status "incomplete" "run reached ${MAX_RUN_SECONDS}s ceiling"
      return 124
    fi
    if (( eval_status == 0 )) && validate_condition_results "$condition" >>"$condition_dir/eval.log" 2>&1; then
      log "Completed ${condition}"
      return
    fi
    log "Condition ${condition} attempt ${attempt} failed with status ${eval_status}; retrying from valid rows"
  done

  update_manifest_status "failed" "${condition} exceeded ${MAX_CONDITION_RESTARTS} restarts"
  return 1
}

write_manifest() {
  python - \
    "$OUTPUT_ROOT" "$RUN_TAG" "$TASK_IDS" "$NUM_SCENE_SETUPS_PER_TASK" "$POOL_DIR" "$POOL_TASK_IDS" \
    "$CONDITIONS" "$CHECKPOINT_DIR" "$PI_RETRIEVER_CHECKPOINT" "$PI_FEATURE_DIR" \
    "$RETRIEVAL_SEARCH_DEVICE" "$RETRIEVAL_DISTANCE_CHUNK_SIZE" "$MAX_RUN_SECONDS" \
    "$DINO_STATE_BETA" "$PI_RETRIEVER_DTYPE" "$SCENE_SETUP_SELECTION" \
    "$SCENE_SETUP_MANIFEST_PATH" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
task_ids = [int(value) for value in sys.argv[3].split(",")]
conditions = sys.argv[7].split()
pool_manifest = json.loads((Path(sys.argv[5]) / "pool_manifest.json").read_text())
scene_manifest_path = Path(sys.argv[17]) if sys.argv[17] else None
payload = {
    "run_tag": sys.argv[2],
    "task_ids": task_ids,
    "scene_setups_per_task": int(sys.argv[4]),
    "pool_dir": str(Path(sys.argv[5]).resolve()),
    "pool_tasks": [int(value) for value in sys.argv[6].split(",")],
    "episodes_per_pool_task": 20,
    "total_pool_episodes": int(pool_manifest["num_episodes"]),
    "conditions": conditions,
    "expected_trials_per_condition": len(task_ids) * int(sys.argv[4]),
    "expected_total_trials": len(task_ids) * int(sys.argv[4]) * len(conditions),
    "retrieved_observations": 4,
    "action_interpolation": False,
    "dino_state_beta": float(sys.argv[14]),
    "policy_checkpoint": str(Path(sys.argv[8]).resolve()),
    "pi_retriever_checkpoint": str(Path(sys.argv[9]).resolve()),
    "pi_feature_dir": str(Path(sys.argv[10]).resolve()),
    "pi_retriever_dtype": sys.argv[15],
    "retrieval_search_device": sys.argv[11],
    "retrieval_distance_chunk_size": int(sys.argv[12]),
    "replan_steps": 5,
    "scene_setup_selection": sys.argv[16],
    "scene_setup_manifest_path": None if scene_manifest_path is None else str(scene_manifest_path.resolve()),
    "scene_setup_manifest_sha256": (
        None if scene_manifest_path is None else hashlib.sha256(scene_manifest_path.read_bytes()).hexdigest()
    ),
    "trials_per_scene": 1,
    "max_run_seconds": int(sys.argv[13]),
    "started_at": datetime.now(timezone.utc).isoformat(),
    "status": "running",
}
root.mkdir(parents=True, exist_ok=True)
path = root / "run_manifest.json"
if path.is_file():
    existing = json.loads(path.read_text())
    # Manifests written before explicit scene-selection provenance used these
    # defaults.  Fill them in so a completed legacy condition can be resumed
    # without changing its rollout identities.
    existing_for_comparison = {
        "scene_setup_selection": "unseen_retrieval",
        "scene_setup_manifest_path": None,
        "scene_setup_manifest_sha256": None,
        **existing,
    }
    immutable = (
        "task_ids",
        "scene_setups_per_task",
        "pool_dir",
        "pool_tasks",
        "conditions",
        "policy_checkpoint",
        "pi_retriever_checkpoint",
        "pi_feature_dir",
        "dino_state_beta",
        "pi_retriever_dtype",
        "scene_setup_selection",
        "scene_setup_manifest_path",
        "scene_setup_manifest_sha256",
    )
    mismatches = [key for key in immutable if existing_for_comparison.get(key) != payload.get(key)]
    if mismatches:
        raise SystemExit(f"Cannot resume {path}; configuration differs for {mismatches}")
    payload = {
        **payload,
        **existing,
        "scene_setup_selection": existing_for_comparison["scene_setup_selection"],
        "scene_setup_manifest_path": existing_for_comparison["scene_setup_manifest_path"],
        "scene_setup_manifest_sha256": existing_for_comparison["scene_setup_manifest_sha256"],
        "status": "running",
        "status_detail": None,
        "resumed_at": datetime.now(timezone.utc).isoformat(),
    }
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
}

update_manifest_status() {
  local status="$1"
  local detail="${2:-}"
  python - "$OUTPUT_ROOT/run_manifest.json" "$status" "$detail" <<'PY'
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["status"] = sys.argv[2]
payload["status_detail"] = sys.argv[3] or None
payload["updated_at"] = datetime.now(timezone.utc).isoformat()
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
temporary.replace(path)
PY
}

mark_complete() {
  update_manifest_status "complete" "validated all expected canonical trials"
}

main() {
  preflight
  write_manifest
  log "Output root: ${OUTPUT_ROOT}"
  for condition in $CONDITIONS; do
    condition_status=0
    run_condition "$condition" || condition_status=$?
    if (( condition_status != 0 )); then
      exit "$condition_status"
    fi
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    for condition in $CONDITIONS; do
      validate_condition_results "$condition"
    done
    uv run python scripts/summarize_libero_pi_vs_dino.py --output-root "$OUTPUT_ROOT"
    mark_complete
  else
    update_manifest_status "dry_run_complete" "validated preflight and all condition launch arguments"
  fi
  log "All conditions complete"
}

main "$@"
