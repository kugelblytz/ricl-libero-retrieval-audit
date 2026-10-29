#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

TASK_IDS="${TASK_IDS:-20-29}"
TASK_INDICES="${TASK_INDICES:-20 21 22 23 24 25 26 27 28 29}"
REPO_ID="${REPO_ID:-local/libero90_official20to29}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_libero90_ricl}"
ASSETS_DIR="${ASSETS_DIR:-assets/pi0_fast_libero90_ricl/libero}"
MAX_DISTANCE_SOURCE="${MAX_DISTANCE_SOURCE:-ricl_libero_preprocessing/libero90_train20x20/max_distance.json}"
PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
SERVER_STOP_GRACE_SECONDS="${SERVER_STOP_GRACE_SECONDS:-20}"
DRY_RUN="${DRY_RUN:-0}"

server_pid=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

retrieval_is_complete() {
  local retrieval_dir="$1"
  local max_episodes="$2"

  python - "$retrieval_dir" "$max_episodes" <<'PY'
import json
import pathlib
import sys

retrieval_dir = pathlib.Path(sys.argv[1])
max_episodes = int(sys.argv[2])
metadata_path = retrieval_dir / "retrieval_metadata.json"
if not metadata_path.exists():
    raise SystemExit(1)
try:
    metadata = json.loads(metadata_path.read_text())
except (json.JSONDecodeError, OSError):
    raise SystemExit(1)
if metadata.get("max_episodes_per_task") != max_episodes:
    raise SystemExit(1)
task_indices = metadata.get("task_indices") or []
if task_indices != list(range(20, 30)):
    raise SystemExit(1)
for task_id in task_indices:
    task_dir = retrieval_dir / f"task_{task_id}"
    if len(list(task_dir.glob("episode_*.npz"))) != max_episodes:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

port_accepts_connections() {
  python - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

require_port_free() {
  if port_accepts_connections "$PORT"; then
    printf 'Port %s is already accepting connections. Stop the existing server or set PORT=...\n' "$PORT" >&2
    exit 1
  fi
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

on_exit() {
  stop_server
}
trap on_exit EXIT

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

build_retrieval() {
  local max_episodes="$1"
  local preprocessing_dir="$2"
  local retrieval_dir="$3"

  if retrieval_is_complete "$retrieval_dir" "$max_episodes"; then
    log "Retrieval memory already complete at ${retrieval_dir}; skipping build"
    return
  fi

  run_cmd uv run python scripts/setup_ricl_libero.py \
    --repo-id "$REPO_ID" \
    --task-indices $TASK_INDICES \
    --task-id-field libero_task_id \
    --num-retrieved-observations 4 \
    --retrieval-key image \
    --embedding-type 64PATCHES \
    --max-episodes-per-task "$max_episodes" \
    --episode-selection spread \
    --preprocessing-dir "$preprocessing_dir" \
    --retrieval-dir "$retrieval_dir" \
    --assets-dir "$ASSETS_DIR" \
    --artifact-role retrieval \
    --max-distance-source "$MAX_DISTANCE_SOURCE"
}

start_server() {
  local retrieval_dir="$1"
  local logfile="$2"

  log "Starting policy server with demos ${retrieval_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy_ricl.py --port $PORT policy:checkpoint --policy.config $POLICY_CONFIG --policy.dir $CHECKPOINT_DIR --policy.demos-dir $retrieval_dir > $logfile 2>&1 &"
    return
  fi

  require_port_free
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

run_eval() {
  local retrieval_dir="$1"
  local video_dir="$2"

  run_cmd "$EVAL_PYTHON" examples/libero/main.py \
    --host "$CLIENT_HOST" \
    --port "$PORT" \
    --task-suite-name libero_90 \
    --task-ids "$TASK_IDS" \
    --num-scene-setups-per-task "$3" \
    --num-trials-per-scene 1 \
    --scene-setup-selection unseen_retrieval \
    --retrieval-demos-dir "$retrieval_dir" \
    --video-out-path "$video_dir" \
    --retrieval-log-path "$video_dir/retrieval_trace.jsonl"
}

run_experiment() {
  local name="$1"
  local max_episodes="$2"
  local eval_setups="$3"
  local preprocessing_dir="ricl_libero_preprocessing/libero90_official20to29_${name}"
  local retrieval_dir="${preprocessing_dir}/retrieval"
  local video_dir="data/libero/videos/libero90_${name}_${RUN_TAG}"
  local server_log="${video_dir}/policy_server.log"

  log "=== Experiment ${name}: ${max_episodes} retrieval episodes/task, ${eval_setups} held-out eval scenes/task ==="
  mkdir -p "$video_dir"
  build_retrieval "$max_episodes" "$preprocessing_dir" "$retrieval_dir"
  start_server "$retrieval_dir" "$server_log"
  run_eval "$retrieval_dir" "$video_dir" "$eval_setups"
  stop_server
  log "Finished ${name}. Outputs: ${video_dir}"
}

main() {
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    printf 'Checkpoint directory not found: %s\nSet CHECKPOINT_DIR=... if the checkpoint lives elsewhere.\n' "$CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$MAX_DISTANCE_SOURCE" ]]; then
    printf 'Max-distance source not found: %s\nSet MAX_DISTANCE_SOURCE=... if needed.\n' "$MAX_DISTANCE_SOURCE" >&2
    exit 1
  fi
  if [[ ! -x "$EVAL_PYTHON" ]]; then
    printf 'LIBERO eval Python not found or not executable: %s\nSet EVAL_PYTHON=... if needed.\n' "$EVAL_PYTHON" >&2
    exit 1
  fi
  "$EVAL_PYTHON" - <<'PY'
import robosuite  # noqa: F401
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
PY

  run_experiment "leftout10_spread40" 40 10
  run_experiment "leftout1_spread49" 49 1
}

main "$@"
