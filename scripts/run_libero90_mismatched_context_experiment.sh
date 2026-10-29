#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"

SOURCE_RETRIEVAL_DIR="${SOURCE_RETRIEVAL_DIR:-ricl_libero_preprocessing/libero90_official20to29_leftout10_spread40/retrieval}"
MISMATCH_RETRIEVAL_DIR="${MISMATCH_RETRIEVAL_DIR:-ricl_libero_preprocessing/libero90_mismatch_successful_tasks/retrieval}"
MISMATCH_MAPS="${MISMATCH_MAPS:-20:21 21:20 24:29 25:24 28:22 29:24}"
MISMATCH_MAX_EPISODES="${MISMATCH_MAX_EPISODES:-10}"

TASK_IDS="${TASK_IDS:-20,21,24,25,28,29}"
NUM_SCENE_SETUPS_PER_TASK="${NUM_SCENE_SETUPS_PER_TASK:-10}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-data/libero/videos/libero90_successful_tasks_mismatch_${RUN_TAG}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_libero90_ricl}"
PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
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

build_map_args() {
  local args=()
  for mapping in $MISMATCH_MAPS; do
    args+=(--map "$mapping")
  done
  printf '%s\0' "${args[@]}"
}

build_mismatch_db() {
  local args=(
    uv run python scripts/make_libero_mismatched_retrieval_db.py
    --source-dir "$SOURCE_RETRIEVAL_DIR"
    --output-dir "$MISMATCH_RETRIEVAL_DIR"
    --overwrite
  )
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(build_map_args)
  if [[ -n "$MISMATCH_MAX_EPISODES" ]]; then
    args+=(--max-episodes "$MISMATCH_MAX_EPISODES")
  fi
  run_cmd "${args[@]}"
}

start_server() {
  local logfile="$1"
  mkdir -p "$(dirname "$logfile")"
  log "Starting policy server with demos ${MISMATCH_RETRIEVAL_DIR}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy_ricl.py --port $PORT policy:checkpoint --policy.config $POLICY_CONFIG --policy.dir $CHECKPOINT_DIR --policy.demos-dir $MISMATCH_RETRIEVAL_DIR > $logfile 2>&1 &"
    return
  fi

  require_port_free
  uv run python scripts/serve_policy_ricl.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$CHECKPOINT_DIR" \
    --policy.demos-dir "$MISMATCH_RETRIEVAL_DIR" \
    >"$logfile" 2>&1 &
  server_pid=$!
  wait_for_server "$logfile"
}

run_eval() {
  run_cmd "$EVAL_PYTHON" examples/libero/main.py \
    --host "$CLIENT_HOST" \
    --port "$PORT" \
    --task-suite-name libero_90 \
    --task-ids "$TASK_IDS" \
    --num-scene-setups-per-task "$NUM_SCENE_SETUPS_PER_TASK" \
    --num-trials-per-scene 1 \
    --scene-setup-selection spread \
    --retrieval-demos-dir "$MISMATCH_RETRIEVAL_DIR" \
    --video-out-path "$VIDEO_OUT_DIR" \
    --retrieval-log-path "$VIDEO_OUT_DIR/retrieval_trace.jsonl"
}

main() {
  if [[ ! -d "$SOURCE_RETRIEVAL_DIR" ]]; then
    printf 'Source retrieval DB not found: %s\n' "$SOURCE_RETRIEVAL_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    printf 'Checkpoint directory not found: %s\nSet CHECKPOINT_DIR=... if needed.\n' "$CHECKPOINT_DIR" >&2
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

  mkdir -p "$VIDEO_OUT_DIR"
  build_mismatch_db
  start_server "$VIDEO_OUT_DIR/policy_server.log"
  run_eval
  stop_server
  log "Finished mismatched-context experiment. Outputs: ${VIDEO_OUT_DIR}"
}

main "$@"
