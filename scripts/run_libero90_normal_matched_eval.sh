#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"

POLICY_CONFIG="${POLICY_CONFIG:-pi0_fast_libero90_normal}"
CHECKPOINT_RUN_DIR="${CHECKPOINT_RUN_DIR:-checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

TASK_IDS="${TASK_IDS:-20-29}"
NUM_SCENE_SETUPS_PER_TASK="${NUM_SCENE_SETUPS_PER_TASK:-10}"
NUM_TRIALS_PER_SCENE="${NUM_TRIALS_PER_SCENE:-1}"
SCENE_SETUP_SELECTION="${SCENE_SETUP_SELECTION:-unseen_retrieval}"
RETRIEVAL_DEMOS_DIR="${RETRIEVAL_DEMOS_DIR:-ricl_libero_preprocessing/libero90_official20to29_leftout10_spread40/retrieval}"

PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VIDEO_OUT_DIR="${VIDEO_OUT_DIR:-data/libero/videos/libero90_normal_leftout10_spread40_${RUN_TAG}}"
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

resolve_checkpoint_dir() {
  if [[ -n "$CHECKPOINT_DIR" ]]; then
    printf '%s\n' "$CHECKPOINT_DIR"
    return
  fi

  if [[ -d "$CHECKPOINT_RUN_DIR/9999" ]]; then
    printf '%s\n' "$CHECKPOINT_RUN_DIR/9999"
    return
  fi

  python - "$CHECKPOINT_RUN_DIR" <<'PY'
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
if not run_dir.exists():
    raise SystemExit(1)
numeric = sorted(
    (path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()),
    key=lambda path: int(path.name),
)
if not numeric:
    raise SystemExit(1)
print(numeric[-1])
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

start_server() {
  local checkpoint_dir="$1"
  local logfile="$2"

  mkdir -p "$(dirname "$logfile")"
  log "Starting normal policy server from ${checkpoint_dir}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config $POLICY_CONFIG --policy.dir $checkpoint_dir > $logfile 2>&1 &"
    return
  fi

  require_port_free
  uv run python scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$POLICY_CONFIG" \
    --policy.dir "$checkpoint_dir" \
    >"$logfile" 2>&1 &
  server_pid=$!
  wait_for_server "$logfile"
}

main() {
  local checkpoint_dir
  if ! checkpoint_dir="$(resolve_checkpoint_dir)"; then
    printf 'Could not resolve checkpoint. Set CHECKPOINT_DIR=... or CHECKPOINT_RUN_DIR=...\n' >&2
    exit 1
  fi

  if [[ ! -d "$checkpoint_dir" ]]; then
    printf 'Checkpoint directory not found: %s\n' "$checkpoint_dir" >&2
    exit 1
  fi
  if [[ ! -d "$RETRIEVAL_DEMOS_DIR" ]]; then
    printf 'Retrieval demo directory not found: %s\nUsed only for scene selection; set RETRIEVAL_DEMOS_DIR=... if needed.\n' "$RETRIEVAL_DEMOS_DIR" >&2
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
  log "Resolved checkpoint: ${checkpoint_dir}"
  log "Writing outputs to: ${VIDEO_OUT_DIR}"

  start_server "$checkpoint_dir" "$VIDEO_OUT_DIR/policy_server.log"
  run_cmd "$EVAL_PYTHON" examples/libero/main.py \
    --host "$CLIENT_HOST" \
    --port "$PORT" \
    --task-suite-name libero_90 \
    --task-ids "$TASK_IDS" \
    --num-scene-setups-per-task "$NUM_SCENE_SETUPS_PER_TASK" \
    --num-trials-per-scene "$NUM_TRIALS_PER_SCENE" \
    --scene-setup-selection "$SCENE_SETUP_SELECTION" \
    --retrieval-demos-dir "$RETRIEVAL_DEMOS_DIR" \
    --video-out-path "$VIDEO_OUT_DIR" \
    --results-jsonl "$VIDEO_OUT_DIR/results.jsonl" \
    --retrieval-log-path "$VIDEO_OUT_DIR/retrieval_trace.jsonl" \
    --checkpoint-dir "$checkpoint_dir" \
    --retrieval-mode none
  stop_server
  log "Finished normal matched LIBERO-90 eval. Outputs: ${VIDEO_OUT_DIR}"
}

main "$@"
