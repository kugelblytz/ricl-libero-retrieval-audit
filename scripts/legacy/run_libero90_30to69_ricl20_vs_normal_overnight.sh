#!/usr/bin/env bash
set -Eeuo pipefail

export USE_TF="${USE_TF:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OPENPI_POLICY_FLOAT32_FOR_CUDA="${OPENPI_POLICY_FLOAT32_FOR_CUDA:-1}"
export PYTHONPATH="${PWD}/src:${PWD}/packages/openpi-client/src:${PWD}/third_party/libero:${PYTHONPATH:-}"

TASK_IDS="${TASK_IDS:-30-69}"
TASK_INDICES="${TASK_INDICES:-$(seq -s ' ' 30 69)}"
NUM_SCENE_SETUPS_PER_TASK="${NUM_SCENE_SETUPS_PER_TASK:-10}"
NUM_TRIALS_PER_SCENE="${NUM_TRIALS_PER_SCENE:-1}"
SCENE_SETUP_SELECTION="${SCENE_SETUP_SELECTION:-unseen_retrieval}"

RICL_POLICY_CONFIG="${RICL_POLICY_CONFIG:-pi0_fast_libero90_ricl}"
RICL_CHECKPOINT_DIR="${RICL_CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999}"
NORMAL_POLICY_CONFIG="${NORMAL_POLICY_CONFIG:-pi0_fast_libero90_normal}"
NORMAL_CHECKPOINT_DIR="${NORMAL_CHECKPOINT_DIR:-checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999}"

REPO_ID="${REPO_ID:-local/libero90_official30to69}"
RAW_LIBERO_DIR="${RAW_LIBERO_DIR:?set RAW_LIBERO_DIR to the official LIBERO HDF5 directory}"
PREPROCESSING_DIR="${PREPROCESSING_DIR:-ricl_libero_preprocessing/libero90_official30to69_leftout10_seeded20}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-${PREPROCESSING_DIR}/retrieval}"
ASSETS_DIR="${ASSETS_DIR:-assets/pi0_fast_libero90_ricl/libero}"
MAX_DISTANCE_SOURCE="${MAX_DISTANCE_SOURCE:-ricl_libero_preprocessing/libero90_train20x20/max_distance.json}"
RETRIEVAL_EPISODES_PER_TASK="${RETRIEVAL_EPISODES_PER_TASK:-20}"
RETRIEVAL_SEED="${RETRIEVAL_SEED:-0}"

PORT="${PORT:-8000}"
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
EVAL_PYTHON="${EVAL_PYTHON:-examples/libero/.venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RICL_VIDEO_OUT_DIR="${RICL_VIDEO_OUT_DIR:-data/libero/videos/libero90_30to69_ricl20_leftout10_${RUN_TAG}}"
NORMAL_VIDEO_OUT_DIR="${NORMAL_VIDEO_OUT_DIR:-data/libero/videos/libero90_30to69_normal_leftout10_${RUN_TAG}}"
REPORT_XLSX="${REPORT_XLSX:-data/libero/reports/libero90_30to69_ricl20_vs_normal_${RUN_TAG}.xlsx}"
MANIFEST_JSON="${MANIFEST_JSON:-data/libero/reports/libero90_30to69_ricl20_vs_normal_${RUN_TAG}.manifest.json}"
SERVER_START_TIMEOUT_SECONDS="${SERVER_START_TIMEOUT_SECONDS:-900}"
SERVER_STOP_GRACE_SECONDS="${SERVER_STOP_GRACE_SECONDS:-20}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_RICL="${SKIP_RICL:-0}"
SKIP_NORMAL="${SKIP_NORMAL:-0}"

server_pid=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

on_error() {
  local exit_code=$?
  log "ERROR: command failed at line ${BASH_LINENO[0]} with exit code ${exit_code}: ${BASH_COMMAND}"
  stop_server
  exit "$exit_code"
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

port_accepts_connections() {
  "$PYTHON_BIN" - "$1" <<'PY'
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

retrieval_is_complete() {
  "$PYTHON_BIN" - "$RETRIEVAL_DIR" "$RETRIEVAL_EPISODES_PER_TASK" <<'PY'
import json
import pathlib
import sys

retrieval_dir = pathlib.Path(sys.argv[1])
max_episodes = int(sys.argv[2])
metadata_path = retrieval_dir / "retrieval_metadata.json"
if not metadata_path.exists():
    raise SystemExit(1)
metadata = json.loads(metadata_path.read_text())
if metadata.get("task_indices") != list(range(30, 70)):
    raise SystemExit(1)
if metadata.get("max_episodes_per_task") != max_episodes:
    raise SystemExit(1)
if metadata.get("episode_selection") != "seeded":
    raise SystemExit(1)
for task_id in range(30, 70):
    task_dir = retrieval_dir / f"task_{task_id:02d}"
    if not task_dir.exists():
        task_dir = retrieval_dir / f"task_{task_id}"
    if len(list(task_dir.glob("episode_*.npz"))) != max_episodes:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

dataset_is_ready() {
  "$PYTHON_BIN" - "$REPO_ID" "$TASK_INDICES" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / sys.argv[1]
expected_task_ids = [int(item) for item in sys.argv[2].split()]
info_path = root / "meta" / "info.json"
episodes_path = root / "meta" / "episodes.jsonl"
manifest_path = root / "libero_conversion_manifest.json"
if not info_path.exists() or not episodes_path.exists() or not manifest_path.exists():
    raise SystemExit(1)
info = json.loads(info_path.read_text())
manifest = json.loads(manifest_path.read_text())
if manifest.get("task_ids") != expected_task_ids:
    raise SystemExit(1)
if manifest.get("total_tasks") != len(expected_task_ids):
    raise SystemExit(1)
if info.get("total_episodes") != len(expected_task_ids) * 50:
    raise SystemExit(1)
if sum(1 for line in episodes_path.read_text().splitlines() if line.strip()) != info.get("total_episodes"):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

raw_libero90_is_ready() {
  "$PYTHON_BIN" - "$RAW_LIBERO_DIR" <<'PY'
import pathlib
import sys

raw_dir = pathlib.Path(sys.argv[1])
for folder_name in ("libero_90", "LIBERO-90", "libero-90", "LIBERO_90"):
    split_dir = raw_dir / folder_name
    if split_dir.exists() and len(list(split_dir.glob("*_demo.hdf5"))) == 90:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

build_or_verify_raw_libero90() {
  if raw_libero90_is_ready; then
    log "Raw LIBERO-90 HDF5 folder is complete under ${RAW_LIBERO_DIR}"
    return
  fi

  log "Raw LIBERO-90 HDF5 folder is incomplete; downloading missing files from Hugging Face"
  run_cmd uv run python - "$RAW_LIBERO_DIR" <<'PY'
import pathlib
import sys

from huggingface_hub import snapshot_download

raw_dir = pathlib.Path(sys.argv[1])
raw_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir=str(raw_dir),
    allow_patterns="libero_90/*",
    local_dir_use_symlinks=False,
)
print(f"Downloaded/verified LIBERO-90 files under {raw_dir / 'libero_90'}")
PY

  if [[ "$DRY_RUN" != "1" ]] && ! raw_libero90_is_ready; then
    printf 'Raw LIBERO-90 data is still incomplete under %s; expected 90 *_demo.hdf5 files.\n' "$RAW_LIBERO_DIR" >&2
    exit 1
  fi
}

build_or_verify_dataset() {
  build_or_verify_raw_libero90

  if dataset_is_ready; then
    log "LeRobot dataset is ready: ${REPO_ID}"
    return
  fi

  log "LeRobot dataset ${REPO_ID} missing or incomplete; converting official LIBERO tasks 30-69"
  run_cmd uv run python examples/libero/convert_libero_data_to_lerobot.py \
    --raw-dir "$RAW_LIBERO_DIR" \
    --repo-id "$REPO_ID" \
    --split libero_90 \
    --task-ids "$(seq -s, 30 69)" \
    --overwrite

  if [[ "$DRY_RUN" != "1" ]] && ! dataset_is_ready; then
    printf 'Dataset conversion did not produce expected %s task 30-69 dataset.\n' "$REPO_ID" >&2
    exit 1
  fi
}

build_or_verify_retrieval() {
  if retrieval_is_complete; then
    log "Retrieval memory is complete at ${RETRIEVAL_DIR}"
    return
  fi

  log "Retrieval memory missing or incomplete; building ${RETRIEVAL_EPISODES_PER_TASK} demos/task for tasks 30-69"
  if [[ "$DRY_RUN" != "1" ]]; then
    log "Cleaning serving retrieval directory only; preserving cached episodes in ${PREPROCESSING_DIR}/episodes"
    rm -rf "$RETRIEVAL_DIR"
  fi
  run_cmd uv run python scripts/setup_ricl_libero.py \
    --repo-id "$REPO_ID" \
    --task-indices $TASK_INDICES \
    --task-id-field libero_task_id \
    --num-retrieved-observations 4 \
    --retrieval-key image \
    --embedding-type 64PATCHES \
    --max-episodes-per-task "$RETRIEVAL_EPISODES_PER_TASK" \
    --episode-selection seeded \
    --seed "$RETRIEVAL_SEED" \
    --preprocessing-dir "$PREPROCESSING_DIR" \
    --retrieval-dir "$RETRIEVAL_DIR" \
    --assets-dir "$ASSETS_DIR" \
    --artifact-role retrieval \
    --max-distance-source "$MAX_DISTANCE_SOURCE" \
    --local-files-only

  if [[ "$DRY_RUN" != "1" ]] && ! retrieval_is_complete; then
    printf 'Retrieval prep did not produce the expected task 30-69 / %s episodes-per-task artifact at %s\n' \
      "$RETRIEVAL_EPISODES_PER_TASK" "$RETRIEVAL_DIR" >&2
    exit 1
  fi
}

start_ricl_server() {
  local logfile="$1"
  mkdir -p "$(dirname "$logfile")"
  log "Starting RICL policy server with demos ${RETRIEVAL_DIR}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy_ricl.py --port $PORT policy:checkpoint --policy.config $RICL_POLICY_CONFIG --policy.dir $RICL_CHECKPOINT_DIR --policy.demos-dir $RETRIEVAL_DIR > $logfile 2>&1 &"
    return
  fi
  require_port_free
  uv run python scripts/serve_policy_ricl.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$RICL_POLICY_CONFIG" \
    --policy.dir "$RICL_CHECKPOINT_DIR" \
    --policy.demos-dir "$RETRIEVAL_DIR" \
    >"$logfile" 2>&1 &
  server_pid=$!
  wait_for_server "$logfile"
}

start_normal_server() {
  local logfile="$1"
  mkdir -p "$(dirname "$logfile")"
  log "Starting normal policy server"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "+ uv run python scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config $NORMAL_POLICY_CONFIG --policy.dir $NORMAL_CHECKPOINT_DIR > $logfile 2>&1 &"
    return
  fi
  require_port_free
  uv run python scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$NORMAL_POLICY_CONFIG" \
    --policy.dir "$NORMAL_CHECKPOINT_DIR" \
    >"$logfile" 2>&1 &
  server_pid=$!
  wait_for_server "$logfile"
}

run_eval() {
  local video_dir="$1"
  local checkpoint_dir="$2"
  local retrieval_mode="$3"

  mkdir -p "$video_dir"
  run_cmd "$EVAL_PYTHON" examples/libero/main.py \
    --host "$CLIENT_HOST" \
    --port "$PORT" \
    --task-suite-name libero_90 \
    --task-ids "$TASK_IDS" \
    --num-scene-setups-per-task "$NUM_SCENE_SETUPS_PER_TASK" \
    --num-trials-per-scene "$NUM_TRIALS_PER_SCENE" \
    --scene-setup-selection "$SCENE_SETUP_SELECTION" \
    --retrieval-demos-dir "$RETRIEVAL_DIR" \
    --video-out-path "$video_dir" \
    --results-jsonl "$video_dir/results.jsonl" \
    --retrieval-log-path "$video_dir/retrieval_trace.jsonl" \
    --checkpoint-dir "$checkpoint_dir" \
    --retrieval-mode "$retrieval_mode"
}

write_manifest() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1; skipping manifest/report generation"
    return
  fi

  run_cmd uv run python scripts/legacy/export_libero90_30to69_comparison_xlsx.py \
    --ricl-results "$RICL_VIDEO_OUT_DIR/results.jsonl" \
    --normal-results "$NORMAL_VIDEO_OUT_DIR/results.jsonl" \
    --output "$REPORT_XLSX" \
    --manifest "$MANIFEST_JSON"

  run_cmd "$PYTHON_BIN" - "$MANIFEST_JSON" "$RUN_TAG" "$TASK_IDS" "$RICL_CHECKPOINT_DIR" "$NORMAL_CHECKPOINT_DIR" \
    "$RETRIEVAL_DIR" "$RICL_VIDEO_OUT_DIR" "$NORMAL_VIDEO_OUT_DIR" "$REPORT_XLSX" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
run_tag, task_ids = sys.argv[2], sys.argv[3]
ricl_checkpoint, normal_checkpoint = sys.argv[4], sys.argv[5]
retrieval_dir, ricl_dir, normal_dir, report_xlsx = sys.argv[6:10]

def summarize(path):
    rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
    return {
        "rollouts": len(rows),
        "successes": sum(bool(row.get("success")) for row in rows),
    }

manifest = {
    "run_tag": run_tag,
    "task_ids": task_ids,
    "ricl_checkpoint": ricl_checkpoint,
    "normal_checkpoint": normal_checkpoint,
    "retrieval_dir": retrieval_dir,
    "ricl_output_dir": ricl_dir,
    "normal_output_dir": normal_dir,
    "ricl": summarize(pathlib.Path(ricl_dir) / "results.jsonl"),
    "normal": summarize(pathlib.Path(normal_dir) / "results.jsonl"),
    "report_xlsx": report_xlsx,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Wrote {manifest_path}")
PY
}

validate_environment() {
  log "Validating checkpoints and environment"
  if [[ ! -d "$RICL_CHECKPOINT_DIR" ]]; then
    printf 'RICL checkpoint directory not found: %s\n' "$RICL_CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -d "$NORMAL_CHECKPOINT_DIR" ]]; then
    printf 'Normal checkpoint directory not found: %s\n' "$NORMAL_CHECKPOINT_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$MAX_DISTANCE_SOURCE" ]]; then
    printf 'Max-distance source not found: %s\n' "$MAX_DISTANCE_SOURCE" >&2
    exit 1
  fi
  if [[ ! -x "$EVAL_PYTHON" ]]; then
    printf 'LIBERO eval Python not found or not executable: %s\n' "$EVAL_PYTHON" >&2
    exit 1
  fi
  log "Checking CUDA with nvidia-smi"
  nvidia-smi >/dev/null
  log "Checking LIBERO eval imports"
  "$EVAL_PYTHON" - <<'PY'
import robosuite  # noqa: F401
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
PY
  log "Checking port ${PORT}"
  require_port_free
  log "Environment validation complete"
}

print_resume_command() {
  cat >&2 <<EOF
Normal resume command:
  JAX_PLATFORMS=cuda OPENPI_POLICY_FLOAT32_FOR_CUDA=1 SKIP_RICL=1 RUN_TAG=${RUN_TAG} \\
  RICL_VIDEO_OUT_DIR=${RICL_VIDEO_OUT_DIR} NORMAL_VIDEO_OUT_DIR=${NORMAL_VIDEO_OUT_DIR} \\
  REPORT_XLSX=${REPORT_XLSX} MANIFEST_JSON=${MANIFEST_JSON} \\
  scripts/legacy/run_libero90_30to69_ricl20_vs_normal_overnight.sh
EOF
}

main() {
  log "Run tag: ${RUN_TAG}"
  log "Task IDs: ${TASK_IDS}; retrieval demos/task: ${RETRIEVAL_EPISODES_PER_TASK}"
  log "RICL outputs: ${RICL_VIDEO_OUT_DIR}"
  log "Normal outputs: ${NORMAL_VIDEO_OUT_DIR}"
  validate_environment
  build_or_verify_dataset
  build_or_verify_retrieval

  if [[ "$SKIP_RICL" != "1" ]]; then
    start_ricl_server "$RICL_VIDEO_OUT_DIR/policy_server.log"
    run_eval "$RICL_VIDEO_OUT_DIR" "$RICL_CHECKPOINT_DIR" "nearest"
    stop_server
    log "Finished RICL eval: ${RICL_VIDEO_OUT_DIR}"
  else
    log "SKIP_RICL=1; using existing RICL results at ${RICL_VIDEO_OUT_DIR}"
  fi

  if [[ "$SKIP_NORMAL" != "1" ]]; then
    start_normal_server "$NORMAL_VIDEO_OUT_DIR/policy_server.log"
    if ! run_eval "$NORMAL_VIDEO_OUT_DIR" "$NORMAL_CHECKPOINT_DIR" "none"; then
      stop_server
      log "Normal eval failed after RICL completed. RICL outputs are preserved at ${RICL_VIDEO_OUT_DIR}"
      print_resume_command
      exit 1
    fi
    stop_server
    log "Finished normal eval: ${NORMAL_VIDEO_OUT_DIR}"
  else
    log "SKIP_NORMAL=1; using existing normal results at ${NORMAL_VIDEO_OUT_DIR}"
  fi

  write_manifest
  log "Finished overnight comparison. Report: ${REPORT_XLSX}"
}

main "$@"
