#!/usr/bin/env bash
set -Eeuo pipefail

EVAL_TASKS="20,24,29,34,43,49,52,56,60,69"
POOL_TASKS="20,21,24,25,28,29,30,31,34,35,43,44,46,49,50,51,52,55,56,57,59,60,61,68,69"
FINAL_PROTOCOL="scripts/libero90_final10_protocol.json"
FINAL_STAGE="ricl_libero_preprocessing/libero90_solvable25_spread20_source"
FINAL_POOL="ricl_libero_preprocessing/libero90_solvable25_spread20/retrieval"
SOURCE_20_29="ricl_libero_preprocessing/libero90_official20to29_spread20/retrieval"
SOURCE_30_69="ricl_libero_preprocessing/libero90_official30to69_leftout10_seeded20/retrieval"
PI_FEATURE_DIR="data/libero/retrieval_benchmark/libero90_solvable25_offline_v1/cache/candidate/pi0_fast_first_fusion"
PILOT_MANIFEST="data/libero/videos/libero90_pi_vs_dino_context_full_20260810/run_manifest.json"
FINAL_CONDITIONS="dino_global dino_state_global pi_global random_global"
ALLOW_INCOMPLETE_PILOT="${ALLOW_INCOMPLETE_PILOT:-0}"

log() {
  printf '[final-retrieval] %s\n' "$*"
}

prepare_stage() {
  mkdir -p "$FINAL_STAGE"
  local task_id
  local source_root
  local source
  local target
  for task_id in ${POOL_TASKS//,/ }; do
    if (( task_id < 30 )); then
      source_root="$SOURCE_20_29"
    else
      source_root="$SOURCE_30_69"
    fi
    source="$(realpath "$source_root/task_$task_id")"
    target="$FINAL_STAGE/task_$task_id"
    if [[ -L "$target" ]]; then
      [[ "$(realpath "$target")" == "$source" ]] || {
        log "Existing stage link points to the wrong source: ${target}"
        exit 1
      }
    elif [[ -e "$target" ]]; then
      log "Existing stage entry is not a symlink: ${target}"
      exit 1
    else
      ln -s "$source" "$target"
    fi
  done

  if [[ ! -f "$FINAL_STAGE/retrieval_metadata.json" ]]; then
    python - "$FINAL_STAGE/retrieval_metadata.json" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "backend": "dinov2",
            "retrieval_key": "image",
            "embedding_type": "64PATCHES",
        },
        indent=2,
    )
    + "\n"
)
PY
  fi
}

validate_pool() {
  python - "$FINAL_POOL" "$POOL_TASKS" "$PI_FEATURE_DIR" "$FINAL_PROTOCOL" "$EVAL_TASKS" <<'PY'
import json
from pathlib import Path
import sys

pool = Path(sys.argv[1])
expected_tasks = [int(value) for value in sys.argv[2].split(",")]
pi_features = Path(sys.argv[3])
protocol_path = Path(sys.argv[4])
evaluation_tasks = [int(value) for value in sys.argv[5].split(",")]
manifest = json.loads((pool / "pool_manifest.json").read_text())
protocol = json.loads(protocol_path.read_text())
if manifest["task_ids"] != expected_tasks:
    raise SystemExit(f"Pool task IDs differ: {manifest['task_ids']} != {expected_tasks}")
if manifest["num_tasks"] != 25 or manifest["num_episodes"] != 500:
    raise SystemExit(
        f"Expected a 25-task/500-episode pool, got {manifest['num_tasks']} tasks and "
        f"{manifest['num_episodes']} episodes"
    )
if protocol.get("retrieval_pool_task_ids") != expected_tasks:
    raise SystemExit("Protocol retrieval-pool tasks differ from the frozen 25-task pool")
if protocol.get("evaluation_task_ids") != evaluation_tasks:
    raise SystemExit("Protocol evaluation tasks differ from the frozen 10-task subset")
if len(evaluation_tasks) != 10 or not set(evaluation_tasks).issubset(expected_tasks):
    raise SystemExit(f"Invalid 10-task evaluation subset: {evaluation_tasks}")

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

artifacts = protocol["screening_artifacts"]
ricl_rows = [
    row
    for row in read_rows(artifacts["ricl_tasks_20_29"])
    if row.get("event_type") == "rollout_summary"
] + read_rows(artifacts["ricl_tasks_30_69"])
ricl_by_task = {task_id: [] for task_id in range(20, 70)}
for row in ricl_rows:
    ricl_by_task[int(row["task_id"])].append(row)
if any(len(rows) != 10 for rows in ricl_by_task.values()):
    raise SystemExit("Capability-screen artifacts do not contain exactly 10 RICL rollouts for every task 20-69")
derived_capability_tasks = sorted(
    task_id for task_id, rows in ricl_by_task.items() if any(bool(row["success"]) for row in rows)
)
if derived_capability_tasks != expected_tasks or protocol.get("capability_task_ids") != expected_tasks:
    raise SystemExit("Frozen capability tasks do not equal the non-zero RICL screening tasks")
positions = protocol.get("evaluation_task_positions")
expected_positions = [index * (len(expected_tasks) - 1) // (len(evaluation_tasks) - 1) for index in range(len(evaluation_tasks))]
if positions != expected_positions:
    raise SystemExit(f"Frozen evaluation positions are not the declared even spread: {positions} != {expected_positions}")
if [expected_tasks[position] for position in positions] != evaluation_tasks:
    raise SystemExit("Frozen 10-task subset does not follow the declared evenly spaced positions")

normal_rows = read_rows(artifacts["normal_tasks_20_29"]) + read_rows(artifacts["normal_tasks_30_69"])
normal_by_task = {task_id: [] for task_id in range(20, 70)}
for row in normal_rows:
    normal_by_task[int(row["task_id"])].append(row)
if any(len(rows) != 10 for rows in normal_by_task.values()):
    raise SystemExit("Normal-policy artifacts do not contain exactly 10 rollouts for every task 20-69")

nointerp_by_task = {task_id: [] for task_id in expected_tasks}
for row in read_rows(artifacts["nointerp_confirmation"]):
    nointerp_by_task[int(row["task_id"])].append(row)
if any(len(rows) != 10 for rows in nointerp_by_task.values()):
    raise SystemExit("No-interpolation confirmation does not contain exactly 10 rollouts per capability task")
diagnostics = protocol["selection_diagnostics"]
derived_diagnostics = {
    "capability25_ricl_screen_successes_out_of_250": sum(
        bool(row["success"]) for task_id in expected_tasks for row in ricl_by_task[task_id]
    ),
    "evaluation10_ricl_screen_successes_out_of_100": sum(
        bool(row["success"]) for task_id in evaluation_tasks for row in ricl_by_task[task_id]
    ),
    "capability25_normal_screen_successes_out_of_250": sum(
        bool(row["success"]) for task_id in expected_tasks for row in normal_by_task[task_id]
    ),
    "evaluation10_normal_screen_successes_out_of_100": sum(
        bool(row["success"]) for task_id in evaluation_tasks for row in normal_by_task[task_id]
    ),
    "capability25_nointerp_successes_out_of_250": sum(
        bool(row["success"]) for task_id in expected_tasks for row in nointerp_by_task[task_id]
    ),
    "evaluation10_nointerp_successes_out_of_100": sum(
        bool(row["success"]) for task_id in evaluation_tasks for row in nointerp_by_task[task_id]
    ),
}
if diagnostics != derived_diagnostics:
    raise SystemExit(f"Protocol selection diagnostics differ from source artifacts: {derived_diagnostics}")

bad_tasks = {
    task_id: manifest["tasks"][str(task_id)]["num_episodes"]
    for task_id in expected_tasks
    if manifest["tasks"][str(task_id)]["num_episodes"] != 20
}
if bad_tasks:
    raise SystemExit(f"Pool tasks without exactly 20 episodes: {bad_tasks}")
for task_id in evaluation_tasks:
    split = protocol["per_task"][str(task_id)]
    memory = set(split["memory"])
    screening = set(split["screening"])
    final = set(split["final"])
    actual_memory = {
        int(path.stem.split("_", 1)[1]) % 50
        for path in (pool / f"task_{task_id}").glob("episode_*.npz")
    }
    if actual_memory != memory:
        raise SystemExit(f"Task {task_id}: protocol memory scenes differ from the actual pool")
    if len(memory) != 20 or len(screening) != 10 or len(final) != 10:
        raise SystemExit(f"Task {task_id}: expected a 20/10/10 memory/screening/final split")
    if memory & screening or memory & final or screening & final:
        raise SystemExit(f"Task {task_id}: memory, screening, and final scenes are not disjoint")
    remaining = sorted(set(range(50)) - memory - screening)
    spread_positions = [index * (len(remaining) - 1) // (len(final) - 1) for index in range(len(final))]
    expected_final = [remaining[position] for position in spread_positions]
    if split["final"] != expected_final:
        raise SystemExit(
            f"Task {task_id}: frozen final scenes are not the declared even spread: "
            f"{split['final']} != {expected_final}"
        )
    screening_rows = ricl_by_task[task_id]
    scene_key = "initial_state_index" if task_id < 30 else "scene_setup_id"
    if {int(row[scene_key]) for row in screening_rows} != screening:
        raise SystemExit(f"Task {task_id}: protocol screening scenes differ from the source artifact")
    ricl_successes = sum(bool(row["success"]) for row in screening_rows)
    normal_successes = sum(bool(row["success"]) for row in normal_by_task[task_id])
    nointerp_successes = sum(bool(row["success"]) for row in nointerp_by_task[task_id])
    if ricl_successes != split["screening_ricl_successes_out_of_10"]:
        raise SystemExit(f"Task {task_id}: protocol RICL screening count differs from its source")
    if normal_successes != split["screening_normal_successes_out_of_10"]:
        raise SystemExit(f"Task {task_id}: protocol normal screening count differs from its source")
    if nointerp_successes != split["nointerp_confirmation_successes_out_of_10"]:
        raise SystemExit(f"Task {task_id}: protocol no-interpolation count differs from its source")
missing_pi = [task_id for task_id in expected_tasks if not (pi_features / f"task_{task_id:02d}.npy").is_file()]
if missing_pi:
    raise SystemExit(f"Missing Pi first-fusion candidate caches for tasks: {missing_pi}")
dino_features = pool / "dino_feature_cache" / "features.npy"
dino_manifest = pool / "dino_feature_cache" / "manifest.json"
if not dino_features.is_file() or not dino_manifest.is_file():
    raise SystemExit(f"Missing complete DINO memory-mapped cache under {dino_features.parent}")
cache = json.loads(dino_manifest.read_text())
if cache["episode_count"] != 500 or cache["shape"][0] != cache["frame_count"]:
    raise SystemExit(f"Invalid DINO cache manifest: {cache}")
print(
    f"validated tasks={manifest['num_tasks']} episodes={manifest['num_episodes']} "
    f"dino_frames={cache['frame_count']} dino_dim={cache['feature_dim']}"
)
PY
}

prepare_pool() {
  prepare_stage
  if [[ ! -f "$FINAL_POOL/pool_manifest.json" ]]; then
    log "Materializing the 500-episode final pool with hardlinks"
    uv run python scripts/make_libero_retrieval_pool.py \
      --source-dir "$FINAL_STAGE" \
      --output-dir "$FINAL_POOL" \
      --task-ids "$POOL_TASKS" \
      --materialization hardlink \
      --max-episodes-per-task 20
  fi
  if [[ ! -f "$FINAL_POOL/dino_feature_cache/features.npy" && ! -f "$FINAL_POOL/dino_feature_cache/manifest.json" ]]; then
    log "Building the final memory-mapped DINO cache"
    uv run python scripts/export_libero_dino_feature_cache.py --demos-dir "$FINAL_POOL"
  fi
  validate_pool
}

require_pilot_complete() {
  [[ -f "$PILOT_MANIFEST" ]] || {
    log "Pilot manifest not found: ${PILOT_MANIFEST}"
    exit 1
  }
  local pilot_status
  pilot_status="$(jq -r '.status' "$PILOT_MANIFEST")"
  if [[ "$pilot_status" == "complete" ]]; then
    return
  fi
  if [[ "$ALLOW_INCOMPLETE_PILOT" == "1" ]]; then
    log "Pilot is ${pilot_status}; continuing because ALLOW_INCOMPLETE_PILOT=1"
    return
  fi
  log "Pilot is ${pilot_status}; wait for it to complete before using port 8000"
  exit 1
}

validate_output() {
  local output_root="$1"
  local expected_task_ids="$2"
  local expected_trials="$3"
  local require_zero_error_rows="$4"
  python - \
    "$output_root" "$expected_task_ids" "$POOL_TASKS" "$FINAL_CONDITIONS" \
    "$expected_trials" "$require_zero_error_rows" "$FINAL_PROTOCOL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
evaluation_tasks = [int(value) for value in sys.argv[2].split(",")]
pool_tasks = [int(value) for value in sys.argv[3].split(",")]
conditions = sys.argv[4].split()
expected_trials = int(sys.argv[5])
require_zero_error_rows = sys.argv[6] == "1"
protocol_path = Path(sys.argv[7])
protocol = json.loads(protocol_path.read_text())
manifest = json.loads((root / "run_manifest.json").read_text())
if manifest.get("status") != "complete":
    raise SystemExit(f"Run is not complete: {manifest.get('status')}: {manifest.get('status_detail')}")
if manifest.get("task_ids") != evaluation_tasks:
    raise SystemExit("Manifest evaluation tasks differ from the frozen protocol")
if manifest.get("pool_tasks") != pool_tasks or manifest.get("conditions") != conditions:
    raise SystemExit("Manifest pool tasks or conditions differ from the frozen protocol")
if manifest.get("scene_setup_selection") != "manifest":
    raise SystemExit("Final run did not use the frozen manifest scene split")
if manifest.get("scene_setup_manifest_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit("Final run used a different scene-setup manifest")
if manifest.get("expected_trials_per_condition") != expected_trials:
    raise SystemExit(
        f"Expected {expected_trials} trials per condition, manifest has "
        f"{manifest.get('expected_trials_per_condition')}"
    )
if manifest.get("expected_total_trials") != expected_trials * len(conditions):
    raise SystemExit(f"Incorrect expected total in manifest: {manifest.get('expected_total_trials')}")
if manifest.get("action_interpolation") is not False or manifest.get("replan_steps") != 5:
    raise SystemExit("Manifest does not preserve no-interpolation/replan_steps=5")

method_by_condition = {
    "dino_global": "dino",
    "dino_state_global": "dino_state",
    "pi_global": "pi0_first_fusion",
    "random_global": "random",
}

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

identity_sets = {}
if expected_trials % len(evaluation_tasks):
    raise SystemExit("Expected trial count is not divisible by the evaluation-task count")
scenes_per_task = expected_trials // len(evaluation_tasks)
expected_identities = {
    (task_id, scene_id, trial_id)
    for task_id in evaluation_tasks
    for trial_id, scene_id in enumerate(protocol["per_task"][str(task_id)]["final"][:scenes_per_task])
}
for condition in conditions:
    result_rows = read_rows(root / condition / "results.jsonl")
    trace_rows = read_rows(root / condition / "retrieval_trace.jsonl")
    error_rows = [row for row in result_rows if row.get("error") is not None]
    if require_zero_error_rows and error_rows:
        raise SystemExit(f"{condition}: smoke test contains {len(error_rows)} infrastructure-error rows")
    valid_results = {
        (int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])): row
        for row in result_rows
        if row.get("error") is None
    }
    valid_traces = {
        (int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])): row
        for row in trace_rows
        if row.get("error") is None
    }
    if len(valid_results) != expected_trials or set(valid_results) != set(valid_traces):
        raise SystemExit(
            f"{condition}: expected {expected_trials} aligned canonical results/traces; "
            f"got {len(valid_results)} results and {len(valid_traces)} traces"
        )
    if set(valid_results) != expected_identities:
        raise SystemExit(f"{condition}: task/scene identities differ from the frozen final split")
    identity_sets[condition] = set(valid_results)
    for identity, row in valid_traces.items():
        items = row.get("retrieval_trace", [])
        if not items:
            raise SystemExit(f"{condition}: empty retrieval trace for {identity}")
        for item in items:
            if item.get("retrieval_method") != method_by_condition[condition]:
                raise SystemExit(f"{condition}: incorrect retrieval label in {identity}: {item}")
            if item.get("retrieval_scope") != "global" or item.get("candidate_pool_num_tasks") != len(pool_tasks):
                raise SystemExit(f"{condition}: trace does not report the 25-task global pool: {item}")
        if condition == "random_global":
            by_step = {}
            for item in items:
                by_step.setdefault(int(item["policy_step"]), []).append(item)
            for policy_step, step_items in by_step.items():
                selected = {
                    (item["retrieved_task_id"], item["episode_id"], item["frame_index"])
                    for item in step_items
                }
                seeds = {item.get("random_query_seed") for item in step_items}
                if len(step_items) != 4 or len(selected) != 4 or len(seeds) != 1 or None in seeds:
                    raise SystemExit(
                        f"random_global: invalid deterministic four-frame sample at {identity}, step {policy_step}"
                    )

first_identities = identity_sets[conditions[0]]
if any(identities != first_identities for identities in identity_sets.values()):
    raise SystemExit("Conditions do not share identical task/scene/trial identities")

equivalence = json.loads((root / "pi_first_fusion_equivalence.json").read_text())
if not equivalence.get("task_top4_equal") or not equivalence.get("global_top4_equal"):
    raise SystemExit(f"Pi first-fusion equivalence failed: {equivalence}")
print(
    f"validated conditions={len(conditions)} canonical_trials_per_condition={expected_trials} "
    f"total={expected_trials * len(conditions)}"
)
PY
}

launch() {
  local mode="$1"
  local default_tag
  local tag
  local output_root
  local task_ids
  local scene_count
  local max_run_seconds

  require_pilot_complete
  prepare_pool
  if [[ "$mode" == "smoke" ]]; then
    default_tag="final10_global25_disjoint_smoke_$(date +%Y%m%d_%H%M%S)"
    task_ids="20"
    scene_count="1"
    max_run_seconds="21600"
  else
    default_tag="final10_global25_disjoint_$(date +%Y%m%d_%H%M%S)"
    task_ids="$EVAL_TASKS"
    scene_count="10"
    max_run_seconds="259200"
  fi
  tag="${RUN_TAG:-$default_tag}"
  output_root="${OUTPUT_ROOT:-data/libero/videos/libero90_retrieval_${tag}}"
  mkdir -p "$output_root"
  log "Launching mode=${mode} tag=${tag} output=${output_root}"
  TASK_IDS="$task_ids" \
  POOL_TASK_IDS="$POOL_TASKS" \
  POOL_DIR="$FINAL_POOL" \
  SCENE_SETUP_SELECTION="manifest" \
  SCENE_SETUP_MANIFEST_PATH="$FINAL_PROTOCOL" \
  NUM_SCENE_SETUPS_PER_TASK="$scene_count" \
  CONDITIONS="$FINAL_CONDITIONS" \
  DINO_STATE_BETA="1.0" \
  PI_RETRIEVER_DTYPE="float32" \
  RETRIEVAL_SEARCH_DEVICE="auto" \
  RETRIEVAL_DISTANCE_CHUNK_SIZE="256" \
  MAX_RUN_SECONDS="$max_run_seconds" \
  MAX_CONDITION_RESTARTS="2" \
  INFERENCE_TIMEOUT_SECONDS="180" \
  MAX_ROLLOUT_SECONDS="900" \
  MAX_VRAM_MIB="32500" \
  RUN_TAG="$tag" \
  OUTPUT_ROOT="$output_root" \
    scripts/run_libero90_pi_vs_dino_context.sh 2>&1 | tee "$output_root/launcher.log"
  if [[ "$mode" == "smoke" ]]; then
    validate_output "$output_root" "20" 1 1
  else
    validate_output "$output_root" "$EVAL_TASKS" 100 0
  fi
}

usage() {
  cat <<'EOF'
Usage: scripts/run_libero90_final_retrieval.sh prepare|smoke|run|validate

  prepare   Validate the 25-task memory, 10-task subset, disjoint split, and caches; run no rollouts.
  smoke     Run the mandatory four-rollout smoke test after the pilot completes.
  run       Run the final 400-rollout experiment: 10 tasks x 10 scenes x 4 methods.
  validate  Validate FINAL_OUT and regenerate all summaries/statistical figures.

Set RUN_TAG and OUTPUT_ROOT to resume an interrupted smoke or final run.
Set FINAL_OUT when using validate.
EOF
}

case "${1:-}" in
  prepare)
    prepare_pool
    ;;
  smoke)
    launch smoke
    ;;
  run)
    launch run
    ;;
  validate)
    [[ -n "${FINAL_OUT:-}" ]] || { log "validate requires FINAL_OUT"; exit 1; }
    validate_pool
    validate_output "$FINAL_OUT" "$EVAL_TASKS" 100 0
    uv run python scripts/summarize_libero_pi_vs_dino.py --output-root "$FINAL_OUT"
    jq '{status,status_detail,task_ids,pool_tasks,scene_setups_per_task,conditions,expected_trials_per_condition,expected_total_trials}' \
      "$FINAL_OUT/run_manifest.json"
    ;;
  *)
    usage
    exit 2
    ;;
esac
