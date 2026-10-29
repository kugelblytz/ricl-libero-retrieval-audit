#!/usr/bin/env bash
set -Eeuo pipefail

EXTENSION_TASKS="21,25,28,30,31,35,44,46,50,51,55,57,59,61,68"
PRIMARY_TASKS="20,24,29,34,43,49,52,56,60,69"
POOL_TASKS="20,21,24,25,28,29,30,31,34,35,43,44,46,49,50,51,52,55,56,57,59,60,61,68,69"
CONDITIONS="dino_global dino_state_global pi_global random_global"
EXTENSION_PROTOCOL="scripts/libero90_extension15_protocol.json"
PRIMARY_PROTOCOL="scripts/libero90_final10_protocol.json"
FINAL_POOL="ricl_libero_preprocessing/libero90_solvable25_spread20/retrieval"
PI_FEATURE_DIR="data/libero/retrieval_benchmark/libero90_solvable25_offline_v1/cache/candidate/pi0_fast_first_fusion"
PRIMARY_RUN_MANIFEST="${PRIMARY_RUN_MANIFEST:-data/libero/videos/libero90_retrieval_final10_global25_disjoint_20260810_211205/run_manifest.json}"
EXTENSION_MAX_RUN_SECONDS="${EXTENSION_MAX_RUN_SECONDS:-432000}"

log() {
  printf '[extension15-retrieval] %s\n' "$*"
}

require_primary_complete() {
  [[ -f "$PRIMARY_RUN_MANIFEST" ]] || {
    log "Primary-run manifest not found: ${PRIMARY_RUN_MANIFEST}"
    exit 1
  }
  local status
  status="$(jq -r '.status' "$PRIMARY_RUN_MANIFEST")"
  [[ "$status" == "complete" ]] || {
    log "Primary 10-task run is ${status}; extension launch is gated until it is complete"
    exit 1
  }
}

validate_preparation() {
  scripts/run_libero90_final_retrieval.sh prepare
  python - \
    "$EXTENSION_PROTOCOL" "$PRIMARY_PROTOCOL" "$FINAL_POOL" "$PI_FEATURE_DIR" \
    "$EXTENSION_TASKS" "$PRIMARY_TASKS" "$POOL_TASKS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

protocol_path = Path(sys.argv[1])
primary_path = Path(sys.argv[2])
pool = Path(sys.argv[3])
pi_features = Path(sys.argv[4])
extension_tasks = [int(value) for value in sys.argv[5].split(",")]
primary_tasks = [int(value) for value in sys.argv[6].split(",")]
pool_tasks = [int(value) for value in sys.argv[7].split(",")]
protocol = json.loads(protocol_path.read_text())
primary = json.loads(primary_path.read_text())
pool_manifest = json.loads((pool / "pool_manifest.json").read_text())

if protocol.get("cohort_role") != "complete_remaining_tasks_extension":
    raise SystemExit("Extension protocol has the wrong cohort role")
if protocol.get("primary_protocol_path") != str(primary_path):
    raise SystemExit("Extension protocol points to a different primary protocol")
if primary.get("evaluation_task_ids") != primary_tasks:
    raise SystemExit("Primary task IDs differ from the frozen primary protocol")
if protocol.get("primary_evaluation_task_ids") != primary_tasks:
    raise SystemExit("Extension protocol records different primary task IDs")
if protocol.get("evaluation_task_ids") != extension_tasks:
    raise SystemExit("Extension task IDs differ from the frozen extension cohort")
if protocol.get("capability_task_ids") != pool_tasks:
    raise SystemExit("Extension capability tasks differ from the 25-task pool")
if protocol.get("retrieval_pool_task_ids") != pool_tasks:
    raise SystemExit("Extension retrieval tasks differ from the 25-task pool")
if pool_manifest.get("task_ids") != pool_tasks:
    raise SystemExit("Pool manifest task IDs differ from the frozen 25-task pool")
if pool_manifest.get("num_tasks") != 25 or pool_manifest.get("num_episodes") != 500:
    raise SystemExit("Expected the existing 25-task, 500-episode memory")
if set(primary_tasks) & set(extension_tasks):
    raise SystemExit("Primary and extension cohorts overlap")
if [task_id for task_id in pool_tasks if task_id not in primary_tasks] != extension_tasks:
    raise SystemExit("Extension is not the exact ordered complement of the primary cohort")
if set(primary_tasks) | set(extension_tasks) != set(pool_tasks):
    raise SystemExit("Primary and extension cohorts do not cover all 25 tasks")

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

artifacts = protocol["screening_artifacts"]
if artifacts != primary["screening_artifacts"]:
    raise SystemExit("Primary and extension protocols use different screening artifacts")
ricl_rows = [
    row for row in read_rows(artifacts["ricl_tasks_20_29"])
    if row.get("event_type") == "rollout_summary"
] + read_rows(artifacts["ricl_tasks_30_69"])
normal_rows = read_rows(artifacts["normal_tasks_20_29"]) + read_rows(artifacts["normal_tasks_30_69"])
nointerp_rows = read_rows(artifacts["nointerp_confirmation"])

def group_by_task(rows, task_ids):
    grouped = {task_id: [] for task_id in task_ids}
    for row in rows:
        task_id = int(row["task_id"])
        if task_id in grouped:
            grouped[task_id].append(row)
    return grouped

ricl = group_by_task(ricl_rows, pool_tasks)
normal = group_by_task(normal_rows, pool_tasks)
nointerp = group_by_task(nointerp_rows, pool_tasks)
for name, grouped in (("RICL", ricl), ("normal", normal), ("no-interpolation", nointerp)):
    bad = {task_id: len(rows) for task_id, rows in grouped.items() if len(rows) != 10}
    if bad:
        raise SystemExit(f"{name} screening artifacts do not have 10 rows per capability task: {bad}")

derived_capability = sorted(
    task_id for task_id, rows in ricl.items() if any(bool(row["success"]) for row in rows)
)
if derived_capability != pool_tasks:
    raise SystemExit("The frozen 25 tasks differ from the non-zero RICL capability set")

diagnostics = {
    "extension15_ricl_screen_successes_out_of_150": sum(
        bool(row["success"]) for task_id in extension_tasks for row in ricl[task_id]
    ),
    "extension15_normal_screen_successes_out_of_150": sum(
        bool(row["success"]) for task_id in extension_tasks for row in normal[task_id]
    ),
    "extension15_nointerp_successes_out_of_150": sum(
        bool(row["success"]) for task_id in extension_tasks for row in nointerp[task_id]
    ),
}
if protocol.get("selection_diagnostics") != diagnostics:
    raise SystemExit(f"Extension diagnostics differ from their source artifacts: {diagnostics}")

for task_id in extension_tasks:
    task = protocol["per_task"].get(str(task_id))
    if not isinstance(task, dict):
        raise SystemExit(f"Task {task_id}: missing frozen split")
    memory = set(task["memory"])
    screening = set(task["screening"])
    final = set(task["final"])
    actual_memory = {
        int(path.stem.split("_", 1)[1]) % 50
        for path in (pool / f"task_{task_id}").glob("episode_*.npz")
    }
    if actual_memory != memory:
        raise SystemExit(f"Task {task_id}: frozen memory scenes differ from the actual pool")
    scene_key = "initial_state_index" if task_id < 30 else "scene_setup_id"
    actual_screening = {int(row[scene_key]) for row in ricl[task_id]}
    if actual_screening != screening:
        raise SystemExit(f"Task {task_id}: frozen screening scenes differ from their source")
    if len(memory) != 20 or len(screening) != 10 or len(final) != 10:
        raise SystemExit(f"Task {task_id}: expected a 20/10/10 memory/screening/final split")
    if memory & screening or memory & final or screening & final:
        raise SystemExit(f"Task {task_id}: memory, screening, and final scenes overlap")
    remaining = sorted(set(range(50)) - memory - screening)
    expected_final = [remaining[index * (len(remaining) - 1) // 9] for index in range(10)]
    if task["final"] != expected_final:
        raise SystemExit(
            f"Task {task_id}: final scenes are not the declared even spread: "
            f"{task['final']} != {expected_final}"
        )
    counts = (
        sum(bool(row["success"]) for row in ricl[task_id]),
        sum(bool(row["success"]) for row in normal[task_id]),
        sum(bool(row["success"]) for row in nointerp[task_id]),
    )
    recorded = (
        task["screening_ricl_successes_out_of_10"],
        task["screening_normal_successes_out_of_10"],
        task["nointerp_confirmation_successes_out_of_10"],
    )
    if counts != recorded:
        raise SystemExit(f"Task {task_id}: recorded screening counts differ: {recorded} != {counts}")

missing_pi = [
    task_id for task_id in pool_tasks
    if not (pi_features / f"task_{task_id:02d}.npy").is_file()
]
if missing_pi:
    raise SystemExit(f"Missing Pi candidate caches: {missing_pi}")
dino_manifest = json.loads((pool / "dino_feature_cache" / "manifest.json").read_text())
if dino_manifest.get("episode_count") != 500 or dino_manifest.get("frame_count") != 69990:
    raise SystemExit(f"DINO cache differs from the frozen memory: {dino_manifest}")

print(
    "validated extension_tasks=15 primary_tasks=10 pooled_tasks=25 "
    "episodes=500 frames=69990 extension_rollouts=600 "
    f"protocol_sha256={hashlib.sha256(protocol_path.read_bytes()).hexdigest()}"
)
PY
}

validate_output() {
  local output_root="$1"
  python - \
    "$output_root" "$EXTENSION_PROTOCOL" "$EXTENSION_TASKS" "$POOL_TASKS" "$CONDITIONS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
protocol_path = Path(sys.argv[2])
evaluation_tasks = [int(value) for value in sys.argv[3].split(",")]
pool_tasks = [int(value) for value in sys.argv[4].split(",")]
conditions = sys.argv[5].split()
protocol = json.loads(protocol_path.read_text())
manifest = json.loads((root / "run_manifest.json").read_text())
expected_trials = len(evaluation_tasks) * 10

if manifest.get("status") != "complete":
    raise SystemExit(f"Extension is not complete: {manifest.get('status')}: {manifest.get('status_detail')}")
if manifest.get("task_ids") != evaluation_tasks:
    raise SystemExit("Extension manifest task IDs differ from the frozen cohort")
if manifest.get("pool_tasks") != pool_tasks or manifest.get("conditions") != conditions:
    raise SystemExit("Extension manifest pool tasks or conditions differ")
if manifest.get("expected_trials_per_condition") != expected_trials:
    raise SystemExit("Extension manifest does not expect 150 trials per method")
if manifest.get("expected_total_trials") != expected_trials * len(conditions):
    raise SystemExit("Extension manifest does not expect 600 total trials")
if manifest.get("scene_setups_per_task") != 10 or manifest.get("trials_per_scene") != 1:
    raise SystemExit("Extension manifest does not preserve 10 scenes and one trial per scene")
if manifest.get("scene_setup_selection") != "manifest":
    raise SystemExit("Extension did not use manifest-selected scenes")
if manifest.get("scene_setup_manifest_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
    raise SystemExit("Extension used a different scene protocol")
if manifest.get("action_interpolation") is not False or manifest.get("replan_steps") != 5:
    raise SystemExit("Extension does not preserve no-interpolation/replan_steps=5")
if manifest.get("retrieved_observations") != 4 or manifest.get("dino_state_beta") != 1.0:
    raise SystemExit("Extension changed retrieval count or DINO-state fusion weight")
if manifest.get("pi_retriever_dtype") != "float32":
    raise SystemExit("Extension did not use the verified float32 Pi feature path")

method_by_condition = {
    "dino_global": "dino",
    "dino_state_global": "dino_state",
    "pi_global": "pi0_first_fusion",
    "random_global": "random",
}

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

expected_identities = {
    (task_id, scene_id, trial_id)
    for task_id in evaluation_tasks
    for trial_id, scene_id in enumerate(protocol["per_task"][str(task_id)]["final"])
}
identity_sets = {}
for condition in conditions:
    results = read_rows(root / condition / "results.jsonl")
    traces = read_rows(root / condition / "retrieval_trace.jsonl")
    valid_results = {
        (int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])): row
        for row in results if row.get("error") is None
    }
    valid_traces = {
        (int(row["task_id"]), int(row["scene_setup_id"]), int(row["trial_id"])): row
        for row in traces if row.get("error") is None
    }
    if set(valid_results) != expected_identities or set(valid_traces) != expected_identities:
        raise SystemExit(
            f"{condition}: valid result/trace identities differ from the 150 frozen extension trials"
        )
    identity_sets[condition] = set(valid_results)
    for identity, row in valid_traces.items():
        items = row.get("retrieval_trace", [])
        if not items:
            raise SystemExit(f"{condition}: empty retrieval trace for {identity}")
        for item in items:
            if item.get("retrieval_method") != method_by_condition[condition]:
                raise SystemExit(f"{condition}: incorrect retrieval method at {identity}")
            if item.get("retrieval_scope") != "global":
                raise SystemExit(f"{condition}: non-global retrieval at {identity}")
            if item.get("candidate_pool_num_tasks") != 25:
                raise SystemExit(f"{condition}: trace does not report the 25-task memory at {identity}")
            if item.get("candidate_pool_num_frames") != 69990:
                raise SystemExit(f"{condition}: trace does not report the 69,990-frame memory at {identity}")
            if item.get("retrieved_task_id") not in pool_tasks:
                raise SystemExit(f"{condition}: retrieved a task outside the frozen memory at {identity}")
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
                        f"random_global: invalid deterministic sample at {identity}, step {policy_step}"
                    )

if any(identities != identity_sets[conditions[0]] for identities in identity_sets.values()):
    raise SystemExit("Extension methods do not share identical trial identities")
equivalence = json.loads((root / "pi_first_fusion_equivalence.json").read_text())
if not equivalence.get("task_top4_equal") or not equivalence.get("global_top4_equal"):
    raise SystemExit(f"Pi feature equivalence failed: {equivalence}")
print("validated extension conditions=4 trials_per_condition=150 total=600")
PY
}

launch() {
  require_primary_complete
  validate_preparation
  local default_tag="extension15_global25_disjoint_$(date +%Y%m%d_%H%M%S)"
  local tag="${RUN_TAG:-$default_tag}"
  local output_root="${OUTPUT_ROOT:-data/libero/videos/libero90_retrieval_${tag}}"
  mkdir -p "$output_root"
  log "Launching frozen extension tag=${tag} output=${output_root}"
  TASK_IDS="$EXTENSION_TASKS" \
  POOL_TASK_IDS="$POOL_TASKS" \
  POOL_DIR="$FINAL_POOL" \
  SCENE_SETUP_SELECTION="manifest" \
  SCENE_SETUP_MANIFEST_PATH="$EXTENSION_PROTOCOL" \
  NUM_SCENE_SETUPS_PER_TASK="10" \
  CONDITIONS="$CONDITIONS" \
  DINO_STATE_BETA="1.0" \
  PI_RETRIEVER_DTYPE="float32" \
  RETRIEVAL_SEARCH_DEVICE="auto" \
  RETRIEVAL_DISTANCE_CHUNK_SIZE="256" \
  MAX_RUN_SECONDS="$EXTENSION_MAX_RUN_SECONDS" \
  MAX_CONDITION_RESTARTS="2" \
  INFERENCE_TIMEOUT_SECONDS="180" \
  MAX_ROLLOUT_SECONDS="900" \
  MAX_VRAM_MIB="32500" \
  RUN_TAG="$tag" \
  OUTPUT_ROOT="$output_root" \
    scripts/run_libero90_pi_vs_dino_context.sh 2>&1 | tee "$output_root/launcher.log"
  validate_output "$output_root"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_libero90_extension15_retrieval.sh prepare|run|validate

  prepare   Validate the frozen 15-task extension, disjoint scenes, 25-task memory, and caches.
            This starts no policy server and runs no rollouts.
  run       After the primary 10-task manifest is complete, run 15 tasks x 10 scenes x 4 methods.
  validate  Validate a completed extension specified by EXTENSION_OUT.

Set RUN_TAG and OUTPUT_ROOT to choose or resume an extension output directory.
Set PRIMARY_RUN_MANIFEST only if the primary 10-task output is stored elsewhere.
EOF
}

case "${1:-}" in
  prepare)
    validate_preparation
    ;;
  run)
    launch
    ;;
  validate)
    [[ -n "${EXTENSION_OUT:-}" ]] || { log "validate requires EXTENSION_OUT"; exit 1; }
    validate_preparation
    validate_output "$EXTENSION_OUT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
