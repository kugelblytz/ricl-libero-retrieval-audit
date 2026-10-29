from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Any, Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
SceneSetupSelection = Literal["first", "spread", "unseen_retrieval", "manifest"]


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    inference_timeout_seconds: float = 180.0
    max_rollout_seconds: float = 900.0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    num_scene_setups_per_task: int | None = None  # Number of initial-state scene setups per task.
    num_trials_per_scene: int = 1  # Number of rollouts for each selected scene setup.
    scene_setup_selection: SceneSetupSelection = "first"
    task_ids: str | None = None  # Comma-separated task ids. If unset, evaluates the whole suite.
    retrieval_demos_dir: str | None = None  # Used only for scene selection, never passed to the policy.
    scene_setup_manifest_path: str | None = None  # JSON mapping task IDs to fixed evaluation scene indices.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_jsonl: str | None = None  # Optional path for per-trial JSONL results.
    retrieval_log_path: str | None = None  # Optional compatibility log path for RICL scene-selection runs.
    checkpoint_dir: str | None = None  # Recorded in results only.
    retrieval_mode: str = "nearest"  # Recorded in results only.
    resume: bool = False  # Skip error-free trials already present in results_jsonl.
    fail_on_error: bool = False  # Exit nonzero after recording an infrastructure error.

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if args.inference_timeout_seconds <= 0:
        raise ValueError("--inference-timeout-seconds must be positive.")
    if args.max_rollout_seconds <= 0:
        raise ValueError("--max-rollout-seconds must be positive.")
    if args.resume and args.results_jsonl is None:
        raise ValueError("--resume requires --results-jsonl.")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.results_jsonl is not None:
        pathlib.Path(args.results_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if args.retrieval_log_path is not None:
        retrieval_log_path = pathlib.Path(args.retrieval_log_path)
        retrieval_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            retrieval_log_path.write_text("")

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(
        args.host,
        args.port,
        inference_timeout_seconds=args.inference_timeout_seconds,
    )
    completed_trials = _load_completed_trials(pathlib.Path(args.results_jsonl)) if args.resume else {}

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_ids = _parse_task_ids(args.task_ids, num_tasks_in_suite)
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        scene_trial_indices = _select_scene_trial_indices(
            task_id=task_id,
            initial_states=initial_states,
            num_trials_per_task=args.num_trials_per_task,
            num_scene_setups_per_task=args.num_scene_setups_per_task,
            num_trials_per_scene=args.num_trials_per_scene,
            scene_setup_selection=args.scene_setup_selection,
            retrieval_demos_dir=args.retrieval_demos_dir,
            scene_setup_manifest_path=args.scene_setup_manifest_path,
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for rollout_idx, scene_idx in enumerate(tqdm.tqdm(scene_trial_indices)):
            trial_key = (args.retrieval_mode, task_id, scene_idx, rollout_idx)
            if trial_key in completed_trials:
                previous = completed_trials[trial_key]
                task_episodes += 1
                total_episodes += 1
                if previous["success"]:
                    task_successes += 1
                    total_successes += 1
                logging.info(
                    "Skipping completed trial: mode=%s task=%d scene=%d trial=%d",
                    args.retrieval_mode,
                    task_id,
                    scene_idx,
                    rollout_idx,
                )
                continue
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[scene_idx])

            # Setup
            t = 0
            done = False
            error = None
            replay_images = []
            video_path = None
            rollout_retrieval_trace = []
            rollout_timing_samples = []
            rollout_started = time.monotonic()

            logging.info(f"Starting episode {task_episodes + 1} with scene setup {scene_idx}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if time.monotonic() - rollout_started > args.max_rollout_seconds:
                        raise TimeoutError(
                            f"Rollout exceeded {args.max_rollout_seconds:.0f} seconds " f"at environment step {t}."
                        )
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "task_index": task_id,
                        }

                        # Query model to get action
                        response = client.infer(element)
                        if isinstance(response.get("timings_ms"), dict):
                            rollout_timing_samples.append(response["timings_ms"])
                        rollout_retrieval_trace.extend(
                            {"policy_step": t, **retrieval_item}
                            for retrieval_item in response.get("retrieval_trace", [])
                        )
                        action_chunk = response.get("actions", response.get("query_actions"))
                        if action_chunk is None:
                            raise KeyError(f"Policy response is missing actions/query_actions: {response.keys()}")
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    logging.exception(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / (
                f"rollout_task_{task_id:02d}_scene_{scene_idx:03d}_trial_{rollout_idx:03d}_{args.retrieval_mode}_{task_segment}_{suffix}.mp4"
            )
            if replay_images:
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            else:
                video_path = None

            _write_trial_result(
                args,
                task_id=task_id,
                task_description=task_description,
                trial_id=rollout_idx,
                scene_setup_id=scene_idx,
                success=done,
                elapsed_steps=t,
                video_path=video_path,
                error=error,
                wall_seconds=time.monotonic() - rollout_started,
                timing_summary=_summarize_timings(rollout_timing_samples),
            )
            _write_retrieval_compat_log(
                args,
                task_id=task_id,
                task_description=task_description,
                trial_id=rollout_idx,
                scene_setup_id=scene_idx,
                success=done,
                elapsed_steps=t,
                error=error,
                retrieval_trace=rollout_retrieval_trace,
                timing_summary=_summarize_timings(rollout_timing_samples),
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if error is not None and args.fail_on_error:
                env.close()
                raise RuntimeError(
                    f"Aborting after infrastructure error in mode={args.retrieval_mode}, "
                    f"task={task_id}, scene={scene_idx}, trial={rollout_idx}: {error}"
                )

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _parse_task_ids(task_ids: str | None, num_tasks_in_suite: int) -> list[int]:
    if task_ids is None or task_ids.strip() == "":
        return list(range(num_tasks_in_suite))
    parsed = []
    for raw_segment in task_ids.split(","):
        segment = raw_segment.strip()
        if not segment:
            continue
        if "-" in segment:
            start_text, end_text = segment.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Task id range must be ascending, got {segment!r}.")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(segment))
    if not parsed:
        raise ValueError("--task-ids was provided but no task ids were parsed.")
    invalid = [task_id for task_id in parsed if task_id < 0 or task_id >= num_tasks_in_suite]
    if invalid:
        raise ValueError(f"Task ids out of range for suite with {num_tasks_in_suite} tasks: {invalid}")
    return parsed


def _select_scene_trial_indices(
    *,
    task_id: int,
    initial_states: np.ndarray,
    num_trials_per_task: int,
    num_scene_setups_per_task: int | None,
    num_trials_per_scene: int,
    scene_setup_selection: SceneSetupSelection,
    retrieval_demos_dir: str | None,
    scene_setup_manifest_path: str | None,
) -> list[int]:
    if num_trials_per_scene <= 0:
        raise ValueError(f"--num-trials-per-scene must be positive, got {num_trials_per_scene}.")

    num_initial_states = len(initial_states)
    if num_initial_states <= 0:
        raise ValueError(f"Task {task_id} has no LIBERO initial states.")

    if num_scene_setups_per_task is None:
        if num_trials_per_task > num_initial_states:
            raise ValueError(
                f"Requested {num_trials_per_task} trials for task {task_id}, but only "
                f"{num_initial_states} initial states are available."
            )
        return list(range(num_trials_per_task))

    if num_scene_setups_per_task <= 0:
        raise ValueError(f"--num-scene-setups-per-task must be positive, got {num_scene_setups_per_task}.")

    selected = _select_scene_setups(
        task_id=task_id,
        num_initial_states=num_initial_states,
        count=num_scene_setups_per_task,
        scene_setup_selection=scene_setup_selection,
        retrieval_demos_dir=retrieval_demos_dir,
        scene_setup_manifest_path=scene_setup_manifest_path,
    )
    return [scene_idx for scene_idx in selected for _ in range(num_trials_per_scene)]


def _select_scene_setups(
    *,
    task_id: int,
    num_initial_states: int,
    count: int,
    scene_setup_selection: SceneSetupSelection,
    retrieval_demos_dir: str | None,
    scene_setup_manifest_path: str | None,
) -> list[int]:
    candidates = list(range(num_initial_states))
    if scene_setup_selection == "first":
        selected = candidates[:count]
    elif scene_setup_selection == "spread":
        selected = _spread_indices(candidates, count)
    elif scene_setup_selection == "unseen_retrieval":
        if retrieval_demos_dir is None:
            raise ValueError("--retrieval-demos-dir is required when --scene-setup-selection=unseen_retrieval.")
        seen = _scene_setups_in_retrieval_dir(pathlib.Path(retrieval_demos_dir), task_id, num_initial_states)
        unseen = [scene_idx for scene_idx in candidates if scene_idx not in seen]
        selected = _spread_indices(unseen, count)
    elif scene_setup_selection == "manifest":
        if scene_setup_manifest_path is None:
            raise ValueError("--scene-setup-manifest-path is required when --scene-setup-selection=manifest.")
        selected = _scene_setups_from_manifest(
            pathlib.Path(scene_setup_manifest_path), task_id, num_initial_states, count
        )
    else:
        raise ValueError(f"Unsupported --scene-setup-selection={scene_setup_selection!r}.")

    if len(selected) != count:
        raise ValueError(
            f"Could not select {count} scene setups for task {task_id} using {scene_setup_selection}; "
            f"selected {len(selected)} from {num_initial_states} initial states."
        )
    logging.info("Task %s selected scene setups: %s", task_id, selected)
    return selected


def _scene_setups_from_manifest(
    manifest_path: pathlib.Path, task_id: int, num_initial_states: int, count: int
) -> list[int]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scene-setup manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    per_task = payload.get("per_task")
    if not isinstance(per_task, dict):
        raise ValueError(f"Scene-setup manifest lacks an object-valued 'per_task' field: {manifest_path}")
    task_payload = per_task.get(str(task_id))
    if not isinstance(task_payload, dict) or not isinstance(task_payload.get("final"), list):
        raise ValueError(f"Scene-setup manifest lacks per_task.{task_id}.final: {manifest_path}")
    available = task_payload["final"]
    if len(available) < count:
        raise ValueError(
            f"Scene-setup manifest has only {len(available)} final scenes for task {task_id}; requested {count}."
        )
    selected = available[:count]
    if any(not isinstance(scene_idx, int) or isinstance(scene_idx, bool) for scene_idx in selected):
        raise ValueError(f"Scene-setup manifest contains a non-integer scene for task {task_id}: {selected}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Scene-setup manifest contains duplicate scenes for task {task_id}: {selected}")
    invalid = [scene_idx for scene_idx in selected if not 0 <= scene_idx < num_initial_states]
    if invalid:
        raise ValueError(
            f"Scene-setup manifest contains out-of-range scenes for task {task_id}: {invalid}; "
            f"valid range is 0-{num_initial_states - 1}."
        )
    return selected


def _spread_indices(candidates: list[int], count: int) -> list[int]:
    if count > len(candidates):
        raise ValueError(f"Cannot select {count} scene setups from {len(candidates)} candidates.")
    if count == len(candidates):
        return list(candidates)
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(position)] for position in positions]


def _scene_setups_in_retrieval_dir(
    retrieval_demos_dir: pathlib.Path, task_id: int, num_initial_states: int
) -> set[int]:
    task_dir = retrieval_demos_dir / f"task_{task_id:02d}"
    if not task_dir.exists():
        task_dir = retrieval_demos_dir / f"task_{task_id}"
    if not task_dir.exists():
        raise FileNotFoundError(f"Retrieval task directory not found for task {task_id}: {retrieval_demos_dir}")

    seen = set()
    for episode_path in task_dir.glob("episode_*.npz"):
        try:
            episode_index = int(episode_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        seen.add(episode_index % num_initial_states)
    if not seen:
        raise ValueError(f"No episode_*.npz files found in {task_dir}.")
    return seen


def _write_trial_result(
    args: Args,
    *,
    task_id: int,
    task_description: str,
    trial_id: int,
    scene_setup_id: int,
    success: bool,
    elapsed_steps: int,
    video_path: pathlib.Path | None,
    error: str | None,
    wall_seconds: float,
    timing_summary: dict[str, dict[str, float]],
) -> None:
    if args.results_jsonl is None:
        return
    row = {
        "checkpoint": args.checkpoint_dir,
        "task_id": task_id,
        "task_language": task_description,
        "trial_id": trial_id,
        "scene_setup_id": scene_setup_id,
        "retrieval_mode": args.retrieval_mode,
        "success": bool(success),
        "elapsed_steps": int(elapsed_steps),
        "video_path": None if video_path is None else str(video_path),
        "error": error,
        "wall_seconds": float(wall_seconds),
        "timings_ms": timing_summary,
    }
    with pathlib.Path(args.results_jsonl).open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_retrieval_compat_log(
    args: Args,
    *,
    task_id: int,
    task_description: str,
    trial_id: int,
    scene_setup_id: int,
    success: bool,
    elapsed_steps: int,
    error: str | None,
    retrieval_trace: list[dict],
    timing_summary: dict[str, dict[str, float]],
) -> None:
    if args.retrieval_log_path is None:
        return
    row = {
        "checkpoint": args.checkpoint_dir,
        "task_id": task_id,
        "task_language": task_description,
        "trial_id": trial_id,
        "scene_setup_id": scene_setup_id,
        "retrieval_mode": args.retrieval_mode,
        "success": bool(success),
        "elapsed_steps": int(elapsed_steps),
        "error": error,
        "retrieval_trace": retrieval_trace,
        "timings_ms": timing_summary,
    }
    with pathlib.Path(args.retrieval_log_path).open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_completed_trials(path: pathlib.Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    """Return the latest error-free row for each stable trial identity."""
    if not path.is_file():
        return {}
    completed = {}
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
            logging.warning("Ignoring malformed result row %s:%d: %s", path, line_number, error)
            continue
        if row.get("error") is None:
            completed[key] = row
    return completed


def _summarize_timings(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summaries = {}
    keys = sorted({key for sample in samples for key, value in sample.items() if np.isscalar(value)})
    for key in keys:
        values = np.asarray(
            [float(sample[key]) for sample in samples if key in sample and np.isfinite(float(sample[key]))],
            dtype=np.float64,
        )
        if len(values) == 0:
            continue
        summaries[key] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summaries


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
