from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses

import numpy as np

DEFAULT_ACTION_HORIZON = 10
DEFAULT_TOP_K = 4
EPS = 1e-8


@dataclasses.dataclass(frozen=True)
class RetrievalMetrics:
    top1_action_rmse: float
    best_of_4_action_rmse: float
    mean_top4_action_rmse: float
    oracle_action_rmse: float
    oracle_regret: float
    progress_error: float
    state_error: float
    unique_episodes: int

    def as_dict(self) -> dict[str, float | int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class MixedTaskRetrievalMetrics:
    top1_task_accuracy: float
    task_recall_at4: float
    correct_task_fraction_at4: float
    task_aware_ndcg_at4: float
    useful_retrieval_at4: float
    conditional_best_action_rmse: float | None
    conditional_oracle_regret: float | None
    exact_oracle_top1: float
    any_oracle_overlap_at4: float
    oracle_overlap_count_at4: float

    def as_dict(self) -> dict[str, float | None]:
        return dataclasses.asdict(self)


def select_query_steps(length: int, *, count: int, action_horizon: int = DEFAULT_ACTION_HORIZON) -> np.ndarray:
    """Select deterministic progress-spaced steps with a complete future action chunk."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}.")
    last_valid = length - action_horizon
    if last_valid < 0:
        raise ValueError(
            f"Episode length {length} is shorter than action horizon {action_horizon}; no valid query frame exists."
        )
    if count > last_valid + 1:
        raise ValueError(f"Cannot select {count} unique query frames from {last_valid + 1} valid frames.")
    positions = np.linspace(0, last_valid, count)
    steps = np.rint(positions).astype(np.int32)
    if len(np.unique(steps)) != count:
        # This branch is uncommon for LIBERO, but keeps the contract exact for short synthetic episodes.
        steps = np.asarray([int(i) for i in np.linspace(0, last_valid + 1, count, endpoint=False)], dtype=np.int32)
    if len(np.unique(steps)) != count:
        raise ValueError(f"Could not select {count} unique query frames from episode length {length}.")
    return steps


def get_action_chunk(actions: np.ndarray, step: int, *, action_horizon: int = DEFAULT_ACTION_HORIZON) -> np.ndarray:
    """Return a fixed action chunk, repeating the final action when a candidate is near episode end."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions with shape (T, D), got {actions.shape}.")
    if len(actions) == 0:
        raise ValueError("Cannot create an action chunk from an empty episode.")
    indices = np.minimum(np.arange(step, step + action_horizon), len(actions) - 1)
    return actions[indices]


def quantile_normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    q01 = np.asarray(q01, dtype=np.float32)
    q99 = np.asarray(q99, dtype=np.float32)
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def z_normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / (
        np.asarray(std, dtype=np.float32) + 1e-6
    )


def l2_normalize(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), EPS)


def make_pi0_observation_input(
    image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    prompt: str,
) -> dict[str, np.ndarray | str]:
    """Build the normal Pi0 observation prefix input, intentionally excluding actions and demonstrations."""
    return {
        "observation/image": np.asarray(image),
        "observation/wrist_image": np.asarray(wrist_image),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": str(prompt),
    }


def l2_distances(candidate_features: np.ndarray, query_feature: np.ndarray) -> np.ndarray:
    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    query_feature = np.asarray(query_feature, dtype=np.float32)
    return np.linalg.norm(candidate_features - query_feature, axis=-1)


def hybrid_distances(image_distances: np.ndarray, state_distances: np.ndarray, beta: float) -> np.ndarray:
    """Combine independently median-scaled distances so beta is comparable across tasks and queries."""
    image_distances = np.asarray(image_distances, dtype=np.float32)
    state_distances = np.asarray(state_distances, dtype=np.float32)
    if image_distances.shape != state_distances.shape:
        raise ValueError(f"Distance shapes differ: {image_distances.shape} != {state_distances.shape}.")
    image_scale = max(float(np.median(image_distances)), EPS)
    state_scale = max(float(np.median(state_distances)), EPS)
    return image_distances / image_scale + float(beta) * state_distances / state_scale


def deterministic_top_k(
    distances: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    k: int = DEFAULT_TOP_K,
) -> np.ndarray:
    distances = np.asarray(distances)
    episode_indices = np.asarray(episode_indices)
    frame_indices = np.asarray(frame_indices)
    if not (len(distances) == len(episode_indices) == len(frame_indices)):
        raise ValueError("Distance, episode, and frame arrays must have equal lengths.")
    if len(distances) < k:
        raise ValueError(f"Cannot select top-{k} from {len(distances)} candidates.")
    if not np.all(np.isfinite(distances)):
        raise ValueError("Retrieval distances contain NaN or infinity.")
    order = np.lexsort((frame_indices, episode_indices, distances))
    return order[:k].astype(np.int64)


def deterministic_global_top_k(
    distances: np.ndarray,
    task_indices: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    k: int = DEFAULT_TOP_K,
) -> np.ndarray:
    """Select exact top-k with deterministic task, episode, and frame tie-breaking."""
    distances = np.asarray(distances)
    task_indices = np.asarray(task_indices)
    episode_indices = np.asarray(episode_indices)
    frame_indices = np.asarray(frame_indices)
    if not (len(distances) == len(task_indices) == len(episode_indices) == len(frame_indices)):
        raise ValueError("Distance, task, episode, and frame arrays must have equal lengths.")
    if len(distances) < k:
        raise ValueError(f"Cannot select top-{k} from {len(distances)} candidates.")
    if not np.all(np.isfinite(distances)):
        raise ValueError("Retrieval distances contain NaN or infinity.")

    partition = np.argpartition(distances, k - 1)[:k]
    threshold = float(np.max(distances[partition]))
    strictly_better = np.flatnonzero(distances < threshold)
    remaining = k - len(strictly_better)
    tied = np.flatnonzero(distances == threshold)
    tied_order = np.lexsort((frame_indices[tied], episode_indices[tied], task_indices[tied]))
    selected = np.concatenate((strictly_better, tied[tied_order[:remaining]]))
    order = np.lexsort(
        (
            frame_indices[selected],
            episode_indices[selected],
            task_indices[selected],
            distances[selected],
        )
    )
    return selected[order].astype(np.int64)


def random_top_k(
    candidate_count: int,
    *,
    repeats: int = 20,
    k: int = DEFAULT_TOP_K,
    seed: int = 0,
) -> list[np.ndarray]:
    if candidate_count < k:
        raise ValueError(f"Cannot select top-{k} from {candidate_count} candidates.")
    rng = np.random.default_rng(seed)
    return [rng.choice(candidate_count, size=k, replace=False).astype(np.int64) for _ in range(repeats)]


def action_rmse(candidate_action_chunks: np.ndarray, query_action_chunk: np.ndarray) -> np.ndarray:
    candidate_action_chunks = np.asarray(candidate_action_chunks, dtype=np.float32)
    query_action_chunk = np.asarray(query_action_chunk, dtype=np.float32)
    if candidate_action_chunks.ndim != 3 or query_action_chunk.ndim != 2:
        raise ValueError(
            f"Expected candidate actions (N,H,D) and query actions (H,D), got "
            f"{candidate_action_chunks.shape} and {query_action_chunk.shape}."
        )
    return np.sqrt(np.mean(np.square(candidate_action_chunks - query_action_chunk[None, ...]), axis=(1, 2)))


def compute_retrieval_metrics(
    *,
    selected_indices: np.ndarray,
    candidate_action_chunks: np.ndarray,
    query_action_chunk: np.ndarray,
    candidate_states: np.ndarray,
    query_state: np.ndarray,
    candidate_progress: np.ndarray,
    query_progress: float,
    candidate_episode_indices: np.ndarray,
    all_action_errors: np.ndarray | None = None,
) -> RetrievalMetrics:
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if all_action_errors is None:
        all_action_errors = action_rmse(candidate_action_chunks, query_action_chunk)
    else:
        all_action_errors = np.asarray(all_action_errors, dtype=np.float32)
        if all_action_errors.shape != (len(candidate_action_chunks),):
            raise ValueError(
                f"Expected all_action_errors shape {(len(candidate_action_chunks),)}, got {all_action_errors.shape}."
            )
    selected_action_errors = all_action_errors[selected_indices]
    selected_states = np.asarray(candidate_states, dtype=np.float32)[selected_indices]
    state_errors = np.linalg.norm(selected_states - np.asarray(query_state, dtype=np.float32), axis=-1)
    progress_errors = np.abs(np.asarray(candidate_progress, dtype=np.float32)[selected_indices] - query_progress)
    oracle = float(np.min(all_action_errors))
    best = float(np.min(selected_action_errors))
    return RetrievalMetrics(
        top1_action_rmse=float(selected_action_errors[0]),
        best_of_4_action_rmse=best,
        mean_top4_action_rmse=float(np.mean(selected_action_errors)),
        oracle_action_rmse=oracle,
        oracle_regret=best - oracle,
        progress_error=float(np.mean(progress_errors)),
        state_error=float(np.mean(state_errors)),
        unique_episodes=int(len(np.unique(np.asarray(candidate_episode_indices)[selected_indices]))),
    )


def compute_mixed_task_metrics(
    *,
    selected_indices: np.ndarray,
    candidate_task_indices: np.ndarray,
    query_task_index: int,
    candidate_action_chunks: np.ndarray,
    query_action_chunk: np.ndarray,
    oracle_indices: np.ndarray | None = None,
    same_task_indices: np.ndarray | None = None,
    same_task_action_errors: np.ndarray | None = None,
    sorted_same_task_action_errors: np.ndarray | None = None,
    useful_action_error_threshold: float | None = None,
    useful_percentile: float = 0.1,
) -> MixedTaskRetrievalMetrics:
    """Score global retrieval while giving wrong-task candidates zero relevance."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    candidate_task_indices = np.asarray(candidate_task_indices, dtype=np.int32)
    candidate_action_chunks = np.asarray(candidate_action_chunks, dtype=np.float32)
    query_action_chunk = np.asarray(query_action_chunk, dtype=np.float32)
    if len(selected_indices) != DEFAULT_TOP_K:
        raise ValueError(f"Expected {DEFAULT_TOP_K} selected candidates, got {len(selected_indices)}.")
    if len(candidate_task_indices) != len(candidate_action_chunks):
        raise ValueError("Candidate task and action arrays must have equal lengths.")
    if not 0 < useful_percentile <= 1:
        raise ValueError("useful_percentile must be in (0, 1].")

    if same_task_indices is None:
        same_task_indices = np.flatnonzero(candidate_task_indices == int(query_task_index))
    else:
        same_task_indices = np.asarray(same_task_indices, dtype=np.int64)
        if not np.all(candidate_task_indices[same_task_indices] == int(query_task_index)):
            raise ValueError("same_task_indices contains a candidate from the wrong task.")
    if len(same_task_indices) < DEFAULT_TOP_K:
        raise ValueError(
            f"Task {query_task_index} has only {len(same_task_indices)} candidates; " f"need at least {DEFAULT_TOP_K}."
        )
    if same_task_action_errors is None:
        same_task_errors = action_rmse(
            candidate_action_chunks[same_task_indices],
            query_action_chunk,
        )
    else:
        same_task_errors = np.asarray(same_task_action_errors, dtype=np.float32)
        if same_task_errors.shape != (len(same_task_indices),):
            raise ValueError(
                f"Expected same_task_action_errors shape {(len(same_task_indices),)}, " f"got {same_task_errors.shape}."
            )
    if oracle_indices is None:
        oracle_local = deterministic_top_k(
            same_task_errors,
            np.zeros(len(same_task_indices), dtype=np.int32),
            same_task_indices,
            k=DEFAULT_TOP_K,
        )
        oracle_indices = same_task_indices[oracle_local]
    else:
        oracle_indices = np.asarray(oracle_indices, dtype=np.int64)

    selected_tasks = candidate_task_indices[selected_indices]
    correct_mask = selected_tasks == int(query_task_index)
    selected_correct = selected_indices[correct_mask]
    top1_correct = float(bool(correct_mask[0]))
    recall = float(bool(np.any(correct_mask)))
    correct_fraction = float(np.mean(correct_mask))

    if sorted_same_task_action_errors is None:
        sorted_errors = np.sort(same_task_errors)
    else:
        sorted_errors = np.asarray(sorted_same_task_action_errors, dtype=np.float32)
        if sorted_errors.shape != same_task_errors.shape:
            raise ValueError(
                f"Expected sorted_same_task_action_errors shape {same_task_errors.shape}, "
                f"got {sorted_errors.shape}."
            )
    discounts = 1.0 / np.log2(np.arange(2, DEFAULT_TOP_K + 2, dtype=np.float64))

    def relevance_for(global_indices: np.ndarray) -> np.ndarray:
        relevance = np.zeros(len(global_indices), dtype=np.float64)
        mask = candidate_task_indices[global_indices] == int(query_task_index)
        if not np.any(mask):
            return relevance
        errors = action_rmse(candidate_action_chunks[global_indices[mask]], query_action_chunk)
        ranks = np.searchsorted(sorted_errors, errors, side="right")
        denominator = max(len(sorted_errors) - 1, 1)
        relevance[mask] = np.clip(1.0 - (ranks - 1) / denominator, 0.0, 1.0)
        return relevance

    selected_relevance = relevance_for(selected_indices)
    oracle_relevance = relevance_for(oracle_indices)
    dcg = float(np.sum(selected_relevance * discounts))
    idcg = float(np.sum(oracle_relevance * discounts))
    ndcg = dcg / max(idcg, EPS)

    useful = 0.0
    conditional_best: float | None = None
    conditional_regret: float | None = None
    if len(selected_correct):
        correct_errors = action_rmse(
            candidate_action_chunks[selected_correct],
            query_action_chunk,
        )
        conditional_best = float(np.min(correct_errors))
        oracle_error = float(np.min(same_task_errors))
        conditional_regret = conditional_best - oracle_error
        useful_threshold = (
            float(np.quantile(same_task_errors, useful_percentile))
            if useful_action_error_threshold is None
            else float(useful_action_error_threshold)
        )
        useful = float(conditional_best <= useful_threshold)

    selected_ids = {int(index) for index in selected_indices}
    oracle_ids = {int(index) for index in oracle_indices}
    overlap = len(selected_ids & oracle_ids)
    return MixedTaskRetrievalMetrics(
        top1_task_accuracy=top1_correct,
        task_recall_at4=recall,
        correct_task_fraction_at4=correct_fraction,
        task_aware_ndcg_at4=ndcg,
        useful_retrieval_at4=useful,
        conditional_best_action_rmse=conditional_best,
        conditional_oracle_regret=conditional_regret,
        exact_oracle_top1=float(int(selected_indices[0]) == int(oracle_indices[0])),
        any_oracle_overlap_at4=float(overlap > 0),
        oracle_overlap_count_at4=float(overlap),
    )


def average_metric_dicts(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty metric collection.")
    keys = rows[0].keys()
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def macro_average(
    rows: Iterable[Mapping[str, object]],
    *,
    metric_names: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return per-task and equal-task macro summaries for query-level result rows."""
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["task_id"]))
        grouped.setdefault(key, []).append(row)

    per_task = []
    for (method, task_id), task_rows in sorted(grouped.items()):
        summary = {
            "method": method,
            "task_id": task_id,
            "query_count": len(task_rows),
        }
        summary.update({name: float(np.mean([float(row[name]) for row in task_rows])) for name in metric_names})
        per_task.append(summary)

    per_method: dict[str, list[Mapping[str, object]]] = {}
    for row in per_task:
        per_method.setdefault(str(row["method"]), []).append(row)
    summary_rows = []
    for method, task_rows in sorted(per_method.items()):
        summary = {
            "method": method,
            "task_count": len(task_rows),
            "query_count": sum(int(row["query_count"]) for row in task_rows),
        }
        summary.update({name: float(np.mean([float(row[name]) for row in task_rows])) for name in metric_names})
        summary_rows.append(summary)
    return per_task, summary_rows


def paired_task_bootstrap(
    baseline: Mapping[int, float],
    comparison: Mapping[int, float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    common = sorted(set(baseline) & set(comparison))
    if not common:
        raise ValueError("No common tasks for paired bootstrap.")
    deltas = np.asarray([comparison[task] - baseline[task] for task in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = np.mean(deltas[selections], axis=1)
    return float(np.mean(deltas)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
