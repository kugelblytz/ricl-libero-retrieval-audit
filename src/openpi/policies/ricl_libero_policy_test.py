# ruff: noqa: SLF001

import json

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import policy as policy_module
from openpi.policies.policy import RiclLiberoPolicy


def test_ricl_libero_policy_retrieve_returns_prefixed_inference_payload(tmp_path):
    demos_dir = tmp_path / "demos"
    task_dir = demos_dir / "task_00"
    task_dir.mkdir(parents=True)
    np.savez(
        task_dir / "episode_0.npz",
        image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        wrist_image=np.ones((2, 4, 4, 3), dtype=np.uint8),
        state=np.asarray([[0.0] * 8, [1.0] * 8], dtype=np.float32),
        actions=np.arange(14, dtype=np.float32).reshape(2, 7),
        prompt=np.asarray("pick up the object"),
    )

    policy = RiclLiberoPolicy.__new__(RiclLiberoPolicy)
    policy._knn_k = 1
    policy._use_action_interpolation = True
    policy._lamda = 10.0
    policy._action_horizon = 10
    policy._retrieval_backend = "state"
    policy._retrieval_key = "state"
    policy._embedding_type = "64PATCHES"
    policy._dinov2 = None
    policy._state_mean = np.zeros(8, dtype=np.float32)
    policy._state_std = np.ones(8, dtype=np.float32)
    policy._max_dist = 100.0
    policy._retrieval_mode = "nearest"
    policy._retrieval_rng = np.random.default_rng(0)
    policy._task_buffers = policy._load_task_buffers(demos_dir)
    policy._single_task_index = 0

    payload = policy.retrieve(
        {
            "state": np.zeros(8, dtype=np.float32),
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        }
    )

    assert payload["inference_time"] is True
    assert "query_actions" not in payload
    assert payload["query_state"].shape == (8,)
    assert payload["retrieved_0_state"].shape == (8,)
    assert payload["retrieved_0_actions"].shape == (10, 7)
    assert payload["exp_lamda_distances"].shape == (2, 1)
    assert payload["query_prompt"] == "pick up the object"

    transformed = libero_policy.RiclLiberoInputs(action_dim=7, num_retrieved_observations=1)(payload)
    transformed = {key: value for key, value in transformed.items() if not key.endswith("prompt")}
    for key, value in transformed.items():
        if key.endswith("image"):
            transformed[key] = {
                image_key: image.astype(np.float32) / 255.0 * 2.0 - 1.0 for image_key, image in value.items()
            }
    transformed = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
    observation = _model.RiclObservation.from_dict(transformed, num_retrieved_observations=1)
    assert observation.query_state.shape == (1, 8)


def test_ricl_libero_policy_image_retrieval_does_not_require_state_norm(tmp_path, monkeypatch):
    demos_dir = tmp_path / "demos"
    task_dir = demos_dir / "task_00"
    task_dir.mkdir(parents=True)
    (demos_dir / "retrieval_metadata.json").write_text(
        json.dumps(
            {
                "backend": "dinov2",
                "retrieval_key": "image",
                "embedding_type": "CLS",
                "metric": "l2",
                "num_retrieved_observations": 1,
            }
        )
    )
    (demos_dir / "max_distance.json").write_text('{"distances": {"max": 100.0}}')
    np.savez(
        task_dir / "episode_0.npz",
        image=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        wrist_image=np.ones((2, 4, 4, 3), dtype=np.uint8),
        state=np.asarray([[0.0] * 8, [1.0] * 8], dtype=np.float32),
        actions=np.arange(14, dtype=np.float32).reshape(2, 7),
        image_embeddings=np.asarray([[10.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        prompt=np.asarray("pick up the object"),
    )

    def fake_embed(images, dinov2, embedding_type):
        del images, dinov2, embedding_type
        return np.asarray([[0.1, 0.0]], dtype=np.float32)

    monkeypatch.setattr(policy_module, "embed", fake_embed)

    policy = RiclLiberoPolicy.__new__(RiclLiberoPolicy)
    policy._knn_k = 1
    policy._use_action_interpolation = True
    policy._lamda = 10.0
    policy._action_horizon = 10
    policy._retrieval_metadata = policy._load_retrieval_metadata(demos_dir)
    policy._retrieval_backend = "dinov2"
    policy._retrieval_key = "image"
    policy._embedding_type = "CLS"
    policy._dinov2 = object()
    policy._retrieval_mode = "nearest"
    policy._retrieval_rng = np.random.default_rng(0)
    policy._task_buffers = policy._load_task_buffers(demos_dir)
    policy._single_task_index = 0
    policy._max_dist = policy._load_max_distance(demos_dir)

    payload = policy.retrieve(
        {
            "state": np.zeros(8, dtype=np.float32),
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        }
    )

    np.testing.assert_allclose(payload["retrieved_0_state"], np.ones(8, dtype=np.float32))
    assert payload["exp_lamda_distances"].shape == (2, 1)
    assert "query_actions" not in payload


def test_ricl_libero_policy_random_retrieval_is_seeded_and_complete(tmp_path):
    demos_dir = tmp_path / "demos"
    task_dir = demos_dir / "task_00"
    task_dir.mkdir(parents=True)
    for episode_idx in range(3):
        np.savez(
            task_dir / f"episode_{episode_idx}.npz",
            image=np.full((2, 4, 4, 3), episode_idx, dtype=np.uint8),
            wrist_image=np.full((2, 4, 4, 3), episode_idx + 1, dtype=np.uint8),
            state=np.full((2, 8), episode_idx, dtype=np.float32),
            actions=np.full((2, 7), episode_idx, dtype=np.float32),
            prompt=np.asarray("pick up the object"),
        )

    def make_policy(seed: int):
        policy = RiclLiberoPolicy.__new__(RiclLiberoPolicy)
        policy._knn_k = 2
        policy._use_action_interpolation = True
        policy._lamda = 10.0
        policy._action_horizon = 10
        policy._retrieval_backend = "state"
        policy._retrieval_key = "state"
        policy._embedding_type = "64PATCHES"
        policy._dinov2 = None
        policy._state_mean = np.zeros(8, dtype=np.float32)
        policy._state_std = np.ones(8, dtype=np.float32)
        policy._max_dist = 100.0
        policy._retrieval_mode = "random"
        policy._retrieval_rng = np.random.default_rng(seed)
        policy._task_buffers = policy._load_task_buffers(demos_dir)
        policy._single_task_index = 0
        return policy

    obs = {
        "task_index": 0,
        "state": np.zeros(8, dtype=np.float32),
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
    }
    first_payload = make_policy(123).retrieve(obs)
    second_payload = make_policy(123).retrieve(obs)

    np.testing.assert_allclose(first_payload["retrieved_0_state"], second_payload["retrieved_0_state"])
    np.testing.assert_allclose(first_payload["retrieved_1_state"], second_payload["retrieved_1_state"])
    assert first_payload["retrieved_0_actions"].shape == (10, 7)
    assert first_payload["retrieved_1_actions"].shape == (10, 7)
    assert first_payload["exp_lamda_distances"].shape == (3, 1)
