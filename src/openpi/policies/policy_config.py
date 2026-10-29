from collections.abc import Sequence
import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

logger = logging.getLogger()


def _maybe_force_float32_policy(train_config: _config.TrainConfig) -> tuple[_config.TrainConfig, jnp.dtype]:
    if os.environ.get("OPENPI_POLICY_FLOAT32_FOR_CUDA", "0") != "1":
        return train_config, jnp.bfloat16
    model_config = train_config.model
    if hasattr(model_config, "dtype"):
        model_config = dataclasses.replace(model_config, dtype="float32")
        train_config = dataclasses.replace(train_config, model=model_config)
    logging.info("OPENPI_POLICY_FLOAT32_FOR_CUDA=1: restoring policy params as float32.")
    return train_config, jnp.float32

@dataclasses.dataclass
class PolicyConfig:
    model: _model.BaseModel
    norm_stats: dict[str, transforms.NormStats]

    input_layers: Sequence[transforms.DataTransformFn]
    output_layers: Sequence[transforms.DataTransformFn]

    model_type: _model.ModelType = _model.ModelType.PI0
    default_prompt: str | None = None
    sample_kwargs: dict[str, Any] | None = None


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    train_config, restore_dtype = _maybe_force_float32_policy(train_config)

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=restore_dtype))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )


def create_trained_ricl_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    demos_dir: str,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    retrieval_mode: str = "nearest",
    retrieval_seed: int = 0,
    retrieval_method: str = "dino",
    retrieval_scope: str = "task",
    pi_feature_dir: str | None = None,
    pi_retriever_config: str = "pi0_fast_libero90_normal",
    pi_retriever_checkpoint: str | None = None,
    pi_retriever_dtype: str = "float32",
    state_norm_checkpoint: str | None = None,
    dino_state_beta: float = 1.0,
    retrieval_search_device: str = "auto",
    retrieval_distance_chunk_size: int = 256,
) -> _policy.BasePolicy:
    """Create a ricl policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        demos_dir: The directory to load the demos from.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    train_config, restore_dtype = _maybe_force_float32_policy(train_config)

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=restore_dtype))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    is_libero_ricl_config = "libero" in train_config.name and "ricl" in train_config.name
    policy_cls = _policy.RiclLiberoPolicy if is_libero_ricl_config else _policy.RiclPolicy
    policy_kwargs = {}
    if is_libero_ricl_config:
        policy_kwargs["norm_stats"] = norm_stats
        policy_kwargs["retrieval_mode"] = retrieval_mode
        policy_kwargs["retrieval_seed"] = retrieval_seed
        policy_kwargs["retrieval_method"] = retrieval_method
        policy_kwargs["retrieval_scope"] = retrieval_scope
        policy_kwargs["pi_feature_dir"] = pi_feature_dir
        policy_kwargs["pi_retriever_config"] = pi_retriever_config
        policy_kwargs["pi_retriever_checkpoint"] = pi_retriever_checkpoint
        policy_kwargs["pi_retriever_dtype"] = pi_retriever_dtype
        policy_kwargs["state_norm_checkpoint"] = state_norm_checkpoint
        policy_kwargs["dino_state_beta"] = dino_state_beta
        policy_kwargs["retrieval_search_device"] = retrieval_search_device
        policy_kwargs["retrieval_distance_chunk_size"] = retrieval_distance_chunk_size
        if data_config.asset_id is not None:
            policy_kwargs["max_distance_path"] = (
                pathlib.Path(checkpoint_dir) / "assets" / data_config.asset_id / "max_distance.json"
            )

    return policy_cls(
        model,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.UnnormalizeRicl(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_config.policy_metadata,
        demos_dir=demos_dir,
        use_action_interpolation=train_config.model.use_action_interpolation,
        lamda=train_config.model.lamda,
        action_horizon=train_config.model.action_horizon,
        **policy_kwargs,
    )
