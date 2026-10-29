from __future__ import annotations

import gc
import logging
import pathlib
from typing import Any

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import gemma_fast
from openpi.models import model as model_lib
from openpi.models import pi0_fast
from openpi.models import siglip

logger = logging.getLogger(__name__)


class FirstFusionFeatureEncoder:
    """Minimal Pi0 encoder for the normalized first-fusion retrieval feature.

    The full Pi0FAST checkpoint contains all 18 Gemma transformer blocks and the
    language-model output stack. First-fusion retrieval only uses the shared
    SigLIP image tower, the token embedding table, and attention from block 0.
    This class restores exactly those checkpoint branches and never places the
    unused transformer/MLP parameters on the accelerator.
    """

    def __init__(self, model_config: pi0_fast.Pi0FASTConfig, checkpoint_dir: pathlib.Path):
        if model_config.paligemma_variant != "gemma_2b":
            raise ValueError(
                "The pruned first-fusion encoder currently supports only the exact gemma_2b "
                f"retrieval checkpoint, got {model_config.paligemma_variant!r}."
            )
        if model_config.dtype != "float32":
            raise ValueError(
                "The cached Pi first-fusion features were exported in float32; "
                f"the pruned encoder requires dtype='float32', got {model_config.dtype!r}."
            )

        gemma_config = gemma_fast.get_config(model_config.paligemma_variant)
        self._image_model = siglip.Module(
            num_classes=gemma_config.width,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=model_config.dtype,
        )
        self._text_embedder = gemma_fast.Embedder(
            vocab_size=gemma_config.vocab_size,
            embed_dim=gemma_config.width,
        )
        self._first_block = gemma_fast.Block(
            num_heads=gemma_config.num_heads,
            num_kv_heads=gemma_config.num_kv_heads,
            embed_dim=gemma_config.width,
            head_dim=gemma_config.head_dim,
            hidden_dim=gemma_config.mlp_dim,
            cache_dtype=model_config.dtype,
        )

        host_params = _restore_first_fusion_params(checkpoint_dir / "params")
        self._parameter_bytes = sum(np.asarray(value).nbytes for value in jax.tree.leaves(host_params))
        self._params = jax.tree.map(jax.device_put, host_params)
        del host_params
        gc.collect()
        self._extract = jax.jit(self._extract_impl)
        logger.info(
            "Loaded pruned Pi first-fusion encoder: %.2f GiB across %d arrays",
            self._parameter_bytes / 2**30,
            len(jax.tree.leaves(self._params)),
        )

    @property
    def parameter_bytes(self) -> int:
        return self._parameter_bytes

    def __call__(self, observation: model_lib.Observation) -> jax.Array:
        return self._extract(self._params, observation)

    def _extract_impl(self, params: dict[str, Any], observation: model_lib.Observation) -> jax.Array:
        observation = model_lib.preprocess_observation(
            None,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )

        token_embeddings = []
        input_masks = []
        ar_masks = []
        for name in observation.images:
            encoded_image, _ = self._image_model.apply({"params": params["img"]}, observation.images[name], train=False)
            token_embeddings.append(encoded_image)
            image_mask = jnp.repeat(observation.image_masks[name][:, None], encoded_image.shape[1], axis=1)
            input_masks.append(image_mask)
            ar_masks.append(jnp.zeros_like(image_mask))

        if observation.tokenized_prompt is None:
            raise ValueError("Tokenized prompt is required for Pi first-fusion retrieval.")
        if observation.tokenized_prompt_mask is None or observation.token_ar_mask is None:
            raise ValueError("Prompt and autoregressive masks are required for Pi first-fusion retrieval.")
        text_embeddings = self._text_embedder.apply(
            {"params": params["embedder"]},
            observation.tokenized_prompt,
            method=self._text_embedder.encode,
        )
        token_embeddings.append(text_embeddings)
        input_masks.append(observation.tokenized_prompt_mask)
        ar_masks.append(observation.token_ar_mask)

        prefix_embeddings = jnp.concatenate(token_embeddings, axis=1)
        prefix_mask = jnp.concatenate(input_masks, axis=1)
        prefix_ar_mask = jnp.concatenate(ar_masks, axis=1)
        prefix_attn_mask = pi0_fast.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_embeddings, prefix_mask, prefix_attn_mask = pi0_fast.left_to_right_align(
            prefix_embeddings, prefix_mask, prefix_attn_mask
        )
        if prefix_attn_mask.ndim == 3:
            prefix_attn_mask = prefix_attn_mask[:, None, :, :]
        prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1

        _, first_attention = self._first_block.apply(
            {"params": params["first_block"]},
            prefix_embeddings,
            None,
            prefix_positions,
            prefix_attn_mask,
            decode=False,
            deterministic=True,
            layer_active=True,
            return_attention_output=True,
        )
        features = first_attention[:, -1]
        return features / jnp.maximum(jnp.linalg.norm(features, axis=-1, keepdims=True), 1e-8)


def _restore_first_fusion_params(params_path: pathlib.Path) -> dict[str, Any]:
    """Restore only checkpoint leaves needed by :class:`FirstFusionFeatureEncoder`."""
    params_path = params_path.resolve()
    if not params_path.is_dir():
        raise FileNotFoundError(f"Pi retriever params not found: {params_path}")

    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(params_path)["params"]
        llm = metadata["PaliGemma"]["llm"]
        selected = {
            "PaliGemma": {
                "img": metadata["PaliGemma"]["img"],
                "llm": {
                    "embedder": {"input_embedding": llm["embedder"]["input_embedding"]},
                    "layers": {
                        "pre_attention_norm": llm["layers"]["pre_attention_norm"],
                        "attn": {
                            name: llm["layers"]["attn"][name] for name in ("q_einsum", "kv_einsum", "attn_vec_einsum")
                        },
                    },
                },
            }
        }
        item = {"params": selected}
        restore_args = jax.tree.map(
            lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray, dtype=np.float32),
            item,
        )
        transforms = jax.tree.map(lambda _: ocp.RestoreTransform(), item)
        restored = checkpointer.restore(
            params_path,
            args=ocp.args.PyTreeRestore(
                item=item,
                restore_args=restore_args,
                transforms=transforms,
            ),
        )["params"]

    flat_params = flax.traverse_util.flatten_dict(restored)
    if not flat_params or not all(key_path[-1] == "value" for key_path in flat_params):
        raise ValueError(f"Unexpected Pi checkpoint parameter structure in {params_path}.")
    restored = flax.traverse_util.unflatten_dict({key_path[:-1]: value for key_path, value in flat_params.items()})
    paligemma = restored["PaliGemma"]
    layers = paligemma["llm"]["layers"]
    first_block = {
        "pre_attention_norm": {"scale": layers["pre_attention_norm"]["scale"][0]},
        "attn": {name: {"w": layers["attn"][name]["w"][0]} for name in ("q_einsum", "kv_einsum", "attn_vec_einsum")},
    }
    return {
        "img": paligemma["img"],
        "embedder": paligemma["llm"]["embedder"],
        "first_block": first_block,
    }
