import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def make_ricl_libero_example(num_retrieved_observations: int = 4, action_horizon: int = 10) -> dict:
    """Creates a random input example for a RICL Libero policy."""
    example = {}
    prefixes = [f"retrieved_{i}_" for i in range(num_retrieved_observations)] + ["query_"]

    for prefix in prefixes:
        example[f"{prefix}state"] = np.random.rand(8)
        example[f"{prefix}image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        example[f"{prefix}wrist_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
        example[f"{prefix}prompt"] = "do something"

    for i in range(num_retrieved_observations):
        example[f"retrieved_{i}_actions"] = np.random.rand(action_horizon, 7)
    example["query_actions"] = np.random.rand(action_horizon, 7)
    example["exp_lamda_distances"] = np.ones((num_retrieved_observations + 1, 1), dtype=np.float32)

    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), wrist_image)
            image_masks = (np.True_, np.True_, np.True_)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (np.True_, np.True_, np.False_ if mask_padding else np.True_)

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class RiclLiberoInputs(transforms.DataTransformFn):
    """
    Converts RICL Libero inputs to the model format.

    RICL examples should already be retrieval-packed with keys like:
      retrieved_0_state, retrieved_0_image, retrieved_0_wrist_image,
      retrieved_0_actions, retrieved_0_prompt, ..., query_state,
      query_image, query_wrist_image, query_actions, query_prompt.

    The physical-intelligence/libero LeRobot dataset has single-observation
    rows, so a retrieval-aware dataset wrapper should create these prefixed
    fields before this transform runs.
    """

    action_dim: int
    num_retrieved_observations: int
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        prefixes = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]

        inputs = {}
        for prefix in prefixes:
            state = transforms.pad_to_dim(data[f"{prefix}state"], self.action_dim)
            base_image = _parse_image(data[f"{prefix}image"])
            wrist_image = _parse_image(data[f"{prefix}wrist_image"])
            padding_image = np.zeros_like(base_image)

            if self.model_type == _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, padding_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            else:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, padding_image)
                image_masks = (np.True_, np.True_, np.False_ if mask_padding else np.True_)

            inputs[f"{prefix}state"] = state
            inputs[f"{prefix}image"] = dict(zip(names, images, strict=True))
            inputs[f"{prefix}image_mask"] = dict(zip(names, image_masks, strict=True))

            if f"{prefix}prompt" in data:
                inputs[f"{prefix}prompt"] = data[f"{prefix}prompt"]

        for i in range(self.num_retrieved_observations):
            action_key = f"retrieved_{i}_actions"
            inputs[action_key] = transforms.pad_to_dim(data[action_key], self.action_dim)

        if "query_actions" in data:
            inputs["query_actions"] = transforms.pad_to_dim(data["query_actions"], self.action_dim)

        if "exp_lamda_distances" in data:
            inputs["exp_lamda_distances"] = data["exp_lamda_distances"]

        if "inference_time" in data:
            inputs["inference_time"] = data["inference_time"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RiclLiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"query_actions": np.asarray(data["query_actions"][..., :7])}
