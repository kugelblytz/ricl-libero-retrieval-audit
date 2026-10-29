"""
This is a copy of the utils.py file in the ricl_droid_preprocessing directory.
Also includes init_logging function from the scripts/train_pi0_fast_ricl.py file.
"""

from datetime import UTC
from datetime import datetime
import functools
import logging
import math

import numpy as np
from openpi_client.image_tools import resize_with_pad as resize_with_pad_numpy
import torch
import torchvision.transforms as torch_vt

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
EMBEDDING_TYPE = "64PATCHES"  # 'CLS', 'AVG', '16PATCHES'


def get_embed_dim(embedding_type=EMBEDDING_TYPE):
    if embedding_type in ("CLS", "AVG"):
        return 768
    if "PATCHES" in embedding_type:
        return int(embedding_type.split("PATCHES")[0]) * 768
    raise ValueError(f"{embedding_type=} is not supported")


EMBED_DIM = get_embed_dim(EMBEDDING_TYPE)  # based on the choice of the embedding type arg above


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def get_time():
    return datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def myprint(s):
    print(f"{get_time()}: {s}")


@functools.cache
def _select_torch_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        test_tensor = torch.ones(1, device="cuda")
        test_tensor = test_tensor + 1
        torch.cuda.synchronize()
        del test_tensor
    except RuntimeError as exc:
        error = str(exc).splitlines()[0]
        logging.warning("CUDA is unavailable for this PyTorch build (%s); using CPU for DINOv2.", error)
        return torch.device("cpu")

    return torch.device("cuda")


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return _select_torch_device()


def load_dinov2(device: str | torch.device | None = None):
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2.eval()
    device = _select_torch_device() if device is None else torch.device(device)
    return dinov2.to(device)


def process_dinov2(images, device: str | torch.device | None = None):
    assert isinstance(images, np.ndarray)
    assert images.dtype == np.uint8
    # if batch dimension not present, add it
    if len(images.shape) == 3:
        images = images[np.newaxis, ...]
    # if resolution is not 224x224, change resolution to 224x224.
    if not (images.shape[1:3] == (224, 224) or images.shape[2:4] == (224, 224)):
        # if channel first, convert to channel last before resolution change
        if images.shape[1] == 3:
            images = images.transpose(0, 2, 3, 1)
        # actual resolution change
        images = resize_with_pad_numpy(images, 224, 224)
    # if channel last, convert to channel first before pytorch steps
    if images.shape[3] == 3:
        images = images.transpose(0, 3, 1, 2)
    # convert uint8 numpy arrays to float32 tensors and normalize from [0,255] to [0,1]
    images = torch.from_numpy(images).float() / 255.0
    # normalize with imagenet mean and std
    normalize = torch_vt.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    images = normalize(images)
    if device is not None:
        images = images.to(torch.device(device))
    return images


def embed(images, dinov2, embedding_type=EMBEDDING_TYPE):
    images = process_dinov2(images, device=_module_device(dinov2))

    with torch.no_grad():
        features = dinov2.forward_features(
            images
        )  # dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks']) # shape of x_norm_regtokens = (batch_size, 0, 768)
        if embedding_type == "CLS":  # output of the CLS token
            batch_embeddings = features["x_norm_clstoken"]  # (batch_size, 768)
        elif (
            embedding_type == "AVG"
        ):  # average of num_tokens (e.g., num_tokens = 256 for 224x224 image since patch size is 14)
            batch_embeddings = features["x_norm_patchtokens"]  # (batch_size, num_tokens, 768)
            batch_embeddings = batch_embeddings.mean(dim=1)  # (batch_size, 768)
        elif "PATCHES" in embedding_type:  # reduces 256 patches to N patches
            batch_embeddings = features["x_norm_patchtokens"]  # (batch_size, 256, 768)
            batch_size = batch_embeddings.shape[0]
            n_patches = int(embedding_type.split("PATCHES")[0])
            assert 256 % n_patches == 0, f"256 is not divisible by {n_patches=}"
            assert math.sqrt(n_patches) ** 2 == n_patches, f"{n_patches=} must be a perfect square"
            patches = []
            rows, cols = 16, 16  # since 16*16 == 256
            patch_rows, patch_cols = int(rows // math.sqrt(n_patches)), int(cols // math.sqrt(n_patches))
            for i in range(0, rows, patch_rows):  # Step by patch height
                for j in range(0, cols, patch_cols):  # Step by patch width
                    patch_indices_2d = [(r, c) for r in range(i, i + patch_rows) for c in range(j, j + patch_cols)]
                    patch_indices_in_flattened = [r * cols + c for r, c in patch_indices_2d]
                    # print(patch_indices_in_flattened)
                    patch = batch_embeddings[:, patch_indices_in_flattened, :]  # (batch_size, 16, 768)
                    assert patch.shape == (batch_size, patch_rows * patch_cols, 768), f"{patch.shape=}"
                    patch = patch.mean(dim=1)  # (batch_size, 768)
                    assert patch.shape == (batch_size, 768), f"{patch.shape=}"
                    patches.append(patch)
            assert len(patches) == n_patches, f"{len(patches)=} {n_patches=}"
            batch_embeddings = torch.cat(patches, dim=1)  # (batch_size, 16*768)
        else:
            raise ValueError(f"{embedding_type=} is not supported")

    return batch_embeddings.cpu().numpy()


def embed_with_batches(images, dinov2, batch_size=256, embedding_type=EMBEDDING_TYPE):
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        images_batch = images[i : i + batch_size]
        embeddings = embed(images_batch, dinov2, embedding_type=embedding_type)
        all_embeddings.append(embeddings)
    return np.concatenate(all_embeddings, axis=0)
