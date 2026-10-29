# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

export SERVER_ARGS="--env LIBERO"
docker compose -f examples/libero/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Convert LIBERO-100 for Training

The converter supports the official LIBERO-100 HDF5 download layout with `LIBERO-90/` and `LIBERO-10/` folders:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --raw-dir /path/to/LIBERO \
  --repo-id your_hf_username/libero100 \
  --split libero_100_all \
  --overwrite
```

Selectable splits:

```bash
# LIBERO-90 only
uv run examples/libero/convert_libero_data_to_lerobot.py --raw-dir /path/to/LIBERO --repo-id your_hf_username/libero90 --split libero_90 --overwrite

# LIBERO-10 only
uv run examples/libero/convert_libero_data_to_lerobot.py --raw-dir /path/to/LIBERO --repo-id your_hf_username/libero10 --split libero_10 --overwrite

# Specific task IDs within each selected split
uv run examples/libero/convert_libero_data_to_lerobot.py --raw-dir /path/to/LIBERO --repo-id your_hf_username/libero_subset --split libero_90 --task-ids 0,1,2 --overwrite

# Task-name filtering
uv run examples/libero/convert_libero_data_to_lerobot.py --raw-dir /path/to/LIBERO --repo-id your_hf_username/libero_mug --split libero_100_all --task-name-filter mug --overwrite
```

The output keeps OpenPI's LIBERO training fields (`image`, `wrist_image`, `state`, `actions`) and adds numeric frame metadata for `libero_split_index`, `libero_task_id`, and `libero_global_task_id`. A `libero_conversion_manifest.json` sidecar is written into the dataset root.

The old RLDS path is still available:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py \
  --source-format rlds \
  --raw-dir /path/to/tfds/data \
  --repo-id your_hf_username/libero_rlds \
  --overwrite
```

## Results

If you follow the training instructions and hyperparameters in the `pi0_libero` and `pi0_fast_libero` configs, you should get results similar to the following:

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0-FAST @ 30k (finetuned) | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| π0 @ 30k (finetuned) | 96.8 | 98.8 | 95.8 | 85.2 | 94.15 |

Note that the hyperparameters for these runs are not tuned and $\pi_0$-FAST does not use a FAST tokenizer optimized for Libero. Likely, the results could be improved with more tuning, we mainly use these results as an example of how to use openpi to fine-tune $\pi_0$ models on a new dataset.
