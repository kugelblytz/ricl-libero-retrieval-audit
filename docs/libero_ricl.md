# RICL-Pi0-FAST on LIBERO: setup, training, serving and evaluation

RICL-Pi0-FAST-LIBERO trains from the public LeRobot dataset `physical-intelligence/libero`. First build the LIBERO-10 RICL cache, retrieval indices, norm stats, and serving buffers:

```bash
uv run python scripts/setup_ricl_libero.py --retrieval-key=image --embedding-type=64PATCHES
```

This writes:

```text
ricl_libero_preprocessing/
├── episodes/
├── ricl_indices_and_distances.npz
├── max_distance.json
├── retrieval_metadata.json
└── retrieval/
    ├── retrieval_metadata.json
    ├── max_distance.json
    ├── task_00/
    ├── ...
    └── task_09/
assets/pi0_fast_libero_ricl/libero/
├── max_distance.json
├── retrieval_metadata.json
└── norm_stats.json
```

The setup script defaults to LIBERO-10 task indices `0..9`, `K=4`, and same-task DINOv2 image-embedding retrieval. The `pi0_fast_libero_ricl` train config uses paper-style RICL retrieval/training settings adapted to LIBERO's 7-D action space and LIBERO-native 10-step action horizon: frozen image encoder, batch size 16, and the same cosine learning-rate schedule as DROID RICL. For a faster debugging baseline, you can build normalized state-L2 retrieval artifacts instead:

```bash
uv run python scripts/setup_ricl_libero.py --retrieval-key=state
```

Train and serve with artifacts produced by the same retrieval mode, since the interpolation distances and retrieval examples are coupled to the generated `retrieval_metadata.json`. Then train:

You do not need to rerun `scripts/setup_ricl_libero.py` before every fine-tuning experiment. Reuse the generated artifacts unless you change the source dataset, `retrieval_key`, `embedding_type`, task indices, `num_retrieved_observations`, or output directories, or unless the generated metadata/cache files are missing or incomplete.

```bash
python scripts/train_pi0_fast_ricl.py pi0_fast_libero_ricl --exp-name={YOUR_EXPERIMENT_NAME_HERE} --overwrite
```

The `pi0_fast_libero_ricl` config starts from `s3://openpi-assets/checkpoints/pi0_fast_libero/params` and uses `RiclLiberoDataset`, `RiclLiberoInputs`, `RiclLiberoOutputs`, and the RICL FAST transforms.

For a paper-style practical LIBERO-90 split, first convert official LIBERO-90 HDF5 data into a LeRobot dataset:

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --source-format official_hdf5 \
  --raw-dir /path/to/LIBERO \
  --split libero_90 \
  --repo-id local/libero90 \
  --overwrite
```

Then build RICL training artifacts from 20 LIBERO-90 tasks with 20 seeded demos per task:

```bash
uv run python scripts/setup_ricl_libero.py \
  --repo-id local/libero90 \
  --task-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --max-episodes-per-task 20 \
  --episode-selection seeded \
  --seed 0 \
  --artifact-role train \
  --preprocessing-dir ricl_libero_preprocessing/libero90_train20x20 \
  --retrieval-dir ricl_libero_preprocessing/libero90_train20x20/retrieval \
  --assets-dir assets/pi0_fast_libero90_ricl/libero \
  --retrieval-key image \
  --embedding-type 64PATCHES
```

Build held-out retrieval buffers for the remaining LIBERO-90 tasks:

```bash
uv run python scripts/setup_ricl_libero.py \
  --repo-id local/libero90 \
  --task-indices 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 \
  --max-episodes-per-task 20 \
  --episode-selection seeded \
  --seed 1 \
  --artifact-role retrieval \
  --preprocessing-dir ricl_libero_preprocessing/libero90_eval70x20 \
  --retrieval-dir ricl_libero_preprocessing/libero90_eval70x20/retrieval \
  --max-distance-source assets/pi0_fast_libero90_ricl/libero/max_distance.json \
  --retrieval-key image \
  --embedding-type 64PATCHES
```

Train the split-specific RICL config:

```bash
uv run python scripts/train_pi0_fast_ricl.py \
  pi0_fast_libero90_ricl \
  --exp-name=libero90_train20x20_ricl \
  --overwrite
```

This config uses the same LIBERO-native 7-D action space and 10-step action horizon, starts from `s3://openpi-assets/checkpoints/pi0_fast_libero/params`, and reads training artifacts from `ricl_libero_preprocessing/libero90_train20x20`.

## Serve RICL-Pi0-FAST-LIBERO
Serve a trained LIBERO RICL checkpoint with the retrieval buffers created by the setup script:

```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint --policy.config=pi0_fast_libero_ricl --policy.dir=checkpoints/pi0_fast_libero_ricl/{YOUR_EXPERIMENT_NAME_HERE}/{STEP} --policy.demos_dir=ricl_libero_preprocessing/retrieval
```

For held-out LIBERO-90 eval tasks, serve the split-specific checkpoint with the held-out retrieval buffer:

```bash
uv run scripts/serve_policy_ricl.py policy:checkpoint \
  --policy.config=pi0_fast_libero90_ricl \
  --policy.dir=checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl/{STEP} \
  --policy.demos_dir=ricl_libero_preprocessing/libero90_eval70x20/retrieval
```

At inference, pass `task_index` with normal LIBERO observation keys (`observation/state`, `observation/image`, `observation/wrist_image`, `prompt`) or query-prefixed keys (`query_state`, `query_image`, `query_wrist_image`, `query_prompt`). If the retrieval directory contains exactly one `task_XX` folder, `task_index` can be omitted.

## Test RICL-Pi0-FAST-LIBERO
Run the cheap local checks without downloading LIBERO:

```bash
bash scripts/run_ricl_libero_quick_checks.sh
```

This runs whitespace validation, focused ruff checks, bytecode compilation, the synthetic RICL-LIBERO pytest suite, and the config smoke test. The script defaults to `JAX_PLATFORMS=cpu` and `USE_TF=0`.

After running `scripts/setup_ricl_libero.py`, validate the real generated artifacts:

```bash
uv run python scripts/validate_ricl_libero_artifacts.py
```

For the LIBERO-90 split artifacts:

```bash
uv run python scripts/validate_ricl_libero_artifacts.py \
  --preprocessing-dir ricl_libero_preprocessing/libero90_train20x20 \
  --assets-dir assets/pi0_fast_libero90_ricl/libero \
  --task-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19

uv run python scripts/validate_ricl_libero_artifacts.py \
  --artifact-role retrieval \
  --preprocessing-dir ricl_libero_preprocessing/libero90_eval70x20 \
  --assets-dir assets/pi0_fast_libero90_ricl/libero \
  --task-indices 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89
```

Then smoke-test the real data-loader route with prefixed norm stats enabled:

```bash
JAX_PLATFORMS=cpu USE_TF=0 uv run python scripts/smoke_ricl_libero_data_loader.py
JAX_PLATFORMS=cpu USE_TF=0 uv run python scripts/smoke_ricl_libero_data_loader.py --config-name pi0_fast_libero90_ricl
```

For a heavier one-step training smoke:

```bash
JAX_PLATFORMS=cpu USE_TF=0 uv run python scripts/train_pi0_fast_ricl.py \
  pi0_fast_libero_ricl \
  --exp-name=smoke_libero_ricl \
  --num-train-steps=1 \
  --batch-size=1 \
  --num-workers=0 \
  --no-wandb-enabled \
  --overwrite
```

After that checkpoint exists, verify the serving route selects `RiclLiberoPolicy`:

```bash
JAX_PLATFORMS=cpu USE_TF=0 uv run python scripts/smoke_ricl_libero_serving.py
```

