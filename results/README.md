# LIBERO-90 closed-loop retrieval reproducibility package

This directory holds the derived tables and figures of the frozen 25-task
global-memory retrieval study. The analysis scripts write them to
`data/libero/reports/libero90_closed_loop_retrieval/`; they are copied here
unchanged. Raw rollout results, retrieval traces, experiment logs, rollout
videos, feature caches, datasets, retrieval pools and checkpoints are not
included in this repository.

## Run status

| Run | Status | Canonical rollouts |
| --- | --- | ---: |
| `libero90_retrieval_final10_global25_disjoint_20260810_211205` | Complete | 400/400 |
| `libero90_retrieval_extension15_global25_disjoint_20260811_145749` | Complete | 600/600 |
| `libero90_pi_vs_dino_context_full_20260810` | Incomplete pilot | See its `run_manifest.json` |

The pilot is retained as provenance, but it is not included in the pooled
1,000-rollout estimates. Its manifest records that it was stopped after the
final evaluation protocol superseded it.

## Required inputs

- The OpenPI environment and LIBERO simulator setup documented in the repository.
- The official LIBERO demonstrations converted into the local LeRobot layout.
- RICL consumer checkpoint:
  `checkpoints/pi0_fast_libero90_ricl/libero90_train20x20_ricl_18795949_20260621_000228/9999`.
- Normal pi0-FAST retriever checkpoint:
  `checkpoints/pi0_fast_libero90_normal/libero90_train20x20_normal/9999`.
- A 500-episode retrieval pool at
  `ricl_libero_preprocessing/libero90_solvable25_spread20/retrieval`, built from
  the 25 task IDs frozen in the protocol files with 20 episodes per task.
- The offline benchmark outputs under
  `data/libero/retrieval_benchmark/libero90_solvable25_mixed_global_v1`.

Raw run manifests record the absolute paths of the original machine. On a
different checkout, override the corresponding runner environment variables.

## Rollout video archive

All 2,960 rollout MP4s are stored in the private Hugging Face dataset
[`kugelblytz/ricl-libero-rollout-videos`](https://huggingface.co/datasets/kugelblytz/ricl-libero-rollout-videos).
The archive metadata (not included here) records
the relative path, byte size, run group, and SHA-256 checksum for every video.

```bash
uv run python scripts/publish_libero_rollout_videos.py --check
uv run python scripts/publish_libero_rollout_videos.py --upload
```

The upload command is rerunnable, preserves paths relative to
`data/libero/videos`, enforces private repository visibility, and verifies the
remote MP4 count after upload.

## Reproduce the evaluations

The exact task and scene partitions are frozen in
`scripts/libero90_final10_protocol.json` and
`scripts/libero90_extension15_protocol.json`.

```bash
scripts/run_libero90_final_retrieval.sh
scripts/run_libero90_extension15_retrieval.sh
```

Both runners are resumable and validate canonical result and trace identities
before marking a run complete. Their default conditions are DINO, DINO+State,
pi0 first-fusion, and uniform-random retrieval over the same global memory.

## Regenerate the analysis and report

```bash
.venv/bin/python scripts/analyze_libero90_closed_loop_retrieval.py
.venv/bin/python scripts/analyze_libero90_phase_retrieval.py
.venv/bin/python scripts/create_libero90_retrieval_final_report.py
uv run python scripts/build_libero90_reproducibility_manifest.py
uv run python scripts/build_libero90_reproducibility_manifest.py --check
```

The unified and phase analyses validate the two frozen protocols, require all
1,000 canonical result and trace rows, and write the CSV, JSON, PNG, and PDF
outputs to `data/libero/reports/libero90_closed_loop_retrieval/`. The phase analysis additionally maps policy-time
quartiles to cached expert-frame quartiles and source-task identities.
`artifact_manifest.json` (copied from the original run) records the size and SHA-256 digest of every
result and report artifact, including raw files that are not in this repository.
