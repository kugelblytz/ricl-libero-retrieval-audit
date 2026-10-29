# RICL LIBERO Findings

Generated: 2026-07-03

## Executive Summary

- Four local LIBERO eval experiments were compared: 20-memory, 40-memory, 49-memory/1-scene, and the expanded mismatched-context diagnostic.
- Increasing retrieval memory from 20 to 40 to 49 episodes/task did not improve aggregate success.
- Retrieval quality still correlates with success in the matched-context runs: successful rollouts have lower top-1 retrieval distances than failed rollouts.
- The expanded mismatch run scored 36/60 on the six previously successful-at-least-once tasks, or 36/100 after assuming the historically failed tasks remain 0/10.
- The mismatch result weakens the causal RICL claim for several known-good tasks: some skills still work with deliberately wrong context.
- Persistent failures align with weak or absent finetune task-family overlap, especially bottom-drawer and wine-bottle/wine-rack tasks.

## Raw Run Comparison

| Run | Retrieval demos/task | Eval tasks | Eval scenes/task | Success | Success rate | Retrieval queries |
| --- | --- | --- | --- | --- | --- | --- |
| 20mem_leftout10 | 20 | 20-29 | 10 | 52/100 | 52.0% | 5334 |
| 40mem_leftout10 | 40 | 20-29 | 10 | 49/100 | 49.0% | 5254 |
| 49mem_leftout1 | 49 | 20-29 | 1 | 5/10 | 50.0% | 535 |
| mismatch_successful_tasks | 10 | 20,21,24,25,28,29 | 10 | 36/60 | 60.0% | 2847 |

## Normalized 100-Rollout Comparison

This table uses 10 eval rollouts/task across tasks 20-29. For the 49-memory run, the single observed rollout per task is projected to 10. For the mismatch run, tasks 22, 23, 26, and 27 are assumed to remain 0/10 because they always failed earlier.

| Run | Normalized success | Normalized rate | Normalization |
| --- | --- | --- | --- |
| 20mem_leftout10 | 52/100 | 52.0% | Observed 10 rollouts/task. |
| 40mem_leftout10 | 49/100 | 49.0% | Observed 10 rollouts/task. |
| 49mem_leftout1 | 50/100 | 50.0% | Projected each 1-rollout task result to 10 rollouts/task. |
| mismatch_successful_tasks | 36/100 | 36.0% | Adds tasks 22, 23, 26, and 27 as assumed 0/10. |

## Per-Task Success

| Task | Prompt | 20mem_leftout10 | 40mem_leftout10 | 49mem_leftout1 | mismatch_successful_tasks |
| --- | --- | --- | --- | --- | --- |
| 20 | turn on the stove | 10/10 | 10/10 | 1/1 | 9/10 |
| 21 | turn on the stove and put the frying pan on it | 7/10 | 4/10 | 1/1 | 0/10 |
| 22 | close the bottom drawer of the cabinet | 0/10 | 0/10 | 0/1 | - |
| 23 | close the bottom drawer of the cabinet and open the top drawer | 0/10 | 0/10 | 0/1 | - |
| 24 | put the black bowl in the bottom drawer of the cabinet | 6/10 | 6/10 | 0/1 | 4/10 |
| 25 | put the black bowl on top of the cabinet | 9/10 | 10/10 | 1/1 | 7/10 |
| 26 | put the wine bottle in the bottom drawer of the cabinet | 0/10 | 0/10 | 0/1 | - |
| 27 | put the wine bottle on the wine rack | 0/10 | 0/10 | 0/1 | - |
| 28 | close the top drawer of the cabinet | 10/10 | 10/10 | 1/1 | 6/10 |
| 29 | put the black bowl in the top drawer of the cabinet | 10/10 | 9/10 | 1/1 | 10/10 |

Normalized mismatch per-task assumption: tasks 22, 23, 26, and 27 are counted as 0/10 in the 100-rollout view because they were not included in the expanded mismatch run and had never succeeded in prior completed matched-context runs.

## Retrieval Distance Findings

Distance threshold used for above-max counts: `158.5907`.

| Run | Outcome | Rollouts | Mean top-1 dist | Mean queries | Above max |
| --- | --- | --- | --- | --- | --- |
| 20mem_leftout10 | success | 52 | 120.3 | 28.7 | 10/1494 (0.7%) |
| 20mem_leftout10 | failure | 48 | 161.3 | 80.0 | 2252/3840 (58.6%) |
| 40mem_leftout10 | success | 49 | 118.6 | 26.0 | 3/1276 (0.2%) |
| 40mem_leftout10 | failure | 51 | 152.8 | 78.0 | 1655/3978 (41.6%) |
| 49mem_leftout1 | success | 5 | 113.1 | 27.0 | 0/135 (0.0%) |
| 49mem_leftout1 | failure | 5 | 160.3 | 80.0 | 190/400 (47.5%) |
| mismatch_successful_tasks | success | 36 | 178.7 | 25.8 | 585/927 (63.1%) |
| mismatch_successful_tasks | failure | 24 | 181.0 | 80.0 | 1304/1920 (67.9%) |

Interpretation: failures remain farther from retrieval neighbors than successes in the matched-context runs. However, the mismatch run has high top-1 distances for both successes and failures, showing that distance alone does not prove RICL is causally helping when the finetuned policy already knows the skill.

## Task Closeness To Finetune Data

| Task | Eval prompt | Closeness | Nearest train task | Score | Finding |
| --- | --- | --- | --- | --- | --- |
| 20 | turn on the stove | related | 20: turn on the stove and put the frying pan on it | 0.5 | Training contains stove/pan tasks, including the longer stove+pan task, but not this exact short task. |
| 21 | turn on the stove and put the frying pan on it | exact | 20: turn on the stove and put the frying pan on it | 1.0 | Exact prompt exists in the finetune audit. |
| 22 | close the bottom drawer of the cabinet | partial | 6: open the bottom drawer of the cabinet | 0.8 | Training has open bottom drawer and close top drawer, but not close bottom drawer. |
| 23 | close the bottom drawer of the cabinet and open the top drawer | partial | 6: open the bottom drawer of the cabinet | 0.7 | Training contains drawer open/close components, but not this exact compound sequence. |
| 24 | put the black bowl in the bottom drawer of the cabinet | related | 29: put the black bowl in the top drawer of the cabinet | 0.8 | Training has black-bowl top-drawer insertion and cabinet placement, but not bottom-drawer insertion. |
| 25 | put the black bowl on top of the cabinet | exact | 25: put the black bowl on top of the cabinet | 1.0 | Exact prompt exists in the finetune audit. |
| 26 | put the wine bottle in the bottom drawer of the cabinet | partial | 6: open the bottom drawer of the cabinet | 0.6 | Training has drawer analogs but no wine-bottle object family. |
| 27 | put the wine bottle on the wine rack | absent | 9: put the black bowl on the plate | 0.4 | No close wine-bottle or wine-rack task family appears in the finetune audit. |
| 28 | close the top drawer of the cabinet | exact | 28: close the top drawer of the cabinet | 1.0 | Exact prompt exists in the finetune audit. |
| 29 | put the black bowl in the top drawer of the cabinet | exact | 29: put the black bowl in the top drawer of the cabinet | 1.0 | Exact prompt exists in the finetune audit. |

## Comparison To Official RICL Paper

Source: https://arxiv.org/html/2508.02062v1

- The official RICL paper evaluates with same-task retrieval data: 20 demonstrations are collected for each evaluation task and used as the retrieval buffer.
- Their eval tasks are described as unseen or novel relative to priming/base data, but still generally sit inside the base VLA's broad manipulation envelope.
- The paper explicitly notes that RICL does not typically enable tasks too far beyond what the base VLA can perform.
- Our setup is structurally similar in that eval uses same-task retrieval buffers, but our results split strongly by finetune task-family closeness.

## Conclusions

- Our current evidence supports: retrieval quality is associated with success in matched-context evaluations.
- Our current evidence does not prove: RICL materially improves performance beyond the finetuned policy.
- Increasing retrieval memory from 20 to 40 demos/task improved retrieval-distance statistics but reduced aggregate success from 52% to 49%.
- The 49-demo/1-scene run is useful as a sanity check but is not directly comparable to 10-scene runs.
- The expanded mismatch run produced 36/60 raw success and 36/100 normalized success, with task 29 still at 10/10 despite wrong-but-related context.
- Task 21 collapsed to 0/10 under mismatched context, suggesting it may depend more on correct contextual retrieval or be especially unstable.
- Hard failures are concentrated where finetune overlap is partial or absent.

## Recommended Next Experiments

1. Run no-retrieval or dummy-retrieval controls on the same eval scenes.
2. Compare correct vs mismatched retrieval using identical held-out scene indices.
3. Repeat the expanded mismatch run with the same scene indices as the 20-memory matched run.
4. Add a nearest-action / retrieve-and-play baseline to separate interpolation from learned policy behavior.
