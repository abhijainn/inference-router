# Where request profiles change scheduling

## Question and frozen policy

We tested whether NVIDIA/Clairvoyant information changes backend choices beyond the context-only scheduler. No scheduler rules, model weights, profile thresholds, or earlier measurements were changed to improve these results. No new paid inference calls were made.

The existing policy uses classifier features in two places: a combined `consider_hosted` nomination can break equal-depth dependency ties or nominate independent foreground work for acceleration; output-length ordering can decide whether a task waits locally or uses spare hosted capacity. NVIDIA task type also affects profile uncertainty through the combined rules, but the scheduler does not otherwise route by task label. NVIDIA constraint scores are recorded but currently have no scheduling effect. This experiment tests the implemented feature paths, not the broader potential of every model output.

## Separate evidence sets

**Controlled mechanism checks:** three tiny hand-authored workloads with synthetic service times and constructed NVIDIA/Clairvoyant probabilities. These show how the policy behaves when signals are informative, and when they are wrong. They are not predictions from the pretrained checkpoints and do not demonstrate their accuracy.

**Real-profile stress sweep:** reuse all 14 cached pretrained profiles and their paired Gemma service-time observations. For every ordered pair of distinct prompts, test second-task arrival offsets 0, 1, and 5 seconds; hosted capacities 1 and 2; and independent foreground tasks versus an AND-join followed by a fixed final classification-like task. There are 14 × 13 × 3 × 2 × 2 = **2,184 pair cases**, each evaluated under seven variants. Pair order reverses which prompt gets task ID `a` versus `b`, exercising deterministic tie order in both directions. The pair sweep initially showed no length-only effect, so we explicitly extended this exploratory experiment with **1,872 occupied-lane cases**: code review is a foreground anchor, every ordered pair of the other 13 prompts supplies background local work and a foreground arrival, with the same offsets, capacities, and two topologies. The join waits for the two foreground tasks, not unrelated background work. This brings the total to **4,056 cases**. The extension tests a previously dormant branch and is not held-out confirmation; it concentrates on the one current acceleration nomination. This is a finite stress matrix, not independent new measurements or a representative traffic distribution. Topologies and offsets are illustrative choices.

The measured outcomes are private to the simulator and consulted only after dispatch. Future task arrivals and completion timestamps are not visible to the policy. Submitted join dependencies may refer to a task whose payload has not arrived; that is declared workflow context, not access to its future profile. Scenarios have one local worker. Durations are fixed standalone observations, so concurrency interference, provider variability, quality differences, retries, and model-loading costs are not simulated. Profile inference is cached and its overhead is excluded; wall-clock benefits are therefore optimistic relative to classifying live requests. The nine earliest prompt pairs lack logged request hashes; explicit legacy mode associates those outcomes by job ID/model. Newer pairs and every cached profile have matching catalog hashes.

## Comparisons

| Variant | Information used |
|---|---|
| Context only | Dependencies, arrivals, foreground/background intent, occupancy; neutral preference and medium length for every task |
| Length signal only | Same rules plus actual predicted length classes; all acceleration nominations removed |
| Nomination only | Same rules plus combined profile preference; all length classes replaced with medium |
| Combined | Existing combined preference and actual predicted length class |
| Always local / always hosted | One backend, with the same dependency/capacity rules |
| Occupancy | Local-first FIFO by arrival/task ID, spilling to hosted when local is occupied |

These are **mechanism ablations**, not clean NVIDIA-only versus Clairvoyant-only models. Nominations already depend on both classifiers. Length-only retains raw argmax classes even for uncertain profiles, matching the scheduler's current use of weak length signals. We do not turn observed lengths or execution times into oracle profiles.

A case is **better** if it finishes foreground work no later and uses no more hosted calls, with at least one strict improvement. **Worse** reverses that relation. **Equal** matches both metrics. **Tradeoff** means faster with more calls or slower with fewer calls. Times use a 1e-9 comparison tolerance. These labels do not impose an arbitrary dollar-versus-delay weighting. Routing changed means at least one task changes backend; ordering/waiting differences can occur without a backend change. Hosted-call count is not a dollar-cost estimate.

## Controlled checks

| Case | Variant | Foreground finish | Hosted calls |
|---|---|---:|---:|
| nomination_breaks_equal_depth_tie | context_only | 40.50s | 1 |
| nomination_breaks_equal_depth_tie | length_signal_only | 40.50s | 1 |
| nomination_breaks_equal_depth_tie | nomination_only | 15.50s | 1 |
| nomination_breaks_equal_depth_tie | combined | 15.50s | 1 |
| length_changes_wait_or_hosted_choice | context_only | 20.00s | 1 |
| length_changes_wait_or_hosted_choice | length_signal_only | 20.00s | 1 |
| length_changes_wait_or_hosted_choice | nomination_only | 11.00s | 2 |
| length_changes_wait_or_hosted_choice | combined | 7.00s | 2 |
| wrong_nomination_hurts | context_only | 15.50s | 1 |
| wrong_nomination_hurts | length_signal_only | 15.50s | 1 |
| wrong_nomination_hurts | nomination_only | 40.50s | 1 |
| wrong_nomination_hurts | combined | 40.50s | 1 |

- **Equal-depth tie:** a small classification-like branch and a larger analysis branch feed a join. A useful nomination sends the analysis branch to hosted and the small branch locally, rather than accelerating whichever task ID sorts first.
- **Wait versus spare hosted slot:** a medium-output foreground task runs on hosted while background work occupies local. A new long-output foreground task arrives. Length information can reject the assumption that existing hosted work provides enough overlap and start the new task on a spare hosted slot. With one hosted slot, that option disappears.
- **Wrong nomination:** a genuinely large task is described as short, while a genuinely small task is nominated for acceleration. The same rule accelerates the wrong branch. This is intentionally a counterexample, showing how classifier errors can reverse the benefit.

## Real cached-profile sweep

| Variant | Better | Equal | Worse | Tradeoff | Cases with backend changes | Mean foreground finish | Mean hosted calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| context_only | 0 | 4056 | 0 | 0 | 0 | 34.92s | 1.024 |
| length_signal_only | 0 | 3991 | 29 | 36 | 65 | 34.82s | 1.040 |
| nomination_only | 960 | 3000 | 80 | 16 | 1056 | 31.40s | 1.028 |
| combined | 911 | 2935 | 93 | 117 | 1121 | 31.20s | 1.060 |
| always_local | 188 | 208 | 0 | 3660 | 3848 | 56.91s | 0.000 |
| always_hosted | 0 | 0 | 1834 | 2222 | 4056 | 36.73s | 2.962 |
| occupancy | 476 | 1092 | 2266 | 222 | 2964 | 38.55s | 1.359 |

All comparisons above are against context-only. Means weight every synthetic case equally, count reused observations repeatedly, and are descriptive only. They are not confidence estimates or expected production savings.

### Combined results by workload condition

| Condition | Cases | Better | Equal | Worse | Tradeoff | Backend changes |
|---|---:|---:|---:|---:|---:|---:|
| family=pair_matrix | 2184 | 104 | 2064 | 0 | 16 | 120 |
| family=occupied_lanes | 1872 | 807 | 871 | 93 | 101 | 1001 |
| slots=1 | 2028 | 480 | 1500 | 40 | 8 | 528 |
| slots=2 | 2028 | 431 | 1435 | 53 | 109 | 593 |
| offset=0 | 1352 | 364 | 988 | 0 | 0 | 364 |
| offset=1 | 1352 | 266 | 970 | 53 | 63 | 382 |
| offset=5 | 1352 | 281 | 977 | 40 | 54 | 375 |
| topology=independent | 2028 | 885 | 998 | 64 | 81 | 1030 |
| topology=join | 2028 | 26 | 1937 | 29 | 36 | 91 |

### Incremental length contribution

Comparing combined against nomination-only isolates the current raw-length mechanism: 130 cases change backend; better=0, equal=3926, worse=58, tradeoff=72. These remain mechanism ablations, not a claim about independent model accuracy.
In those changed cases, adding length finishes faster in 72, at the same time in 46, and slower in 12. In this particular sweep all 130 changes add one hosted call: 46 add no final-completion benefit and 12 are slower as well. This directly exposes unnecessary acceleration while other work is still determining completion.

### Selected real-profile examples

#### better: `architecture-medium-01__code-review-01__arrival0__slots1__join`

| Strategy | Foreground finish | Hosted calls | Dispatch order (task → backend) |
|---|---:|---:|---|
| context_only | 52.04s | 1 | a → hosted at 0.00s, b → local at 0.00s, join → local at 51.70s |
| combined | 35.77s | 1 | b → hosted at 0.00s, a → local at 0.00s, join → local at 35.43s |

These examples were selected after the full sweep for the largest absolute finish-time difference in their category; they are illustrations, not an unbiased sample. In pair cases task `a` is the first named prompt and `b` the second. In occupied-lane cases `a` is code review, `b` the named background prompt, and `c` the named arriving foreground prompt. The join uses the saved held-out classification duration.

#### worse: `occupied__constraint-plan-01__evidence-synthesis-01__arrival1__slots1__independent`

| Strategy | Foreground finish | Hosted calls | Dispatch order (task → backend) |
|---|---:|---:|---|
| context_only | 51.70s | 1 | a → local at 0.00s, c → hosted at 1.00s, b → local at 51.70s |
| combined | 64.69s | 1 | a → hosted at 0.00s, b → local at 0.00s, c → local at 32.21s |

These examples were selected after the full sweep for the largest absolute finish-time difference in their category; they are illustrations, not an unbiased sample. In pair cases task `a` is the first named prompt and `b` the second. In occupied-lane cases `a` is code review, `b` the named background prompt, and `c` the named arriving foreground prompt. The join uses the saved held-out classification duration.

#### tradeoff: `occupied__evidence-synthesis-01__holdout-incident-01__arrival1__slots2__join`

| Strategy | Foreground finish | Hosted calls | Dispatch order (task → backend) |
|---|---:|---:|---|
| context_only | 63.00s | 1 | a → hosted at 0.00s, b → local at 0.00s, c → local at 32.48s, join → local at 62.66s |
| combined | 35.77s | 2 | a → hosted at 0.00s, b → local at 0.00s, c → hosted at 1.00s, join → local at 35.43s |

These examples were selected after the full sweep for the largest absolute finish-time difference in their category; they are illustrations, not an unbiased sample. In pair cases task `a` is the first named prompt and `b` the second. In occupied-lane cases `a` is code review, `b` the named background prompt, and `c` the named arriving foreground prompt. The join uses the saved held-out classification duration.

## What this establishes

The actual cached profiles change at least one backend choice in **1121/4,056** cases. Against context-only they yield 911 better cases, 2935 equal cases, 93 worse cases, and 117 tradeoffs under the two-metric definition. Unlike the earlier three scenarios, the feature paths are now exercised by varied ordering, arrivals, and capacity.

The harmful occupied-lane example does not necessarily mean the code-review nomination was wrong: code review really is lengthy. Accelerating it changes which resource is occupied when another request arrives. With no knowledge of that future arrival, a locally sensible decision can still worsen the eventual foreground finish.

The controlled checks demonstrate plausible mechanisms, while the real sweep tests the current imperfect predictions. Neither establishes a generally better policy. Review the overall results and both helpful and harmful traces before deciding to enable additional routing rules. With 12 of the 14 combined profiles uncertain, weak raw length classes and the single code-review nomination dominate the current feature effects. More varied scenarios cannot repair bad predictions by themselves.

In this sweep, keeping the acceleration nominations while disabling the raw-length overlap rule gives 960 better cases and 80 worse cases relative to context-only; adding length changes this to 911 better and 93 worse, with more tradeoffs. This makes the raw-length overlap rule a candidate for revision, not a validated feature to enable more aggressively. Most gains concentrate on the single code-review nomination, which the occupied-lane extension deliberately anchors; this does not demonstrate general task-type routing ability.

The next credible validation is a fresh prompt/workflow set, repeated service measurements, classifier overhead included, randomized tie identifiers, and a predefined completion-versus-hosted-use objective. Keep context-only, mechanism ablations, and simple baselines. Do not tune rules on this sweep and then report the same sweep as independent validation.

## Verification

The added checks cover useful nomination, the length mechanism with and without spare capacity, a harmful wrong signal, input immutability during ablations, two-metric comparison boundaries, and complete case enumeration with both pair orders. Cached combined profiles are checked against their raw NVIDIA/Clairvoyant predictions and recorded rule settings before evaluation. The full suite passed all 74 tests with the optional real-model checks installed. Test success verifies behavior, not general scheduling performance. No production scheduler behavior was changed.

## Reproduce and inspect

```bash
python scripts/evaluate_profile_value.py \
  --allow-legacy-no-hash \
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \
  --local-log runs/gemma-validation-20261110-local.jsonl \
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl
```

The evaluator uses `runs/combined-profiles.jsonl` from the previous profile run. It needs no model runtime, API credentials, or network. The full stress-case inputs and per-case metrics/backend assignments are written to ignored local files; the public report contains aggregate tables and selected traces only. Every case can be rerun with `python -m inference_router simulate PATH` from the generated spec directory. Controlled cases are explicitly marked in their own directory. Weights and private measurements remain excluded from Git.
