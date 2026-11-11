# Combined request profiles and dynamic scheduling

> **Precision scope:** These historical results use local Q4_K_M versus hosted BF16. They are not a matched-precision hardware comparison. The later BF16 timing rerun was invalidated by environmental interruptions; see the [precision audit](precision-audit.md).

## What is implemented

`profile` combines frozen NVIDIA task/complexity outputs with frozen Clairvoyant output-length probabilities. It does not convert them into seconds. The policy preserves each signal and returns a provisional `prefer_local`, `consider_hosted`, or `uncertain` result with reasons. GLiClass is not loaded. The previous fixed-budget policy remains a separate experiment.

`simulate` is an offline event-driven scheduler, not a live Ollama/DeepInfra dispatcher. At every arrival or completion it recomputes readiness, visible dependency depth, and backend occupancy, then dispatches more work if appropriate. Tasks are non-preemptive. It has one local worker and two hosted slots in this experiment. Future arrivals are hidden from decisions. Declared dependency nodes become visible only when submitted; submitted blocked tasks can influence priorities. No additional paid inference calls were made.

## Profile rules, fixed before inspecting combined results

Default thresholds: NVIDIA selected-task probability at least 0.40, Clairvoyant maximum class probability at least 0.55, and reasoning score at least 0.30 for a higher-reasoning flag. These are explicit experimental settings, not trained thresholds or calibrated confidence levels. NVIDIA Other/Unknown, truncated inputs, low scores, or classification/extraction with a non-short length prediction produce `uncertain`.

For remaining cases: short output prefers local; medium/long output with the higher-reasoning flag becomes `consider_hosted`; long output with lower reasoning prefers local as a background candidate. Medium output without a stronger signal remains uncertain. Constraints are retained as a feature but are not multiplied into an arbitrary composite suitability score. No rule guarantees quality or a deadline, and hosted remains the same Gemma family rather than a more capable model.

## All 14 combined profiles

Counts: 1 prefer local, 1 consider hosted, 12 uncertain.

| Prompt | NVIDIA task | Short / medium / long | Reasoning | Preference | Issues |
|---|---|---|---:|---|---|
| extract-01 | Extraction | 0.56 / 0.39 / 0.04 | 0.014 | prefer_local | Likely short output; test local before paying for acceleration. |
| classify-01 | Classification | 0.17 / 0.78 / 0.05 | 0.005 | uncertain | task_and_length_disagree |
| summarize-01 | Summarization | 0.03 / 0.95 / 0.02 | 0.016 | uncertain | uncertain_task_type |
| explain-01 | Text Generation | 0.28 / 0.67 / 0.05 | 0.071 | uncertain | Medium output alone does not establish a backend preference. |
| code-01 | Code Generation | 0.29 / 0.17 / 0.54 | 0.091 | uncertain | uncertain_output_length |
| incident-01 | Text Generation | 0.33 / 0.49 / 0.17 | 0.221 | uncertain | uncertain_output_length |
| code-review-01 | Text Generation | 0.26 / 0.57 / 0.17 | 0.334 | consider_hosted | Substantial output and reasoning signal; accelerate only when workload context justifies it. |
| constraint-plan-01 | Open QA | 0.35 / 0.17 / 0.49 | 0.542 | uncertain | uncertain_task_type, uncertain_output_length |
| evidence-synthesis-01 | Closed QA | 0.33 / 0.56 / 0.11 | 0.347 | uncertain | uncertain_task_type |
| runbook-medium-01 | Text Generation | 0.19 / 0.34 / 0.47 | 0.185 | uncertain | uncertain_output_length |
| architecture-medium-01 | Text Generation | 0.48 / 0.45 / 0.07 | 0.117 | uncertain | uncertain_output_length |
| holdout-incident-01 | Text Generation | 0.27 / 0.34 / 0.39 | 0.142 | uncertain | uncertain_output_length |
| holdout-code-01 | Code Generation | 0.39 / 0.47 / 0.14 | 0.149 | uncertain | uncertain_output_length |
| holdout-classify-01 | Classification | 0.13 / 0.83 / 0.04 | 0.004 | uncertain | uncertain_task_type, task_and_length_disagree |

The high uncertain count is an observed limitation, not a failed software check. The simple classification prompts still receive medium-length forecasts; the combined policy now exposes that conflict rather than claiming accurate durations. These profiles have not been validated as true fast/slow labels. The raw distributions remain available even when a provisional preference is uncertain.

## Dynamic scheduling rules

1. Re-evaluate after each submitted arrival or completion. Run tasks only after all dependencies finish; respect capacity and never preempt work already running.
2. Favor foreground results and their ancestors over unrelated background work. Use the longest visible dependency chain, counted in tasks, as a structural priority. This is not a duration-based critical-path calculation.
3. Consider hosted for a ready task that unlocks a foreground chain, unless an equal-or-deeper chain is already running on hosted. Among equal structural priorities, a `consider_hosted` profile breaks ties. Such a profile can also nominate independent foreground work for hosted execution.
4. Use available local capacity for other ready work. When local is occupied, send waiting foreground work to hosted unless hosted is already running equal-or-deeper foreground work with an equal-or-larger predicted output class. In that case, retain the task for local and reconsider at the next event.
5. Background work uses local capacity unless it is an ancestor of a submitted foreground result. An uncertain profile is neutral, not an automatic hosted escalation; its raw length class is still a weak input to the overlap heuristic.

Rule 4 is an overlap heuristic, not a proof that local work will finish before hosted work. It uses only the ordering short < medium < long, with no assumed token lengths, speeds, seconds, or monetary costs. It does not estimate remaining service time or account for how long a running task has already executed. It can wait too long if predictions are wrong, and hosted work can finish first; the next completion event then revises the decision. Finite replay tasks eventually finish, but ongoing-stream fairness, deadlines, and starvation protection are not established.

The scheduler must receive foreground/background intent and dependencies from a caller/workflow. The classifiers do not invent those facts. A background task that later becomes necessary is not migrated or restarted. This prototype uses queued ready tasks and completion events; it does not implement retries, cancellation, live provider limits, or persistent state.

## Replay design and baselines

The three hand-authored scenarios reuse the previously measured standalone Gemma durations, with no new generation. Profiles were generated from prompt text only. Durations live in a separate outcome mapping consulted by the simulator only after a dispatch decision; the decision function receives no service times or future event timestamps. A decision's trace records which tasks were visible, its backend, and its reason.

- **Background and arrivals:** code review starts as background at time zero; classification arrives at 1s, extraction at 2s. The objective is prompt foreground completion while the background task continues.
- **Parallel join:** code review, classification, and extraction are submitted together; a final classification-like task waits for all three. This represents an AND-join.
- **Dependency chain with background:** code review, code generation, and final classification form a chain, alongside an independent background runbook.

These are synthetic workflows using measured service-time fixtures, not real dependencies or data flow between the original prompts. For example, the join does not consume newly generated review text. Duplicate task use across scenarios is not new independent evidence. The nine earliest calibration prompts predate request hashes in both logs. For these known historical logs we explicitly enable `--allow-legacy-no-hash`, associating outcomes by job ID and model rather than verified message hash; this limits provenance assurance. The other five pairs and all new profiles have matching hashes. Mismatched nonempty hashes are always rejected. The snapshots assume local and hosted executions do not interfere and service time does not change under hosted concurrency. Quality equivalence is assumed, as before.

Baselines share tasks, outcomes, capacity, and dependency enforcement. `always_local` and `always_hosted` use one backend. `occupancy` uses FIFO by arrival/id, local first then hosted, without task/dependency priority. `context_only` uses the same dynamic rules with every profile replaced by neutral preference and medium length. It isolates the additional effect of classifier features. `profile_dynamic` uses the combined profiles. Ties are deterministic; conclusions may change with tie order or other workloads.

Profiles are precomputed in replay. Profile inference overhead is excluded from the timeline and included neither as service nor queue time. The separate classifier report measures that overhead; a live system would need to include it. The following results describe these exact fixtures, not validated production savings.

## Replay results

| Scenario | Strategy | Foreground finished by | All tasks finished by | Mean foreground flow | Hosted calls | Hosted calls avoided | Queue wait total |
|---|---|---:|---:|---:|---:|---:|---:|
| background_and_arrivals | always_local | 53.01s | 53.01s | 50.99s | 0 | 3 | 100.68s |
| background_and_arrivals | always_hosted | 3.44s | 35.43s | 0.92s | 3 | 0 | 0.00s |
| background_and_arrivals | occupancy | 3.44s | 51.70s | 0.92s | 2 | 1 | 0.00s |
| background_and_arrivals | context_only | 3.44s | 51.70s | 0.92s | 2 | 1 | 0.00s |
| background_and_arrivals | profile_dynamic | 3.44s | 51.70s | 0.92s | 2 | 1 | 0.00s |
| parallel_join | always_local | 53.35s | 53.35s | 52.51s | 0 | 4 | 103.68s |
| parallel_join | always_hosted | 35.68s | 35.68s | 18.34s | 4 | 0 | 0.40s |
| parallel_join | occupancy | 52.04s | 52.04s | 26.40s | 2 | 2 | 0.00s |
| parallel_join | context_only | 35.77s | 35.77s | 18.20s | 1 | 3 | 0.27s |
| parallel_join | profile_dynamic | 35.77s | 35.77s | 18.20s | 1 | 3 | 0.27s |
| dependency_chain_with_background | always_local | 80.91s | 96.38s | 71.06s | 0 | 4 | 80.91s |
| dependency_chain_with_background | always_hosted | 53.65s | 53.65s | 47.50s | 4 | 0 | 0.00s |
| dependency_chain_with_background | occupancy | 80.91s | 80.91s | 71.06s | 1 | 3 | 0.00s |
| dependency_chain_with_background | context_only | 53.75s | 53.75s | 47.53s | 2 | 2 | 0.00s |
| dependency_chain_with_background | profile_dynamic | 53.75s | 53.75s | 47.53s | 2 | 2 | 0.00s |

Foreground finish is the timestamp of the last non-background task, relative to time zero; mean foreground flow is finish minus arrival, averaged across those tasks. All-task finish includes background work. Queue wait begins when a task is both submitted and dependency-ready. Hosted calls avoided counts local executions relative to always-hosted, including background tasks. All strategies finish every task—there are no abstention credits or deadline guarantees. No deadline/budget-miss metric is reported because these scenarios specify no deadlines.

## Dispatch traces for the profile policy

### background_and_arrivals

| Time | Task | Backend | Reason |
|---:|---|---|---|
| 0.00s | background | local | Run ready work locally while retaining hosted capacity. |
| 1.00s | ticket | hosted | Local is occupied; serve waiting foreground work on hosted. |
| 2.00s | extract | hosted | Local is occupied; serve waiting foreground work on hosted. |

### parallel_join

| Time | Task | Backend | Reason |
|---:|---|---|---|
| 0.00s | a-review | hosted | Accelerate a visible dependency chain that unlocks foreground work. |
| 0.00s | b-ticket | local | Run ready work locally while retaining hosted capacity. |
| 0.27s | c-extract | local | Run ready work locally while retaining hosted capacity. |
| 35.43s | join | local | Run ready work locally while retaining hosted capacity. |

### dependency_chain_with_background

| Time | Task | Backend | Reason |
|---:|---|---|---|
| 0.00s | a-review | hosted | Accelerate a visible dependency chain that unlocks foreground work. |
| 0.00s | z-background | local | Run ready work locally while retaining hosted capacity. |
| 35.43s | b-code | hosted | Accelerate a visible dependency chain that unlocks foreground work. |
| 53.41s | c-label | local | Run ready work locally while retaining hosted capacity. |


## Interpretation

The profile-informed and context-only scheduler produce identical metrics in all three scenarios. The observed benefits relative to simple baselines come from workflow context, not demonstrated incremental classifier value. The simulator demonstrates the intended behaviors: foreground arrivals can bypass occupied local background work, and other tasks can remain local while hosted execution is already advancing a foreground path. It also exposes that the current profiles are mostly uncertain. A functioning simulator does not establish an optimal or generally useful scheduling algorithm.

Next evaluation should use fresh workflows, varied arrival order, capacity and service-time variability, and profile-inference overhead. Keep context-only as an ablation. Before live dispatch, add real completion notifications, explicit service errors and cancellation semantics, quality eligibility, and a persistent task queue. No live scheduling or automatic spending was enabled here.

## Verification

All 68 tests pass with the optional NVIDIA, GLiClass, and Clairvoyant runtimes installed. Checks include invalid probabilities, rule boundaries, conflicting/truncated profiles, no dependence on duration fields, dependency readiness, capacity, future-arrival isolation, overlap behavior, context-only ablation, outcome-independent initial dispatch, and offline CLI execution. The 14 profiles were also generated using the real frozen NVIDIA and Clairvoyant weights. Passing these tests verifies implementation behavior; it does not validate scheduling optimality or profile accuracy.

## Commands

```bash
# In the optional classifier environment, add Clairvoyant support:
python -m pip install -r requirements-profiles.txt
python scripts/download_estimator.py
python scripts/download_classifiers.py
HF_HUB_OFFLINE=1 python -m inference_router profile "Your prompt"

# Replay the measured fixtures locally (raw logs remain ignored):
HF_HUB_OFFLINE=1 python scripts/evaluate_combined_policy.py --run --allow-legacy-no-hash \
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \
  --local-log runs/gemma-validation-20261110-local.jsonl \
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl

# The evaluator writes ignored workflow specs with separate task/outcome fields:
python -m inference_router simulate examples/generated/parallel_join.json --strategy all
```

`profile --input PATH --output PATH` profiles a JSONL job file. `simulate --hosted-slots N --output PATH` controls capacity and saves detailed traces. The simulator requires no model dependencies or API key. The evaluator without `--run` uses its saved profiles but still verifies their request hashes against the catalog. Thresholds are configurable via the Python `ProfileRules` API and recorded in every profile. GLiClass remains an explicit optional `classify --classifiers gliclass` choice; `classify` now defaults to NVIDIA only.

## Follow-up: where profiles change decisions

The [expanded stress test](profile-value.md) exercises the unchanged rules on
4,056 varied synthetic cases and separate controlled checks. This extends the
three original scenarios; it does not replace their results or establish live
performance. The follow-up includes harmful cases and isolates the raw-length
overlap mechanism from the combined acceleration nomination.
