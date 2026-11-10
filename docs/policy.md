# Budget policy and paired Gemma replay

The standalone policy now selects a backend without executing a request. It prefers local inference when its adjusted estimate fits the task budget; otherwise it chooses hosted if that estimate fits. If neither fits, it returns `neither_fits` with no backend. Missing estimates can return `uncertain`. Dispatch, dependency scheduling, persistent queues, and automatic deadline allocation remain future work.

## What this experiment was trying to answer

The intended tradeoff is to reserve hosted inference for requests that need its speed and run other requests locally to avoid an API call. A request is a good fit for local when it can finish within the acceptable wait, not simply because it is long or complex. For example, a 30-second local job can fit a 60-second budget while a 3-second local job can fail a 2-second budget.

This experiment tested a minimal version of that idea: **Can the existing duration estimates choose local often while meeting the same per-request budgets as hosted?** It deliberately did not add dollar-cost estimates, workflow dependencies, traffic simulation, or a learned scheduling model. The acceptable wait was a configured test input, not something inferred from the request text.

We reused completed paired measurements, made a simulated choice for each request, and looked up how long the chosen backend had actually taken. We did not send each request again. Consequently, “calls avoided” means calls the policy would have avoided in this replay; it is not a claim that those historical paid benchmark calls were refunded or never occurred.

## What changed in the repository

| Part | Purpose |
|---|---|
| `inference_router/policy.py` | A decision function using backend estimates, eligibility, optional queue waits, task budget, and allowance. Actual outcomes cannot enter this function. |
| `inference_router/policy_replay.py` | Pairs logged measurements, creates retrospective estimates when explicitly requested, and scores policy, always-local, and always-hosted. |
| `policy` command | Returns a backend choice and explanation from numeric estimates or the existing two timing profiles. It does not execute inference. |
| `replay-policy` command | Reproduces comparisons from private logs and outputs aggregate tables or detailed results. |
| `tests/test_policy.py` | Checks decisions, accounting, offline commands, and exclusion of the tested prompt from retrospective calibration. |

Clairvoyant was not retrained or replaced. Its short/medium/long output-length probabilities were still converted to estimated seconds using each backend's measured length representatives and speed. The policy then applied a 25% allowance to those estimates. The timing profiles were varied for leave-one-prompt-out evaluation only; that evaluation is not an online adaptation experiment.

## Decision rule

For each eligible backend:

    adjusted_seconds[b] = additional_queue_seconds[b] + safety_factor * estimated_seconds[b]
    margin_seconds[b] = budget_seconds - adjusted_seconds[b]

Choose local when its margin is nonnegative; otherwise choose hosted when its margin is nonnegative; otherwise return no route. Backend eligibility is available in the Python API. The caller supplies the acceptable task budget; a prompt cannot reliably reveal how long its user is willing to wait. Separate local and hosted profiles supply duration estimates. Clairvoyant remains frozen and supplies length probabilities, not deadlines or routing labels. The existing empirical conversion supplies each backend's estimated seconds.

The default safety factor of 1.25 is an illustrative allowance, **not** a calibrated confidence bound. It inflates duration but does not multiply separately supplied queue wait. Additional queue wait defaults to zero and must exclude any provider wait already included in hosted estimates. This replay is independent-request evaluation, not a contention or scheduler simulation. Local preference expresses lower incremental API spend; no monetary or electricity cost model is needed.

## Evaluation design

Budgets of 5, 20, and 60 seconds and a factor of 1.25 were fixed for this comparison. Each budget applies separately to every request; it is not a shared workflow budget. These settings were not tuned to obtain good replay results. All 14 Gemma pairs completed naturally, with one observed completion per backend and prompt. Local is Ollama Gemma 3 4B Q4_K_M; hosted is DeepInfra Gemma 3 4B BF16. Different precision and generated answer lengths limit causal speed comparisons. Quality equivalence has not been established and remains an eligibility assumption.

The primary evaluation uses three held-out requests with predictions recorded before their executions. The secondary evaluation adds 11 calibration prompts using retrospective leave-one-prompt-out predictions: each prompt's outcomes on each backend are excluded from that backend's timing profile. Every calibration observation with identical message content is removed, including aliases and repeats. The three held-out predictions remain unchanged. This secondary set is not 14 independent prospective validation requests and is not chronological online learning.

Decisions use only forecasts and the configured policy inputs. Actual elapsed times are read afterward to score the selected backend. No new inference or paid API calls were made.

**Hosted calls avoided** counts only requests routed locally, relative to always-hosted. Declined requests earn no avoidance credit. **Budget misses** counts late selected completions plus requests not routed. This prevents abstention from making the policy appear to meet deadlines. “Late” means strictly greater than the budget. Baselines execute every request on their named backend.

## Primary: three held-out requests

| Budget | Strategy | Hosted calls avoided | Hosted calls | Budget misses | Late completions | Not routed |
|---:|---|---:|---:|---:|---:|---:|
| 5 sec | policy | 0/3 | 0 | 3/3 | 0 | 3 |
| 5 sec | always_local | 3/3 | 0 | 2/3 | 2 | 0 |
| 5 sec | always_hosted | 0/3 | 3 | 2/3 | 2 | 0 |
| 20 sec | policy | 2/3 | 0 | 2/3 | 1 | 1 |
| 20 sec | always_local | 3/3 | 0 | 2/3 | 2 | 0 |
| 20 sec | always_hosted | 0/3 | 3 | 0/3 | 0 | 0 |
| 60 sec | policy | 3/3 | 0 | 0/3 | 0 | 0 |
| 60 sec | always_local | 3/3 | 0 | 0/3 | 0 | 0 |
| 60 sec | always_hosted | 0/3 | 3 | 0/3 | 0 | 0 |

At 20 seconds the policy selects local for code and classification, and declines the incident request. Code runs late. Classification actually takes about 0.34 seconds locally, but is estimated at 15.85 seconds before the allowance. The incident hosted estimate is 17.51 seconds, so the allowance raises it above 20 even though its observed hosted completion is 14.24 seconds. The policy therefore does not outperform always-local at this budget, and always-hosted meets the budget for all three requests in this scenario.

## Secondary: 14 pairs, retrospective leave-one-prompt-out

| Budget | Strategy | Hosted calls avoided | Hosted calls | Budget misses | Late completions | Not routed |
|---:|---|---:|---:|---:|---:|---:|
| 5 sec | policy | 0/14 | 0 | 14/14 | 0 | 14 |
| 5 sec | always_local | 14/14 | 0 | 8/14 | 8 | 0 |
| 5 sec | always_hosted | 0/14 | 14 | 8/14 | 8 | 0 |
| 20 sec | policy | 9/14 | 1 | 8/14 | 4 | 4 |
| 20 sec | always_local | 14/14 | 0 | 6/14 | 6 | 0 |
| 20 sec | always_hosted | 0/14 | 14 | 3/14 | 3 | 0 |
| 60 sec | policy | 14/14 | 0 | 0/14 | 0 | 0 |
| 60 sec | always_local | 14/14 | 0 | 0/14 | 0 | 0 |
| 60 sec | always_hosted | 0/14 | 14 | 0/14 | 0 | 0 |

At 20 seconds, avoiding 9 hosted calls does not establish success: the policy has 8 misses, compared with 6 for always-local and 3 for always-hosted. Four policy requests are late and four are not routed. At 5 seconds it declines every request despite six observed requests fitting each baseline. At 60 seconds all local observations fit, so always-local achieves exactly the same savings and misses as the policy.

The implementation is testable, but these observations do **not** validate the present estimator as a useful routing policy. The next improvement should target request-specific length estimation, particularly constrained short outputs, and test on additional held-out prompts. Adding a fixed multiplier cannot repair large classification errors. Keep the budget explicit; infer backend duration from measurements rather than treating “long task” as “suited to slow inference.” Threshold tuning on these same 14 outcomes would require a fresh validation set before claiming improvement.

## Per-request decisions at 20 seconds

The following is the secondary replay. Estimates below are before the 1.25 allowance; outcomes never enter the decision.

| Request | Local estimate | Hosted estimate | Choice | Local actual | Hosted actual |
|---|---:|---:|---|---:|---:|
| architecture-medium-01 | 11.48s | 10.42s | local | 14.39s | 7.49s |
| classify-01 | 15.49s | 13.76s | local | 0.27s | 0.40s |
| code-01 | 22.43s | 19.58s | neither_fits | 3.58s | 1.26s |
| code-review-01 | 15.23s | 14.41s | local | 51.70s | 35.43s |
| constraint-plan-01 | 21.40s | 16.07s | neither_fits | 32.21s | 45.54s |
| evidence-synthesis-01 | 14.29s | 11.56s | local | 32.48s | 39.17s |
| explain-01 | 13.69s | 12.15s | local | 3.88s | 2.48s |
| extract-01 | 9.68s | 8.23s | local | 1.03s | 1.44s |
| holdout-classify-01 | 15.85s | 14.18s | local | 0.34s | 0.25s |
| holdout-code-01 | 13.82s | 12.07s | local | 28.87s | 17.98s |
| holdout-incident-01 | 20.05s | 17.51s | neither_fits | 30.18s | 14.24s |
| incident-01 | 15.76s | 14.97s | local | 28.54s | 13.67s |
| runbook-medium-01 | 22.86s | 20.25s | neither_fits | 15.46s | 11.94s |
| summarize-01 | 16.94s | 15.28s | hosted | 2.45s | 1.48s |

## Why the policy made poor choices

There were three separate failure modes at a 20-second budget:

| Failure mode | Concrete example | Consequence |
|---|---|---|
| Overestimating a short answer | Held-out classification: local forecast 15.85s versus actual 0.34s. With the allowance it looks like 19.81s. | It barely qualifies at 20s and is rejected at 5s, despite actually being fast. |
| Underestimating a longer answer | Held-out code: local forecast 13.82s, adjusted to 17.28s, versus actual 28.87s. Hosted actually finishes in 17.98s. | The policy chooses local and misses a budget that hosted could meet. |
| Allowance pushes a feasible estimate over the cutoff | Held-out incident: hosted forecast 17.51s becomes 21.89s after the allowance; actual hosted time is 14.24s. | Neither adjusted estimate fits, so this policy returns no route even though hosted could meet the budget. |

The three held-out requests therefore produce one successful local completion, one late local completion, and one declined request at 20s. They avoid two hosted calls but incur two budget misses. Always-hosted completes all three within 20s; always-local incurs two misses while avoiding all three hosted calls.

Across the expanded 14-pair replay, the policy sends nine requests locally, one to hosted, and declines four. Five local requests and the one hosted request meet the budget; four local requests run late. The four declined requests are also failures for the headline metric, making eight total misses. The single hosted selection is `summarize-01`, whose local completion actually fit too; this replay did not successfully reserve hosted calls for the tasks that needed them.

Using the observed outcomes only as an after-the-fact diagnostic, three requests exceed 20s on **both** backends: code review, constraint planning, and evidence synthesis. No backend choice could meet 20s for those particular completions. The other five policy misses were avoidable in these observations:

| Request | Policy action | Backend that actually met 20s |
|---|---|---|
| `code-01` | Declined | Local or hosted |
| `runbook-medium-01` | Declined | Local or hosted |
| `holdout-incident-01` | Declined | Hosted |
| `incident-01` | Local, late | Hosted |
| `holdout-code-01` | Local, late | Hosted |

This diagnostic uses information unavailable at decision time and is not an achievable prediction guarantee. It explains why always-hosted has three misses while the policy has eight. It also isolates a real opportunity in these observations: eight requests fit locally, and three additional requests fit only on hosted. The current estimates did not identify that split accurately.

The 5s and 60s scenarios expose the same problem from different directions. At 5s, inflated forecasts cause the policy to decline every request, although six observed local completions fit. At 60s, all 14 local completions fit, so a policy provides no advantage over simply choosing local for everything.

## What we learned and what remains open

- The decision rule is implemented and its behavior is verified. Passing 44 software tests establishes implementation behavior, not estimation accuracy or policy usefulness.
- On these measurements, the current policy offers no improvement over the tested baselines. Nine avoided hosted calls alone would be a misleading success metric because it comes with eight misses and four declined tasks.
- The estimated durations need better separation between constrained short answers and longer completions. The classifier's coarse length probabilities and their empirical conversion are plausible contributors, but this replay does not isolate which component causes each error. A useful next diagnostic is predicted versus actual output length, followed by backend duration error conditional on length.
- The 1.25 allowance is a design choice, not a statistically established reliability level. It can reject feasible work and cannot reliably repair large underestimates. Choosing a different multiplier on these same results would be tuning, requiring new held-out evaluation.
- Returning no route is an explicit experimental behavior. A production application would need a defined response, such as reporting an infeasible deadline, asking for a relaxed budget, or dispatching best effort. Dispatching anyway would require a separate measured policy comparison; it is not what these tables evaluate.
- A fuzzy suitability score could express margin or uncertainty, but changing the label does not remove forecast errors. Its usefulness still needs evaluation against hosted use and budget failures.
- Workflow scheduling can eventually supply budgets from dependency slack and deadlines. It will also need queue and contention estimates. That does not fix the request-duration errors observed here, so scheduler work is not required to diagnose this first policy.

The evidence is preliminary: three prospective held-out requests plus 11 retrospectively evaluated calibration prompts, one paired sample each, no quality scoring, no confidence intervals, no concurrency testing, and different local/hosted precision and answer lengths. The next evaluation should use fresh varied prompts and repeated paired measurements, retain predictions before execution, and report quality eligibility alongside deadline performance and hosted use. No new inference was run for this documentation update.

## Run the policy

Numeric estimates require no predictor dependencies, credentials, or network:

```bash
python -m inference_router policy --budget-seconds 20 --local-seconds 25 --hosted-seconds 10
```

Use existing backend profiles to estimate and decide on a new prompt without executing it (requires optional predictor setup):

```bash
python -m inference_router policy --budget-seconds 20 \
  --prompt "Review this Python function for correctness: ..." \
  --local-profile profiles/gemma-local.json \
  --hosted-profile profiles/gemma-hosted.json
```

Profiles must represent the intended deployments. This offline command verifies profile backend types but does not contact servers to check that deployed settings still match.

Replay the saved prospective validation predictions:

```bash
python -m inference_router replay-policy \
  --local-log runs/gemma-validation-20261110-local.jsonl \
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl \
  --budgets 5 20 60 --safety-factor 1.25 --format markdown
```

For retrospective calibration replay, supply calibration logs and a matching generated prompt catalog:

```bash
python examples/make_requests.py
python -m inference_router replay-policy \
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \
  --mode leave-one-out \
  --calibration-local runs/gemma-calibration-combined-20261110-local.jsonl \
  --calibration-hosted runs/gemma-calibration-combined-20261110-hosted.jsonl \
  --requests examples/generated/calibration.jsonl --format markdown
```

That last command replays the 11 calibration pairs only. The 14-pair table above additionally appends the three validation rows to each input log and combines both prompt catalogs; it preserves their saved predictions. Do not append component calibration logs to combined logs: they contain the same measurements. Raw logs and profiles remain private/ignored, so a fresh clone contains prompts, code, and aggregate reports, but no original measurements.

Without `--format markdown`, replay emits detailed JSON with decisions and diagnostics. `--output PATH` saves either format. Missing pairs, duplicate job/repeat keys, mismatched request hashes, failed/truncated rows, and missing recorded estimates are rejected. Leave-one-out mode is explicit because retrospective estimates must not be confused with recorded prospective predictions.

## Verification

Unit tests cover inclusive budget boundaries, preference, hosted fallback, eligibility, unknown estimates, queues, invalid inputs, baseline scoring, abstention accounting, pair validation, prompt exclusion, preservation of recorded estimates, and offline CLI use. Changing actual outcomes leaves policy decisions unchanged. The full router suite passes with the optional predictor installed. Replay output contains no prompts or generated responses. Existing `.env`, raw data, and model-weight exclusions remain in place.
