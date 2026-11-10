# Gemma local versus hosted experiment

Experiment date: November 10, 2026 (America/Chicago). All numbers below are from real completed requests. Raw logs and generated JSON profiles are excluded from Git; these tables are a compact snapshot. The synthetic prompts are preserved in `examples/make_requests.py`.

## Setup and method

| Setting | Local | Hosted |
|---|---|---|
| Backend | Ollama on Apple M4 MacBook Air, 16 GB unified memory | DeepInfra public API, standard tier |
| Model | `gemma3:4b-it-q4_K_M` | `google/gemma-3-4b-it` |
| Precision | Q4_K_M | BF16 |
| Artifact identity | `68ceab7e468a31c988b44d5a4984a0e21f58c4f7158c6ce29c1804babac38df7` | No verifiable artifact digest exposed |
| Ollama version | 0.40.16 | Not applicable |
| Temperature / seed | 0 / 42 | 0 / 42 |
| Context | 4,096 configured | Provider context; not forced to 4,096 |
| Concurrency | One active request | One active request |
| Warmup | Excluded from measurement rows | No extra warmup request |
| Streaming | Disabled | Disabled |
| Output limit | `num_predict=-1` | `max_tokens` omitted; provider limits remain |

Eleven prompts per backend built the timing profiles: five basic tasks, four complex tasks, and two medium-length tasks. Three different prompts per backend then tested fixed profiles. Validation outcomes did not update their own profiles. Each prompt ran once per backend: these are single observations, not repeated-run means or percentiles. All 28 measured requests reported natural stopping; answer correctness was not automatically graded.

## Actual end-to-end timings

| Request | Phase | Local seconds | Hosted seconds | Local output tokens | Hosted output tokens |
|---|---|---:|---:|---:|---:|
| Extraction | calibration | 1.03 | 1.44 | 28 | 28 |
| Classification | calibration | 0.27 | 0.40 | 3 | 3 |
| Summarization | calibration | 2.45 | 1.48 | 72 | 69 |
| Explanation | calibration | 3.88 | 2.48 | 132 | 141 |
| Basic code generation | calibration | 3.58 | 1.26 | 116 | 58 |
| Incident investigation | calibration | 28.54 | 13.67 | 954 | 957 |
| Code review | calibration | 51.70 | 35.43 | 1723 | 1760 |
| Constraint planning | calibration | 32.21 | 45.54 | 1078 | 1207 |
| Evidence synthesis | calibration | 32.48 | 39.17 | 1074 | 1076 |
| Operational runbook | calibration | 15.46 | 11.94 | 552 | 573 |
| Architecture review | calibration | 14.39 | 7.49 | 510 | 503 |
| Validation: incident | validation | 30.18 | 14.24 | 1011 | 1035 |
| Validation: code | validation | 28.87 | 17.98 | 977 | 1285 |
| Validation: classification | validation | 0.34 | 0.25 | 3 | 3 |

Sum of measured request durations: **245.39 seconds local** and **192.76 seconds hosted**. Hosted had a lower duration in 10 of 14 pairs; its summed duration was 21.4% lower. These sums are not workflow makespans. Output lengths differ, so they do not isolate hardware performance.

## Measured duration-profile inputs

| Backend | Short count / mean length | Medium count / mean length | Long count / mean length | Rate, length units/sec | Overhead, sec |
|---|---:|---:|---:|---:|---:|
| Local | 5 / 71.40 | 2 / 638.00 | 4 / 1323.00 | 38.09 | 0.381 |
| Hosted | 5 / 58.00 | 2 / 654.00 | 4 / 1294.75 | 42.28 | 0.000 |

Length units are characters divided by four, not actual model tokens. Local rate uses server generation time; hosted rate uses end-to-end time. Hosted overhead is zero to avoid counting network/input/hidden waiting twice. The averages above replace arbitrary representative lengths. Only two observations populate each medium category, so these estimates are provisional.

## Duration prediction validation

| Fresh request | Local predicted | Local actual | Hosted predicted | Hosted actual |
|---|---:|---:|---:|---:|
| Validation: incident | 20.05 sec | 30.18 sec | 17.51 sec | 14.24 sec |
| Validation: code | 13.82 sec | 28.87 sec | 12.07 sec | 17.98 sec |
| Validation: classification | 15.85 sec | 0.34 sec | 14.18 sec | 0.25 sec |

| Metric, three fresh requests each | Local | Hosted |
|---|---:|---:|
| Mean absolute error | 13.57 sec | 7.70 sec |
| Requests underestimated | 2 / 3 | 1 / 3 |

The classification error is material. Clairvoyant predicts coarse output-length probabilities; converting them with measured category means does not correct a poor prediction. These results establish a working measurement pipeline, not accurate general-purpose estimates. Three validation prompts cannot establish calibrated confidence or broad ranking quality. Percentage error is especially unstable for sub-second requests.

## Hosted cost

| Item | Recorded value |
|---|---:|
| Input tokens, all 14 hosted requests | 1,485 |
| Output tokens, all 14 hosted requests | 8,698 |
| Standard input list rate | $0.05 / million tokens |
| Standard output list rate | $0.10 / million tokens |
| Estimated token-usage cost | $0.00094405 |

The cost is calculated from reported usage and public catalog rates at the experiment, not a billing invoice. Any account funding minimum is separate. Local electricity and hardware costs were not measured. Current prices and availability must be rechecked before later paid runs.

## Findings and limits

- Local and hosted profiles need separate timing and response-length observations; one deployment's profile cannot stand in for the other.
- The hosted option was faster overall in this pilot, with four request-level exceptions. Treating hosted as the fast option remains a POC convention.
- Nonidentical precision, runtimes, context configuration, hidden caching/queueing, and differing answers prevent an exact hardware comparison.
- Clairvoyant stayed frozen. No model training was performed; only empirical conversion statistics were calculated.
- Real API calls and all simulated infrastructure tests are separate evidence. The original complete setup passed 34 tests.
- No deadlines, dependency-graph executions, queue-load sweeps, energy measurements, systematic answer-quality scores, or suitability/routing labels were collected.
- Next tests should measure repeatability, compare task-history baselines and fresh requests, and evaluate workflow delay versus API cost saved.

## Sources and reproduction

- [DeepInfra model and published prices](https://deepinfra.com/google/gemma-3-4b-it)
- [DeepInfra API](https://docs.deepinfra.com/chat/overview)
- [Local Ollama model](https://ollama.com/library/gemma3:4b-it-q4_K_M)
- [Clairvoyant pretrained artifact](https://huggingface.co/Aravind0495/clairvoyant-scheduler)
- [Setup and reproduction commands](../README.md)
- [Estimator equations and policy scope](estimation.md)

## Budget-policy replay

See [policy results](policy.md) for the three held-out and expanded 14-pair comparisons against always-local and always-hosted. The current estimator does not demonstrate a routing advantage under the tested budgets. No additional paid calls were made.

The [expanded policy explanation](policy.md#why-the-policy-made-poor-choices) separates unavoidable misses from avoidable routing errors and explains why hosted calls avoided alone is insufficient.
