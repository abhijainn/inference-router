# Corrected Gemma timings and offline scheduler replay

## Source and scope

The project owner supplied this corrected 14-row table on November 11, 2026. It is the current timing fixture for this replay. Four local values changed from the original logs: extraction 1.03 → 2.03 seconds; classification 0.27 → 1.27; constraint planning 32.21 → 52.21; evidence synthesis 32.48 → 52.48. All other supplied values are unchanged. These are owner-supplied corrections, not independently verified new measurements. Original raw logs, tokens, phase timings, and saved predictions were preserved.

“Slow inference” means local Ollama `gemma3:4b-it-q4_K_M`; “fast inference” means DeepInfra `google/gemma-3-4b-it`, reported BF16. These are heterogeneous deployment choices, not a matched-precision hardware benchmark. The interrupted local BF16 run is excluded. The original study ran each prompt once per backend, with natural stopping and no automated answer-quality grading. The supplied corrections contain no new outputs or repetitions.

## All 14 requests

| Request | Slow Inference | Fast Inference |
|---|---:|---:|
| Extraction | 2.03s | 1.44s |
| Classification | 1.27s | 0.40s |
| Summarization | 2.45s | 1.48s |
| Explanation | 3.88s | 2.48s |
| Basic code generation | 3.58s | 1.26s |
| Incident investigation | 28.54s | 13.67s |
| Code review | 51.70s | 35.43s |
| Constraint planning | 52.21s | 45.54s |
| Evidence synthesis | 52.48s | 39.17s |
| Operational runbook | 15.46s | 11.94s |
| Architecture review | 14.39s | 7.49s |
| Validation: incident | 30.18s | 14.24s |
| Validation: code | 28.87s | 17.98s |
| Validation: classification | 0.34s | 0.25s |

Sum of displayed values: **287.38s slow/local** and **192.77s fast/hosted**. Hosted is faster in **14/14 pairs**; its summed duration is **32.9% lower**. Sums are not workflow makespans.

## Regenerated scheduler comparison

The unchanged scheduler rules and the original cached NVIDIA/Clairvoyant profiles are replayed against the corrected CSV. Profiles are verified against prompt hashes and frozen combination rules. This recomputes the same 2,184 pair cases and 1,872 occupied-lane cases (4,056 total), with one local worker, one or two hosted slots, arrival offsets 0/1/5 seconds, and independent/join topologies. Case construction and strategy definitions are documented in [the historical mechanism report](profile-value.md); its old numeric tables are superseded for this corrected fixture.

These cases reuse 14 pairs; they are not 4,056 independent measurements. Decisions receive task profiles, dependencies, arrivals and occupancy, not actual durations. Actual durations are consulted only after dispatch. Classifier runtime overhead, concurrency interference, network variability, retries, energy cost, and answer-quality differences are excluded. No inference API calls were made for this replay. “Calls” below are simulated placements, not newly billed requests. There are no workflow deadlines in these scenarios, so budget misses are not defined here.

| Strategy | Mean foreground finish | Hosted placements | Local placements | Hosted calls avoided vs always-hosted |
|---|---:|---:|---:|---:|
| context_only | 36.94s | 4,338 | 7,674 | 7,674 |
| length_signal_only | 36.77s | 4,395 | 7,617 | 7,617 |
| nomination_only | 34.09s | 4,448 | 7,564 | 7,564 |
| combined | 33.76s | 4,560 | 7,452 | 7,452 |
| always_local | 63.09s | 0 | 12,012 | 12,012 |
| always_hosted | 36.73s | 12,012 | 0 | 0 |
| occupancy | 40.23s | 5,564 | 6,448 | 6,448 |

| Strategy vs context only | Better | Equal | Worse | Tradeoff |
|---|---:|---:|---:|---:|
| length_signal_only | 12 | 3,987 | 26 | 31 |
| nomination_only | 768 | 3,002 | 192 | 94 |
| combined | 744 | 2,933 | 197 | 182 |

“Better” means no later foreground completion and no more hosted placements, with at least one strict improvement; “worse” reverses this comparison. “Tradeoff” changes completion time and hosted use in opposite directions. Hosted calls avoided are a usage proxy, not measured dollar savings. Results describe this finite fixture and do not establish production benefit or equal answer quality.

## Reproduce

```bash
python3 scripts/replay_corrected_timings.py \
  --timings examples/corrected-gemma-timings.csv \
  --profiles runs/combined-profiles.jsonl
```

The CSV is public and small. Cached real classifier profiles remain private/ignored as in the original experiment, so a fresh clone must generate those profiles before reproducing the replay. This command loads no model and requires no API key when the cached profiles already exist. Historical duration-estimator and budget-policy reports remain explicitly labeled with their original inputs; they were not recalibrated using end-to-end corrections without replacement phase data.

## Predictor follow-up

[The frozen LTR versus Clairvoyant comparison](ltr-comparison.md) reuses this exact corrected timing fixture and the original saved output lengths. No Gemma requests were repeated. It keeps the existing scheduler result above as the reference baseline.
