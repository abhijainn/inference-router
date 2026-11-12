# Pretrained LTR versus Clairvoyant

## Method and provenance

This is a frozen, zero-shot predictor comparison on the 14 existing prompts. No Gemma generation, hosted inference calls, retraining, outcome-fitted thresholds, or prompt rewriting were performed. The single selected checkpoint was fixed before looking at its results: `LLM-ltr/OPT-Predictors`, `opt-125m-llama3-8b-lmsys-score-trainbucket10-b32`, revision `39df2b41ffe88d5ed967c6035d3838b5b5960379`. Its 250,502,584-byte safetensors file has SHA-256 `e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab`. OPT tokenizer revision: `27dcfa74d334bc871f3234de431e71c6eeba5dd6`.

The authors trained inverse-length labels: higher raw LTR logits mean shorter outputs. We negate logits for a consistent **higher means longer** score. It is not a token count, time estimate, confidence, or probability. We reproduce the single-score `OPTForSequenceClassification` architecture with pinned local assets; all weights must load without missing/unexpected keys. The 14 inputs each have one user message, passed as plain text. The 2,048-token input limit matches the authors’ usage config; truncated inputs are rejected by this evaluation. Inference uses CPU float32, one thread, one prompt at a time, with gradients disabled. The published weight file is float16; float32 CPU inference avoids a GPU requirement. It is a transfer test from Llama-3-8B/LMSYS training to Gemma outputs, not a Gemma-trained predictor.

Sources: [authors’ code](https://github.com/hao-ai-lab/vllm-ltr/tree/13bbf6ff3dab661791d41362551b089e5f77c91c), [checkpoints](https://huggingface.co/LLM-ltr/OPT-Predictors/tree/39df2b41ffe88d5ed967c6035d3838b5b5960379), [paper](https://arxiv.org/abs/2408.15792). Asset weights remain ignored and are not committed.

Clairvoyant is rerun on the same text and checked against its cached probabilities (tolerance 1e-6). Its continuous comparator is `p(medium) + 2*p(long)`, the expected ordinal class, which avoids arbitrary representative token lengths. We also report its existing argmax class ordering. NVIDIA predictions and existing acceleration nominations are frozen.

Observed output tokens and character/4 proxy units come from the original naturally completed local Q4_K_M and hosted BF16 responses. Service durations come from the [owner-corrected timing fixture](corrected-timings.md), not modified raw logs. Nine original pairs lack prompt hashes; explicit legacy mode checks job/model association; every cached profile and new predictor row is hash-checked. Quality is ungraded. Output lengths differ between backends.

## Ranking accuracy

Higher Spearman correlation is better. Pair accuracy compares every pair with different observed lengths, awards 1 for correct order and 0.5 for a predicted tie, and excludes observed ties. These are ranking metrics; there is no valid absolute-time/token MAE for an uncalibrated LTR score. The 14 prompts are a small exploratory sample, not evidence of general superiority.

| Predictor | Target | Backend | Spearman | Pair accuracy | Comparable pairs |
|---|---|---|---:|---:|---:|
| clairvoyant_classes | tokens | local | 0.331 | 60.0% | 90 |
| clairvoyant_classes | proxy_units | local | 0.272 | 57.8% | 90 |
| clairvoyant_classes | seconds | local | 0.331 | 59.9% | 91 |
| clairvoyant_classes | tokens | hosted | 0.242 | 56.7% | 90 |
| clairvoyant_classes | proxy_units | hosted | 0.183 | 54.4% | 90 |
| clairvoyant_classes | seconds | hosted | 0.247 | 57.7% | 91 |
| clairvoyant_score | tokens | local | 0.121 | 55.6% | 90 |
| clairvoyant_score | proxy_units | local | 0.073 | 53.3% | 90 |
| clairvoyant_score | seconds | local | 0.090 | 53.8% | 91 |
| clairvoyant_score | tokens | hosted | 0.024 | 51.1% | 90 |
| clairvoyant_score | proxy_units | hosted | -0.037 | 47.8% | 90 |
| clairvoyant_score | seconds | hosted | 0.020 | 51.6% | 91 |
| ltr | tokens | local | 0.603 | 70.0% | 90 |
| ltr | proxy_units | local | 0.585 | 67.8% | 90 |
| ltr | seconds | local | 0.560 | 65.9% | 91 |
| ltr | tokens | hosted | 0.651 | 72.2% | 90 |
| ltr | proxy_units | hosted | 0.638 | 71.1% | 90 |
| ltr | seconds | hosted | 0.574 | 65.9% | 91 |

`tokens` is the direct output-length target. `proxy_units` matches Clairvoyant’s original class unit. `seconds` is a secondary check against corrected end-to-end service durations, not a latency calibration.

## Every request

Ranks are relative to these 14 prompts (1 = shortest expected output), computed for display only. They are not used as scheduler inputs; the scheduler sees raw continuous scores, so no future cohort or measured outcome is needed. Observed columns are output tokens, not seconds.

| Request | Local tokens | Hosted tokens | Clairvoyant class | Clairvoyant rank | LTR rank | LTR raw score |
|---|---:|---:|---|---:|---:|---:|
| extract-01 | 28 | 28 | short | 1.0 | 3.0 | 2.2522 |
| classify-01 | 3 | 3 | medium | 7.0 | 1.0 | 2.9960 |
| summarize-01 | 72 | 69 | medium | 10.0 | 2.0 | 2.8324 |
| explain-01 | 132 | 141 | medium | 4.0 | 8.0 | -4.4430 |
| code-01 | 116 | 58 | long | 13.0 | 5.0 | 0.2568 |
| incident-01 | 954 | 957 | medium | 6.0 | 11.0 | -5.3162 |
| code-review-01 | 1723 | 1760 | medium | 8.0 | 10.0 | -5.3080 |
| constraint-plan-01 | 1078 | 1207 | long | 12.0 | 9.0 | -4.5389 |
| evidence-synthesis-01 | 1074 | 1076 | medium | 5.0 | 6.0 | -1.8189 |
| runbook-medium-01 | 552 | 573 | long | 14.0 | 13.0 | -6.1067 |
| architecture-medium-01 | 510 | 503 | short | 2.0 | 14.0 | -7.0690 |
| holdout-incident-01 | 1011 | 1035 | long | 11.0 | 7.0 | -4.2213 |
| holdout-code-01 | 977 | 1285 | medium | 3.0 | 12.0 | -5.9730 |
| holdout-classify-01 | 3 | 3 | medium | 9.0 | 4.0 | 1.2560 |

## Scheduler comparison

We reuse the same 4,056 scenarios (2,184 pair cases and 1,872 occupied-lane cases), unchanged arrivals/dependencies/capacities, and corrected durations. The only new optional scheduler input is a continuous length-order score with a declared source; it replaces the short/medium/long comparison in the existing overlap heuristic. It does not add shortest-job-first queue reordering. Existing callers without scores retain the original behavior. Scores are compared only on a matching scale; context-only discards them. No actual durations or future completion timestamps enter dispatch decisions.

- `*_length_only`: all acceleration preferences are neutral. This directly compares length signals without Clairvoyant-based nominations.
- `clairvoyant_classes_combined`: the existing combined policy, unchanged.
- `*_frozen_nominations`: keep the existing combined policy’s NVIDIA/Clairvoyant nominations, replacing only the length comparison. **The LTR hybrid still uses Clairvoyant through those nominations; it is not a complete Clairvoyant replacement.** LTR has no calibrated class probabilities, so we do not manufacture them for the original confidence rules.
- `clairvoyant_score_*`: controls whether any improvement comes simply from retaining Clairvoyant’s probability ordering instead of coarse argmax classes.

All scenario durations are fixed standalone observations. Classifier prediction overhead, thermal state, provider variability, concurrency interference, retries, quality and monetary cost are excluded from simulated finish times. The scenarios repeat these same observations and are not independent new measurements. Calls below mean simulated placements; hosted calls avoided equal local placements. There are no deadlines here, so budget misses are undefined.

| Variant | Mean foreground finish | Hosted placements | Local placements / hosted calls avoided |
|---|---:|---:|---:|
| always_local | 63.09s | 0 | 12,012 |
| always_hosted | 36.73s | 12,012 | 0 |
| occupancy | 40.23s | 5,564 | 6,448 |
| context_only | 36.94s | 4,338 | 7,674 |
| clairvoyant_classes_length_only | 36.77s | 4,395 | 7,617 |
| clairvoyant_score_length_only | 36.77s | 4,424 | 7,588 |
| ltr_length_only | 36.80s | 4,394 | 7,618 |
| clairvoyant_classes_combined | 33.76s | 4,560 | 7,452 |
| clairvoyant_score_frozen_nominations | 33.76s | 4,618 | 7,394 |
| ltr_frozen_nominations | 33.82s | 4,556 | 7,456 |

| Variant vs context only | Better | Equal | Worse | Tradeoff |
|---|---:|---:|---:|---:|
| clairvoyant_classes_length_only | 12 | 3,987 | 26 | 31 |
| clairvoyant_score_length_only | 20 | 3,950 | 55 | 31 |
| ltr_length_only | 12 | 3,988 | 24 | 32 |
| clairvoyant_classes_combined | 744 | 2,933 | 197 | 182 |
| clairvoyant_score_frozen_nominations | 723 | 2,896 | 226 | 211 |
| ltr_frozen_nominations | 736 | 2,934 | 204 | 182 |

“Better” finishes no later with no more hosted placements and one strict improvement; “worse” reverses that relation. “Tradeoff” exchanges time for hosted use.

| Variant | Pair-case mean finish | Occupied-lane mean finish |
|---|---:|---:|
| always_local | 41.63s | 88.13s |
| always_hosted | 25.87s | 49.40s |
| occupancy | 29.83s | 52.36s |
| context_only | 28.69s | 46.56s |
| clairvoyant_classes_length_only | 28.69s | 46.20s |
| clairvoyant_score_length_only | 28.69s | 46.20s |
| ltr_length_only | 28.69s | 46.26s |
| clairvoyant_classes_combined | 27.95s | 40.54s |
| clairvoyant_score_frozen_nominations | 27.95s | 40.53s |
| ltr_frozen_nominations | 27.95s | 40.67s |

## Predictor overhead

One warmup per predictor is excluded. Each prompt is timed three times, alternating predictor order; per-prompt medians are summarized below. LTR is CPU-only; Clairvoyant uses one-thread CPU ONNX. These are observed timings on this Mac, not universal speed claims. Load includes local asset validation and model initialization. A live scheduler would need to account for this overhead, especially for sub-second requests.

| Predictor | Load | Median prompt prediction | Maximum prompt median |
|---|---:|---:|---:|
| ltr | 2.771s | 22.314ms | 57.420ms |
| clairvoyant | 0.389s | 0.142ms | 0.227ms |

## Findings on this fixture

- **Local output ordering:** pair accuracy 60.0% → 70.0%; Spearman 0.331 → 0.603.
- **Hosted output ordering:** pair accuracy 56.7% → 72.2%; Spearman 0.242 → 0.651.
- **Scheduler, frozen nominations:** LTR mean foreground finish 33.82s versus 33.76s for existing Clairvoyant classes (+0.059s); hosted placements 4,556 versus 4,560 (-4).

Direct comparison of the LTR hybrid with the existing combined policy:

| Better | Equal | Worse | Tradeoff |
|---:|---:|---:|---:|
| 58 | 3,850 | 52 | 96 |

The LTR checkpoint improves response-length ordering on this small sample but does not establish a scheduler improvement. Only the overlap branch consumes the new scores; dependency rules and frozen nominations still dominate decisions. Its pair-case mean finish is unchanged, with differences concentrated in occupied-lane cases. It also makes mistakes: architecture review is ranked longest, despite several saved reviews/investigations producing longer outputs. Keep LTR as an optional experimental signal rather than changing the default policy based on these 14 prompts. A future comparison could change queue ordering or nomination rules, but that would be a separate policy experiment, not an isolated predictor replacement.

## Use and reproduce

```bash
# Reuse the existing optional model environment; no Gemma or GPU is needed.
python -m pip install -e '.[ltr,estimator]'
python scripts/download_ltr.py
python -m inference_router rank 'Return only one category for this support ticket.'
python scripts/evaluate_ltr.py --run --allow-legacy-no-hash \
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \
  --local-log runs/gemma-validation-20261110-local.jsonl \
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl
```

The optional scheduler profile fields are `length_order_score` (higher means longer) and `length_order_source` (a shared, named predictor scale). The `rank` command emits both fields; merge them into an existing request profile before simulation. They do not replace NVIDIA task labels or produce a new acceleration preference.

Omit `--run` to replay cached predictor results without loading either predictor. `rank --input PATH --output PATH` supports JSONL request files. LTR is optional and does not replace the default Clairvoyant/NVIDIA profiler. Full responses, private logs, weights, and JSON prediction caches remain ignored. A fresh clone needs the original saved outputs/profiles to reproduce these exact historical results; it does not include those private artifacts. No model selection or thresholds were tuned on these 14 outcomes.
