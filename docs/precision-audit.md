# Precision audit and interrupted BF16 rerun

## What was verified

The old November 10 results used **Q4_K_M locally and BF16 on DeepInfra**. They are a mixed-precision deployment comparison, not a same-precision hardware benchmark. Their original values remain unchanged and must not be relabeled.

On November 11, 2026, a new setup was verified directly against Ollama model/runtime metadata and DeepInfra's current public catalog:

| Field | Local | Hosted |
|---|---|---|
| Model | `gemma3:4b-it-bf16` | `google/gemma-3-4b-it` |
| Declared weight precision | `BF16` | `bfloat16` |
| Artifact digest | `4e0b0e7a1c0af47d128a4a50588806f73f04c2db911cfef3181294de592e144e` | Not exposed in public catalog |
| Runtime evidence | Ollama reports the BF16 model loaded fully GPU-resident, context 4096 | Public catalog assertion; no per-request dtype attestation |

BF16 and FP16 are not interchangeable. The preflight refuses other formats, missing precision, another model label, or a missing local digest. Completed responses also identify the expected model. This verifies matching **reported weight format**, not identical checkpoint revisions, compute/accumulation dtype, tokenizers/templates, or serving stacks. Precision need not match for a transparently labeled heterogeneous routing study; it must not be falsely claimed to match, and a hardware-only causal comparison requires substantially more control.

## What was attempted

The plan was 14 prompts × 3 repetitions × 2 backends, with natural completion, fixed temperature/seed, shuffled prompt order and alternating backend order. Old logs were preserved. The BF16 model was downloaded alongside the Q4 model rather than replacing it.

There are **23 completed local requests and 24 completed hosted requests**, including one full 14-prompt paired pass and part of repetition two. All recorded successful completions identify the BF16 configurations. One DeepInfra DNS lookup failed; it was audited and explicitly resumed after connectivity recovered, without rerunning completed measurements. A subsequent local request was cancelled when the environmental problem below was discovered.

## Why the timing study was stopped

System power logs recorded idle sleep at **16:42:44 America/Chicago** (248 seconds) and **thermal emergency sleep at 17:05:50** (260 seconds) while the benchmark was in progress. Substantial swap allocation was also observed. The thermal condition's cause was not isolated; the logs do not establish that the model alone caused it.

The benchmark process was interrupted, and its BF16 model was unloaded. Ollama then reported no loaded models. No sleep or thermal protection was bypassed. The downloaded model remains available.

The partial timing run is marked `invalidated_environment` with `resume_allowed=false`. It is **not a valid completed three-repetition replacement dataset**, even though precision metadata matches. Its original observations and failure audit remain private. No new scheduler performance percentages, medians, or blog figures are reported from these timings.

The evaluator refuses this invalidated run. Future runs record both wall-clock and client-clock intervals and stop if they differ by more than five seconds, flagging possible suspension/clock adjustment rather than silently accepting such timings. This is a useful check, not a complete thermal or system-health monitor.

## Usage recorded before stopping

| Item | Value |
|---|---:|
| Successful hosted requests | 24 |
| Hosted input tokens | 2,583 |
| Hosted output tokens | 14,736 |
| Estimated standard-list token cost | $0.001603 |

This estimate uses the catalog rates captured for the run; it is not a billing invoice. The DNS failure produced no completion. There was no GPU rental. Raw responses, provider IDs, local paths, model assets, and the API key are not published.

## How to proceed

Before collecting replacement timings, use a plugged-in, awake, thermally normal machine with the lid open and adequate ventilation. Address memory pressure and begin with a short validation run. Do not force the machine through thermal sleep. Start in a **fresh output directory**; this invalidated dataset cannot be resumed into a clean benchmark.

Metadata can be checked without any generation request:

```bash
python scripts/run_matched_precision.py --check-only
```

When conditions are stable, a fresh paired run can be made:

```bash
python scripts/run_matched_precision.py \
  --output-dir runs/gemma-bf16-paired-fresh --repeat 1
```

A measured run uses the local API key and makes billable hosted requests; the metadata-only check does not. Once a stable repeated dataset exists, the offline evaluator can replay the unchanged policies and regenerate tables. The experiment must still be described as a matched-reported-weight-format **deployment** comparison; the hosted API does not expose enough information to attribute every difference solely to hardware.

## Blog wording supported now

> The initial comparison used Q4_K_M locally and BF16 on the hosted endpoint. A later setup check confirmed that both deployments could report BF16 weights, but the matched-format timing rerun was interrupted by system sleep and thermal conditions and was not used for replacement performance claims.

The earlier experiment reports remain useful as explicitly mixed-precision exploratory replays. They cannot be described as having identical precision or as demonstrating production cost savings.
