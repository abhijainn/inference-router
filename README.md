# Inference Router

A proof of concept for comparing local and hosted Gemma inference, estimating
request duration, and investigating which workflow tasks can use cheaper,
slower inference without materially delaying the final result.

The implementation benchmarks, describes requests, and makes provisional backend
selections. It also simulates dependency-aware scheduling with dynamic arrivals
offline. Live inference dispatch remains future work.

## Equal-capacity scheduling comparison

[The equal-capacity replay](docs/equal-capacity.md) compares two hosted slots against one hosted plus one local slot on 2,028 unique workloads. Hosted FIFO finishes foreground work in 30.67s on average; the combined mixed policy takes 34.09s while avoiding 63.0% of hosted placements. Additional all-hosted priority controls separate queue ordering from backend placement. These are offline fixture results, not new inference measurements.

## Optional pretrained LTR comparison

The frozen OPT-125M length ranker is available through `python -m inference_router rank` after `python scripts/download_ltr.py` (optional `ltr` dependencies). It uses CPU and performs no Gemma generation. [The 14-prompt comparison](docs/ltr-comparison.md) reports improved output-length ranking but no average scheduler speedup with existing nominations held fixed. LTR remains optional; the default profile policy is unchanged.

## Current timing data

The [14-row corrected timing table and regenerated scheduler replay](docs/corrected-timings.md) use the project owner’s supplied corrections. A [small CSV fixture](examples/corrected-gemma-timings.csv) preserves all 14 pairs. Older policy/scheduler reports are explicitly historical and retain their original inputs. No new inference was run for this correction.

## Precision correction and rerun status

The original timing/scheduling results compare local **Q4_K_M** with hosted
**BF16**. They are not a same-precision hardware benchmark. A matched reported
BF16 setup was verified, but the timing rerun was stopped after system sleep
and thermal emergency events; it is invalidated and supplies no replacement
performance claims. See the [precision audit](docs/precision-audit.md) for
verified metadata, recorded usage, and a safe fresh-run procedure.

## What is implemented

| Component | Status |
|---|---|
| Sequential Ollama adapter | Implemented |
| DeepInfra public API adapter | Implemented |
| `.env` loading and secret exclusions | Implemented |
| Frozen Clairvoyant Model B length prediction | Implemented; weights downloaded separately |
| Separate local and hosted empirical duration profiles | Implemented |
| Predictions recorded before execution and error scoring afterward | Implemented |
| Optional profile updates from completed requests | Implemented; no classifier retraining |
| Combined NVIDIA + Clairvoyant request profiles | Implemented; [rules and results](docs/combined-policy.md) |
| Event-driven offline scheduler with arrivals and dependencies | Implemented; simulated dispatch only |
| Optional NVIDIA task/complexity and GLiClass features | Implemented; [14-prompt comparison](docs/classifiers.md) |
| Budget-based local/hosted policy and offline paired replay | Implemented; [results and usage](docs/policy.md) |
| Live dispatch, workflow deadlines, persistent queue | Future work |

See the [experiment report](docs/experiments.md) for all 14 paired Gemma timings,
prediction-versus-actual tables, costs, measured category averages, and limitations.
See [estimation and policy design](docs/estimation.md) for the equations and
routing objective. See [policy results](docs/policy.md) for hosted calls avoided,
budget misses, and comparisons with always-local and always-hosted.

## Setup

Python 3.10+ runs the benchmark clients without third-party dependencies.
Use **Python 3.14** for the optional predictor: ONNX Runtime did not provide a
Python 3.15 wheel in the tested setup. Ollama must already be installed/running.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Edit `.env` locally:

```dotenv
DEEPINFRA_API_KEY=your_key_here
```

The CLI loads this file automatically. Existing shell environment values take
precedence; `unset DEEPINFRA_API_KEY` removes an older shell override.
The template is blank, and `.env` / `.env.*` are ignored by Git.

Create a key with inference permission in the [DeepInfra dashboard](https://deepinfra.com/dash/api_keys).
Use its public API; no dedicated endpoint or GPU rental is needed. The adapter
rechecks public model metadata and prices. A catalog check does not validate
the key; a successful `chat` request does and is billable.

```bash
ollama pull gemma3:4b-it-q4_K_M
python -m inference_router check
python -m inference_router --backend deepinfra check
python -m inference_router --backend deepinfra chat "Return only the word OK." --max-tokens -1
```

Local defaults to `gemma3:4b-it-q4_K_M`; hosted defaults to
`google/gemma-3-4b-it`. Local Q4_K_M and hosted BF16 are different precision
and artifacts. These experiments compare deployments, not hardware in isolation.

## Reproduce the paired benchmark

Generate the synthetic request files locally. The 11 calibration and 3 validation
prompts are committed as readable Python source in [examples/make_requests.py](examples/make_requests.py).
Generated inputs and all result logs are ignored, not committed.

```bash
python examples/make_requests.py

python -m inference_router --timeout 0 run examples/generated/calibration.jsonl \
  --max-tokens -1 --repeat 1 --save-responses --output runs/gemma-local.jsonl

python -m inference_router --backend deepinfra --timeout 0 run examples/generated/calibration.jsonl \
  --max-tokens -1 --repeat 1 --save-responses --output runs/gemma-hosted.jsonl

python -m inference_router build-timing-profile runs/gemma-local.jsonl --output profiles/gemma-local.json
python -m inference_router build-timing-profile runs/gemma-hosted.jsonl --output profiles/gemma-hosted.json
```

Global options go before `check`, `chat` or `run`. Runs append results; use fresh
output filenames for separate experiments. Local execution has an excluded
warmup; hosted execution has no additional warmup request. Avoid other Ollama
activity during the local experiment.

`--max-tokens -1` removes Ollama's explicit output-token cap. For DeepInfra it
omits `max_tokens`; provider/model limits still apply. `done_reason: length`
is truncation, even when the HTTP request succeeds. No automatic retries or
response continuation are performed. `--timeout 0` disables the client socket
timeout; interrupting the client does not guarantee server-side cancellation.

## Predict and validate

```bash
python -m pip install -r requirements-estimator.txt
python scripts/download_estimator.py

python -m inference_router estimate "Explain retries and idempotency." \
  --timing-profile profiles/gemma-local.json \
  --timing-profile profiles/gemma-hosted.json

python -m inference_router --timeout 0 run examples/generated/validation.jsonl \
  --max-tokens -1 --estimate --timing-profile profiles/gemma-local.json \
  --save-responses --output runs/validation-local.jsonl

python -m inference_router --backend deepinfra --timeout 0 run examples/generated/validation.jsonl \
  --max-tokens -1 --estimate --timing-profile profiles/gemma-hosted.json \
  --save-responses --output runs/validation-hosted.jsonl

python -m inference_router evaluate-predictions runs/validation-local.jsonl
python -m inference_router evaluate-predictions runs/validation-hosted.jsonl
```

The download is pinned and SHA-256 checked, about 1.6 MB, and stored under the
ignored `.models/` directory. No training or inference API call is needed for
`estimate`. See [third-party provenance](THIRD_PARTY_NOTICES.md).

Add `--update-timing-profile` to a measured run to update its profile **after**
recording/scoring the prediction. Omit it for a fixed validation profile.
A missing profile can be bootstrapped with that option; initial estimates are
honestly null. Profiles are configuration-specific and retain at most 100
eligible observations. Do not use concurrent writers for the same profile.

## Tests and repository contents

```bash
python -m unittest discover -s tests -v
```

HTTP tests use simulated local servers and never require a real API key or
paid inference. Predictor tests run when optional dependencies and downloaded
weights are available; otherwise they are skipped. The original full setup
passed 34 tests. CI tests both paths.

Raw responses, JSON/JSONL measurements, generated profiles, model weights,
environments, and credentials are intentionally excluded. Results are preserved
as lightweight Markdown tables. This repository contains no API key.

See [the policy experiment explanation](docs/policy.md#what-this-experiment-was-trying-to-answer) for the goal, implementation changes, failure examples, and next steps.

## Optional prompt classifiers

NVIDIA and GLiClass now expose local-only request features through `classify`.
See [setup and measured predictions](docs/classifiers.md). Both recognize the six
clearest task prompts; GLiClass's tested custom attribute configuration selects
every attribute for every prompt and should not drive routing. Running both has
not demonstrated a routing improvement. They do not change the budget policy.

## Combined profiles and dynamic scheduling

`profile` combines NVIDIA and Clairvoyant without estimating seconds.
`simulate` evaluates arrivals, dependencies, and local/hosted capacity offline.
See [the implementation and replay report](docs/combined-policy.md) for all
14 profiles, five strategy comparisons, traces, setup, and limitations.
The original three scenarios show the value of workflow context, with no observed
incremental classifier benefit over the context-only ablation. This is not
a live inference scheduler.

The [expanded feature-value experiment](docs/profile-value.md) tests 4,056 synthetic
workloads using cached real predictions, plus separately labeled controlled
examples. It reports helpful, harmful, unchanged, and speed/call-count tradeoffs;
the original scheduling rules are unchanged.
