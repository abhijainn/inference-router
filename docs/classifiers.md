# NVIDIA and GLiClass: existing-prompt comparison

## Purpose and scope

We added two optional frozen classifiers to describe requests before workflow scheduling: NVIDIA for task type and complexity dimensions, GLiClass for task type and custom output attributes. Neither was trained to decide local versus hosted for our Gemma deployments. This experiment asks whether they supply distinct, usable features, not whether a new routing policy saves calls or meets workflow deadlines.

We ran the existing 11 calibration and 3 validation prompts through both classifiers on CPU with four threads, one model warmup on a separate greeting prompt, and three consecutive measured repeats per prompt/model. NVIDIA uses one forward pass; GLiClass uses two: an 11-way task softmax and six independent attribute sigmoid scores. Processing was sequential, NVIDIA then GLiClass, with both models resident. Timings include tokenization, prediction, score conversion, and truncation accounting; they exclude loading and warmup. There are no new Gemma completions or paid API calls. Prior calibration/validation names identify the prompt sets only: this is a new exploratory classifier assessment on all 14, not independent held-out routing validation.

Model revisions and package versions are pinned. Both load local safetensors only after the explicit download step; no remote Python code is executed. NVIDIA's backbone is constructed from its pinned base configuration and all checkpoint keys are loaded strictly. Its published code-generation example is reproduced (task probability approximately 0.766669, reasoning approximately 0.0632). The adapter retains complete task distributions rather than upstream's rounded top-two display. GLiClass's attribute scores are checked against the library pipeline. An initial Transformers 5 smoke run was discarded; the measured experiment uses compatible Transformers 4.57.6 and GLiClass 0.1.12, with `pip check` passing.

## Models and runtime

| Model | Pinned revision | CPU median per prompt | Load time |
|---|---|---:|---:|
| nvidia/prompt-task-and-complexity-classifier | `fea1121511eafabaf7dd6fc66863dcb04f74defb` | 304.4 ms | 3.62s |
| knowledgator/gliclass-small-v1.0 | `21edefaf7951f68c68c505f9139ba536d3b448f7` | 84.4 ms | 0.47s |

Measured package versions: torch 2.15.0, transformers 4.57.6, gliclass 0.1.12. Platform: macOS-15.6-arm64-arm-64bit-Mach-O.

Combined median sequential prediction time: **400.2 ms** across 42 paired repetitions. Initialization imports can be charged to whichever model loads first; load times are not isolated cold-start benchmarks. The model files total approximately 1.3 GB and remain ignored. CPU overhead can be comparable to our sub-second Gemma classification completions, so additional classifiers must justify their cost.

Both adapters use a 512-token limit. NVIDIA prepends `Prompt: `, matching its example. GLiClass counts candidate-label tokens as part of its input budget and lowercases label text through the library's uni-encoder preparation. Every prompt here fits without truncation. Multi-message inputs preserve roles, but these 14 are single-user prompts. The adapters report token limits for new inputs. NVIDIA marks truncation; GLiClass returns `input_too_long` with no prediction if prompt plus labels exceed 512 tokens, preventing loss of candidate labels. Truncated NVIDIA features should be treated cautiously. A generic tokenizer regex warning appears with this Transformers version; we do not apply the suggested Mistral-specific patch to DeBERTa. Matching NVIDIA's published example and the GLiClass library path is a compatibility check, not proof of correctness on arbitrary inputs.

## Every task prediction

Probabilities are each model's own scores and are not calibrated fast/slow suitability probabilities. They should not be averaged across models.

| Prompt | NVIDIA task (probability) | Reasoning | Constraints | Composite complexity | GLiClass task (probability) |
|---|---|---:|---:|---:|---|
| extract-01 | Extraction (0.403) | 0.014 | 0.840 | 0.256 | Extraction (0.603) |
| classify-01 | Classification (0.674) | 0.005 | 0.563 | 0.185 | Classification (0.702) |
| summarize-01 | Summarization (0.363) | 0.016 | 0.850 | 0.257 | Summarization (0.644) |
| explain-01 | Text Generation (0.528) | 0.071 | 0.947 | 0.428 | Chatbot (0.233) |
| code-01 | Code Generation (0.508) | 0.091 | 0.942 | 0.350 | Code Generation (0.514) |
| incident-01 | Text Generation (0.485) | 0.221 | 0.918 | 0.468 | Brainstorming (0.401) |
| code-review-01 | Text Generation (0.448) | 0.334 | 0.725 | 0.484 | Code Generation (0.838) |
| constraint-plan-01 | Open QA (0.340) | 0.542 | 0.842 | 0.506 | Code Generation (0.440) |
| evidence-synthesis-01 | Closed QA (0.297) | 0.347 | 0.518 | 0.437 | Open QA (0.516) |
| runbook-medium-01 | Text Generation (0.724) | 0.185 | 0.911 | 0.493 | Text Generation (0.360) |
| architecture-medium-01 | Text Generation (0.840) | 0.117 | 0.901 | 0.486 | Code Generation (0.437) |
| holdout-incident-01 | Text Generation (0.727) | 0.142 | 0.885 | 0.481 | Code Generation (0.294) |
| holdout-code-01 | Code Generation (0.691) | 0.149 | 0.869 | 0.347 | Code Generation (0.700) |
| holdout-classify-01 | Classification (0.249) | 0.004 | 0.210 | 0.116 | Classification (0.327) |

NVIDIA also returns creativity, contextual knowledge, domain knowledge, few-shot count, and the checkpoint's no-label-reason score, plus all 12 task probabilities (including Unknown). GLiClass returns all 11 candidate task probabilities. Full numeric records stay in the ignored local results file.

## Reference comparison and complementarity

References in `examples/classifier_reference.py` were authored from prompt instructions before this 14-prompt run, without using completion timings or model predictions. They are single-author exploratory annotations, not an independent benchmark. Some long analysis prompts have ambiguous categories; allowed task alternatives are explicit. The six clearest requests (extraction, two classifications, summarization, and two code-generation requests) are reported separately.

| Feature configuration | Clear-task reference agreement | All-prompt allowed-task agreement | Attribute behavior |
|---|---:|---:|---|
| nvidia alone | 6/6 | 9/14 | No custom attribute heads |
| gliclass alone | 6/6 | 8/14 | All six custom labels selected on every prompt at 0.5 |
| Both, retaining separate features | No additional clear-task coverage | No fused decision rule evaluated | Supplies disagreement and additional scores, not a demonstrated routing gain |

The two task labels agree on **7/14** prompts. An after-the-fact oracle that picked whichever model matches our reference would cover 10/14; this is not a deployable fusion result because the correct reference is unknown at decision time. GLiClass adds an accepted task label only for code review, which NVIDIA calls Text Generation and GLiClass calls Code Generation. Conversely, GLiClass calls the architecture review and held-out incident Code Generation, where NVIDIA's labels are accepted by our reference. A blanket override toward GLiClass's code label would therefore introduce errors.

NVIDIA does add distinguishable continuous signals: its reasoning score is low on the simple classifications and higher on constraint planning. This is useful descriptive separation, but neither the reasoning score nor the composite is a prediction of output length, latency, urgency, or benefit from hosted execution. Task labels alone do not separate the tiny `code-01` response from the much larger `holdout-code-01` response; output expectations and Clairvoyant remain separate inputs.

## GLiClass custom-attribute test

Labels and threshold 0.5 were fixed before the 14-prompt run. These are overlapping attributes, not mutually exclusive classes. Full wording:

- `tiny_answer`: the requested answer is a single category or a few extracted fields.
- `brief_answer`: the requested answer is a short summary or brief explanation.
- `source_code`: the requested answer includes source code.
- `detailed_answer`: the requested answer is a detailed analysis, plan, review, or design.
- `structured_format`: the answer must be JSON, a table, a list, or numbered sections.
- `evidence_or_tradeoffs`: the task asks for tradeoffs, alternative explanations, assessment of evidence, or comparison of design choices.

| Prompt | Tiny | Brief | Source code | Detailed | Structured | Evidence/tradeoffs |
|---|---:|---:|---:|---:|---:|---:|
| extract-01 | 0.9955 | 0.9999 | 0.9997 | 0.9855 | 1.0000 | 1.0000 |
| classify-01 | 0.9997 | 0.9999 | 0.9982 | 0.9914 | 1.0000 | 1.0000 |
| summarize-01 | 0.9950 | 1.0000 | 0.9986 | 0.9869 | 0.9998 | 1.0000 |
| explain-01 | 0.9871 | 1.0000 | 0.9997 | 0.9903 | 0.9998 | 1.0000 |
| code-01 | 0.9948 | 0.9999 | 1.0000 | 0.9894 | 0.9991 | 1.0000 |
| incident-01 | 0.9609 | 1.0000 | 0.9996 | 0.9980 | 0.9998 | 1.0000 |
| code-review-01 | 0.8306 | 0.9999 | 1.0000 | 0.9770 | 0.9997 | 0.9999 |
| constraint-plan-01 | 0.9934 | 1.0000 | 0.9997 | 0.9970 | 0.9999 | 1.0000 |
| evidence-synthesis-01 | 0.9365 | 0.9999 | 0.9996 | 0.9976 | 0.9999 | 1.0000 |
| runbook-medium-01 | 0.9723 | 1.0000 | 0.9999 | 0.9966 | 1.0000 | 1.0000 |
| architecture-medium-01 | 0.9971 | 1.0000 | 0.9998 | 0.9982 | 0.9999 | 1.0000 |
| holdout-incident-01 | 0.9890 | 1.0000 | 0.9998 | 0.9976 | 0.9999 | 1.0000 |
| holdout-code-01 | 0.9194 | 1.0000 | 1.0000 | 0.9955 | 0.9995 | 1.0000 |
| holdout-classify-01 | 0.9997 | 1.0000 | 0.9995 | 0.9900 | 0.9999 | 1.0000 |

GLiClass selects all **84/84** prompt/attribute combinations. The manual references mark 23/84 positive: this yields 23 true positives, 61 false positives, zero false negatives, and zero true negatives. Precision is 27.4%, recall 100%, and specificity 0%—identical to an always-positive baseline. In particular, source-code scores are high even on classification and extraction. These attributes therefore add no discriminating binary signal in this configuration. Different raw scores do not establish useful separation without a separate validation exercise.

This is a failure of the tested checkpoint/label/threshold configuration, not evidence that every GLiClass variant is unsuitable. It reproduces the official library's sigmoid scoring path; the model's published example also emits high scores for several unrelated labels. We did not tune the threshold, rewrite labels against these results, select another checkpoint, or retrofit a fast/slow rule to improve this table. Such changes need a new evaluation set.

## Decision from this experiment

Use NVIDIA's task and individual complexity signals as candidate features; keep the optional GLiClass adapter available for experiments. Do not enable the tested GLiClass custom attributes in a routing policy. Both models together expose disagreement and one complementary task prediction, but **there is no demonstrated routing improvement from running both**. A selective second-classifier policy or alternative GLiClass formulation could be tested later; it has not been validated here.

No fast/slow routing labels, workflow deadlines, backend occupancy, or workflow completion measurements were introduced. We therefore make no claim about hosted calls avoided, budget misses, or workflow speed from these classifiers. The budget-policy experiment remains separate. The useful next test is a fresh prompt set with explicit task/output-shape references and a comparison of NVIDIA-only versus combined features under the same routing objective, including classifier overhead.

## Reproduce

Use Python 3.14 in a separate environment so these heavier optional dependencies do not affect the lightweight router:

```bash
python3.14 -m venv .venv-classifiers
source .venv-classifiers/bin/activate
python -m pip install -r requirements-classifiers.txt
python scripts/download_classifiers.py
HF_HUB_OFFLINE=1 python -m inference_router classify \
  "Classify the ticket: I was charged twice. Return only a category." \
  --classifiers nvidia gliclass
HF_HUB_OFFLINE=1 python scripts/evaluate_classifiers.py --run \
  --results runs/classifier-comparison.jsonl --output docs/classifiers.md
```

The evaluator runs both on the 14 built-in prompts, warming separately and recording three repeats. Without `--run`, it renders an existing private results file. `classify --input PATH --repeat 3 --output PATH` handles other JSONL job files. Use `--classifiers nvidia` or `--classifiers gliclass` to load only one. Classification returns features and never dispatches Gemma requests or alters the existing budget policy. After setup, classification is local-only; missing assets give a setup error rather than downloading during requests. Neither adapter reads an API key.

Sources: [NVIDIA model card](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier), [GLiClass checkpoint](https://huggingface.co/knowledgator/gliclass-small-v1.0), [GLiClass implementation](https://github.com/Knowledgator/GLiClass). Licenses and provenance are listed in THIRD_PARTY_NOTICES.md. Weights, tokenizer assets, raw JSONL, environment files, and credentials are excluded from Git. Only source, references, and this lightweight report are published.
