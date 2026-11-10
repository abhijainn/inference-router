"""Run the fixed 14-prompt classifier comparison or render its private results."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.make_requests import CALIBRATION_JOBS, VALIDATION_JOBS
from examples.classifier_reference import REFERENCES
from inference_router.classifiers import ATTRIBUTE_LABELS, build_classifiers, classify_job
from inference_router.classifier_assets import MODELS

CORE = {'extract-01','classify-01','summarize-01','code-01','holdout-code-01','holdout-classify-01'}


def render(rows):
    if len(rows) != 14 or {r['job_id'] for r in rows} != set(REFERENCES):
        raise ValueError('Expected exactly the 14 existing prompts without duplicates')
    accepted = {name: sum(r['predictions'][name]['task_type'] in REFERENCES[r['job_id']][0] for r in rows) for name in MODELS}
    agree = sum(r['predictions']['nvidia']['task_type'] == r['predictions']['gliclass']['task_type'] for r in rows)
    both = sum(any(r['predictions'][name]['task_type'] in REFERENCES[r['job_id']][0] for name in MODELS) for r in rows)
    text = '''# NVIDIA and GLiClass: existing-prompt comparison

## Purpose and scope

We added two optional frozen classifiers to describe requests before workflow scheduling: NVIDIA for task type and complexity dimensions, GLiClass for task type and custom output attributes. Neither was trained to decide local versus hosted for our Gemma deployments. This experiment asks whether they supply distinct, usable features, not whether a new routing policy saves calls or meets workflow deadlines.

We ran the existing 11 calibration and 3 validation prompts through both classifiers on CPU with four threads, one model warmup on a separate greeting prompt, and three consecutive measured repeats per prompt/model. NVIDIA uses one forward pass; GLiClass uses two: an 11-way task softmax and six independent attribute sigmoid scores. Processing was sequential, NVIDIA then GLiClass, with both models resident. Timings include tokenization, prediction, score conversion, and truncation accounting; they exclude loading and warmup. There are no new Gemma completions or paid API calls. Prior calibration/validation names identify the prompt sets only: this is a new exploratory classifier assessment on all 14, not independent held-out routing validation.

Model revisions and package versions are pinned. Both load local safetensors only after the explicit download step; no remote Python code is executed. NVIDIA's backbone is constructed from its pinned base configuration and all checkpoint keys are loaded strictly. Its published code-generation example is reproduced (task probability approximately 0.766669, reasoning approximately 0.0632). The adapter retains complete task distributions rather than upstream's rounded top-two display. GLiClass's attribute scores are checked against the library pipeline. An initial Transformers 5 smoke run was discarded; the measured experiment uses compatible Transformers 4.57.6 and GLiClass 0.1.12, with `pip check` passing.

## Models and runtime

| Model | Pinned revision | CPU median per prompt | Load time |
|---|---|---:|---:|
'''
    for name in MODELS:
        times=[v for r in rows for v in r['predictions'][name]['timing_samples_seconds']]
        text+=f"| {MODELS[name][0]} | `{MODELS[name][1]}` | {statistics.median(times)*1000:.1f} ms | {rows[0]['model_load_seconds'][name]:.2f}s |\n"
    joint=[sum(r['predictions'][name]['timing_samples_seconds'][i] for name in MODELS) for r in rows for i in range(3)]
    versions = rows[0].get('package_versions', {})
    text += "\nMeasured package versions: " + ", ".join(f"{name} {value}" for name, value in versions.items()) + ". Platform: " + rows[0].get('platform', 'not recorded') + ".\n"
    text+=f"\nCombined median sequential prediction time: **{statistics.median(joint)*1000:.1f} ms** across 42 paired repetitions. Initialization imports can be charged to whichever model loads first; load times are not isolated cold-start benchmarks. The model files total approximately 1.3 GB and remain ignored. CPU overhead can be comparable to our sub-second Gemma classification completions, so additional classifiers must justify their cost.\n"
    text+='''
Both adapters use a 512-token limit. NVIDIA prepends `Prompt: `, matching its example. GLiClass counts candidate-label tokens as part of its input budget and lowercases label text through the library's uni-encoder preparation. Every prompt here fits without truncation. Multi-message inputs preserve roles, but these 14 are single-user prompts. The adapters report token limits for new inputs. NVIDIA marks truncation; GLiClass returns `input_too_long` with no prediction if prompt plus labels exceed 512 tokens, preventing loss of candidate labels. Truncated NVIDIA features should be treated cautiously. A generic tokenizer regex warning appears with this Transformers version; we do not apply the suggested Mistral-specific patch to DeBERTa. Matching NVIDIA's published example and the GLiClass library path is a compatibility check, not proof of correctness on arbitrary inputs.

## Every task prediction

Probabilities are each model's own scores and are not calibrated fast/slow suitability probabilities. They should not be averaged across models.

| Prompt | NVIDIA task (probability) | Reasoning | Constraints | Composite complexity | GLiClass task (probability) |
|---|---|---:|---:|---:|---|
'''
    for r in rows:
        n,g=r['predictions']['nvidia'],r['predictions']['gliclass']
        text+=f"| {r['job_id']} | {n['task_type']} ({n['task_probability']:.3f}) | {n['reasoning']:.3f} | {n['constraint_ct']:.3f} | {n['prompt_complexity_score']:.3f} | {g['task_type']} ({g['task_probabilities'][g['task_type']]:.3f}) |\n"
    text+='''
NVIDIA also returns creativity, contextual knowledge, domain knowledge, few-shot count, and the checkpoint's no-label-reason score, plus all 12 task probabilities (including Unknown). GLiClass returns all 11 candidate task probabilities. Full numeric records stay in the ignored local results file.

## Reference comparison and complementarity

References in `examples/classifier_reference.py` were authored from prompt instructions before this 14-prompt run, without using completion timings or model predictions. They are single-author exploratory annotations, not an independent benchmark. Some long analysis prompts have ambiguous categories; allowed task alternatives are explicit. The six clearest requests (extraction, two classifications, summarization, and two code-generation requests) are reported separately.

| Feature configuration | Clear-task reference agreement | All-prompt allowed-task agreement | Attribute behavior |
|---|---:|---:|---|
'''
    for name in MODELS:
        core=sum(r['predictions'][name]['task_type'] in REFERENCES[r['job_id']][0] for r in rows if r['job_id'] in CORE)
        text+=f"| {name} alone | {core}/6 | {accepted[name]}/14 | {'No custom attribute heads' if name=='nvidia' else 'All six custom labels selected on every prompt at 0.5'} |\n"
    text+=f"| Both, retaining separate features | No additional clear-task coverage | No fused decision rule evaluated | Supplies disagreement and additional scores, not a demonstrated routing gain |\n\nThe two task labels agree on **{agree}/14** prompts. An after-the-fact oracle that picked whichever model matches our reference would cover {both}/14; this is not a deployable fusion result because the correct reference is unknown at decision time. GLiClass adds an accepted task label only for code review, which NVIDIA calls Text Generation and GLiClass calls Code Generation. Conversely, GLiClass calls the architecture review and held-out incident Code Generation, where NVIDIA's labels are accepted by our reference. A blanket override toward GLiClass's code label would therefore introduce errors.\n"
    text+='''
NVIDIA does add distinguishable continuous signals: its reasoning score is low on the simple classifications and higher on constraint planning. This is useful descriptive separation, but neither the reasoning score nor the composite is a prediction of output length, latency, urgency, or benefit from hosted execution. Task labels alone do not separate the tiny `code-01` response from the much larger `holdout-code-01` response; output expectations and Clairvoyant remain separate inputs.

## GLiClass custom-attribute test

Labels and threshold 0.5 were fixed before the 14-prompt run. These are overlapping attributes, not mutually exclusive classes. Full wording:

'''
    for key,label in ATTRIBUTE_LABELS.items():
        text+=f"- `{key}`: {label}.\n"
    text+='\n| Prompt | Tiny | Brief | Source code | Detailed | Structured | Evidence/tradeoffs |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        text+='| '+r['job_id']+' | '+' | '.join(f"{r['predictions']['gliclass']['attribute_scores'][key]:.4f}" for key in ATTRIBUTE_LABELS)+' |\n'
    positives=sum(len(v[1]) for v in REFERENCES.values())
    text+=f"\nGLiClass selects all **84/84** prompt/attribute combinations. The manual references mark {positives}/84 positive: this yields {positives} true positives, {84-positives} false positives, zero false negatives, and zero true negatives. Precision is {positives/84:.1%}, recall 100%, and specificity 0%—identical to an always-positive baseline. In particular, source-code scores are high even on classification and extraction. These attributes therefore add no discriminating binary signal in this configuration. Different raw scores do not establish useful separation without a separate validation exercise.\n"
    text+='''
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
HF_HUB_OFFLINE=1 python -m inference_router classify \\
  "Classify the ticket: I was charged twice. Return only a category." \\
  --classifiers nvidia gliclass
HF_HUB_OFFLINE=1 python scripts/evaluate_classifiers.py --run \\
  --results runs/classifier-comparison.jsonl --output docs/classifiers.md
```

The evaluator runs both on the 14 built-in prompts, warming separately and recording three repeats. Without `--run`, it renders an existing private results file. `classify --input PATH --repeat 3 --output PATH` handles other JSONL job files. Use `--classifiers nvidia` or `--classifiers gliclass` to load only one. Classification returns features and never dispatches Gemma requests or alters the existing budget policy. After setup, classification is local-only; missing assets give a setup error rather than downloading during requests. Neither adapter reads an API key.

Sources: [NVIDIA model card](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier), [GLiClass checkpoint](https://huggingface.co/knowledgator/gliclass-small-v1.0), [GLiClass implementation](https://github.com/Knowledgator/GLiClass). Licenses and provenance are listed in THIRD_PARTY_NOTICES.md. Weights, tokenizer assets, raw JSONL, environment files, and credentials are excluded from Git. Only source, references, and this lightweight report are published.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--results',default='runs/classifier-comparison.jsonl')
    parser.add_argument('--output',default='docs/classifiers.md')
    args=parser.parse_args()
    if args.run:
        models=build_classifiers(['nvidia','gliclass'])
        for model in models.values():
            model.predict([{'role':'user','content':'Write a short greeting.'}])
        destination=Path(args.results);destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('w') as output:
            for job in CALIBRATION_JOBS+VALIDATION_JOBS:
                row=classify_job(job,models,repeat=3)
                row['model_load_seconds']={name:m.load_seconds for name,m in models.items()}
                row['platform']=platform.platform()
                row['package_versions']={name:importlib.metadata.version(name) for name in ['torch','transformers','gliclass']}
                output.write(json.dumps(row)+'\n');output.flush()
                print('Classified',job['id'],flush=True)
    rows=[json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(render(rows));print('Saved',destination)


if __name__=='__main__':
    main()
