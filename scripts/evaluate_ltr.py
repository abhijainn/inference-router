"""Compare frozen length rankers on saved Gemma outputs and corrected timings."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_router.ltr import REVISION, SUBDIR, WEIGHT_SHA256, TOKENIZER_REVISION
from inference_router.request_profile import combine_predictions, ProfileRules
from inference_router.scheduler import simulate
from scripts.evaluate_combined_policy import JOBS, read, request_hash, measurements
from scripts.evaluate_profile_value import real_cases, category
from scripts.replay_corrected_timings import load_fixture

VARIANTS = ('always_local', 'always_hosted', 'occupancy', 'context_only',
            'clairvoyant_classes_length_only', 'clairvoyant_score_length_only', 'ltr_length_only',
            'clairvoyant_classes_combined', 'clairvoyant_score_frozen_nominations', 'ltr_frozen_nominations')


def average_ranks(values):
    ranks = [0.] * len(values)
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    for _, group in itertools.groupby(ordered, key=lambda i: values[i]):
        indices = list(group)
        # Positions in ordered, including ties, use one-based average ranks.
        start = min(ordered.index(i) for i in indices) + 1
        rank = start + (len(indices)-1)/2
        for i in indices: ranks[i] = rank
    return ranks


def rank_metrics(predicted, actual):
    if len(predicted) != len(actual) or len(actual) < 2:
        raise ValueError('Require matching vectors with at least two items')
    if any(not math.isfinite(v) for v in predicted+actual):
        raise ValueError('Ranks require finite observations')
    correct = tied = count = 0
    for i,j in itertools.combinations(range(len(actual)), 2):
        if actual[i] == actual[j]: continue
        count += 1
        direction = (predicted[i]-predicted[j]) * (actual[i]-actual[j])
        correct += direction > 0
        tied += predicted[i] == predicted[j]
    a,b = average_ranks(predicted), average_ranks(actual)
    ma,mb = statistics.mean(a), statistics.mean(b)
    denominator = math.sqrt(sum((x-ma)**2 for x in a)*sum((x-mb)**2 for x in b))
    return {'spearman':sum((x-ma)*(y-mb) for x,y in zip(a,b))/denominator if denominator else None,
            'pairwise_accuracy':(correct+.5*tied)/count if count else None,
            'pairs':count, 'correct_pairs':correct, 'predicted_ties':tied}


def scores(clairvoyant, ltr):
    p = clairvoyant['length_class_probabilities']
    return {'clairvoyant_classes':float(('short','medium','long').index(clairvoyant['length_class'])),
            'clairvoyant_score':p['medium']+2*p['long'], 'ltr':ltr['length_order_score']}


def variant_tasks(tasks, variant, by_id):
    tasks = deepcopy(tasks)
    for task in tasks:
        profile = task['profile']
        for name in ['length_order_score','length_order_source']:profile.pop(name, None)
        if variant.endswith('_length_only'):profile['preference'] = 'uncertain'
        source = 'ltr' if variant.startswith('ltr_') else 'clairvoyant_score' if variant.startswith('clairvoyant_score_') else None
        if source:
            profile['length_order_score'] = by_id[profile['predictor_job_id']]['scores'][source]
            profile['length_order_source'] = source
    return tasks


def predict_all(destination, repeats=3):
    from inference_router.ltr import LTRPredictor
    from inference_router.estimator import DurationEstimator
    ranker = LTRPredictor()
    started = time.perf_counter();clairvoyant = DurationEstimator();clair_load = time.perf_counter()-started
    warmup = [{'role':'user','content':'Reply with a greeting.'}]
    ranker.predict(warmup);clairvoyant.predict(warmup)
    rows = []
    for job in JOBS:
        elapsed = {'ltr':[], 'clairvoyant':[]}; predictions = {}
        for repeat in range(repeats):
            order = ['ltr','clairvoyant'] if repeat%2==0 else ['clairvoyant','ltr']
            for name in order:
                predictor = ranker if name=='ltr' else clairvoyant
                t = time.perf_counter();predictions[name] = predictor.predict(job['messages']);elapsed[name].append(time.perf_counter()-t)
        rows.append({'job_id':job['id'], 'request_sha256':request_hash(job), **predictions,
                     'runtime_seconds':{name:statistics.median(values) for name,values in elapsed.items()}})
        print('Predicted', job['id'], flush=True)
    destination = Path(destination);destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'schema_version':1, 'checkpoint':SUBDIR, 'revision':REVISION,
        'weight_sha256':WEIGHT_SHA256, 'tokenizer_revision':TOKENIZER_REVISION,
        'repeats':repeats, 'load_seconds':{'ltr':ranker.load_seconds,'clairvoyant':clair_load}, 'rows':rows}, indent=2))


def evaluate(args):
    fixture = load_fixture(args.timings)
    paired = {r['job_id']:{'local':float(r['slow_seconds']),'hosted':float(r['fast_seconds'])} for r in fixture}
    # Validate model, completion, unique IDs and all available hashes, preserving raw logs.
    measurements(args.local_log, args.hosted_log, args.allow_legacy_no_hash)
    outputs = {backend:{row['job_id']:row for path in paths for row in read(path)}
               for backend,paths in [('local',args.local_log),('hosted',args.hosted_log)]}
    saved = json.loads(Path(args.predictions).read_text())
    if (saved['revision'],saved['weight_sha256'],saved['tokenizer_revision'],saved['checkpoint']) != (REVISION, WEIGHT_SHA256, TOKENIZER_REVISION, SUBDIR):
        raise ValueError('Unexpected LTR checkpoint provenance')
    rows = saved['rows'];catalog = {j['id']:j for j in JOBS}
    cached = read(args.profiles)
    for group in [rows, cached]:
        if len(group)!=len(catalog) or {r['job_id'] for r in group}!=set(catalog):raise ValueError('Require all 14 predictions/profiles without duplicates')
        for row in group:
            if row['request_sha256']!=request_hash(catalog[row['job_id']]):raise ValueError('Prompt hash mismatch')
    cached = {r['job_id']:r for r in cached}
    for row in rows:
        key = row['job_id'];old = cached[key]
        expected = combine_predictions(old['nvidia'], old['clairvoyant'], ProfileRules(**old['profile']['rules']))
        if expected!=old['profile']:raise ValueError('Frozen combined profile mismatch')
        for name,prob in old['clairvoyant']['length_class_probabilities'].items():
            if abs(prob-row['clairvoyant']['length_class_probabilities'][name])>1e-6:raise ValueError('Fresh Clairvoyant differs from cached baseline')
        if row['ltr']['truncated']:raise ValueError('Truncated LTR input; do not silently treat as full-prompt prediction')
        row['scores'] = scores(row['clairvoyant'], row['ltr'])
        row['profile'] = dict(old['profile'], predictor_job_id=key)
        row['observed'] = {}
        for backend in ['local','hosted']:
            raw = outputs[backend][key]
            tokens = raw.get('output_tokens');response = raw.get('response')
            if not isinstance(tokens,int) or tokens<=0 or not isinstance(response,str):raise ValueError('Require saved natural output tokens/text')
            row['observed'][backend] = {'tokens':tokens,'proxy_units':len(response)//4,'seconds':paired[key][backend]}
    by_id = {r['job_id']:r for r in rows};ordered = [by_id[r['job_id']] for r in fixture]
    metrics = {}
    for source in ['clairvoyant_classes','clairvoyant_score','ltr']:
        predicted = [r['scores'][source] for r in ordered]
        metrics[source] = {backend:{target:rank_metrics(predicted,[r['observed'][backend][target] for r in ordered])
                                    for target in ['tokens','proxy_units','seconds']} for backend in ['local','hosted']}
    profiles = {key:row['profile'] for key,row in by_id.items()}
    aggregate = {v:{'finish_sum':0.,'hosted':0,'local':0,'categories':{k:0 for k in ['better','equal','worse','tradeoff']},'vs_existing':{k:0 for k in ['better','equal','worse','tradeoff']},'families':{}} for v in VARIANTS}
    count = 0
    for case in real_cases(profiles,paired):
        count += 1;results = {}
        for variant in VARIANTS:
            strategy = variant if variant in ['always_local','always_hosted','occupancy','context_only'] else 'profile_dynamic'
            tasks = variant_tasks(case['spec']['tasks'],variant,by_id)
            results[variant] = simulate(tasks,case['spec']['outcomes'],strategy,case['slots'])['metrics']
        for variant,m in results.items():
            a = aggregate[variant];a['finish_sum']+=m['foreground_finish_seconds'];a['hosted']+=m['hosted_calls'];a['local']+=m['hosted_calls_avoided']
            a['categories'][category(results['context_only'],m)]+=1
            a['vs_existing'][category(results['clairvoyant_classes_combined'],m)]+=1
            family = a['families'].setdefault(case['family'], {'count':0,'finish_sum':0.,'hosted':0})
            family['count']+=1;family['finish_sum']+=m['foreground_finish_seconds'];family['hosted']+=m['hosted_calls']
    for a in aggregate.values():
        a['mean_finish'] = a.pop('finish_sum')/count
        for family in a['families'].values():family['mean_finish'] = family.pop('finish_sum')/family['count']
    ranks = {s:average_ranks([r['scores'][s] for r in ordered]) for s in ['clairvoyant_score','ltr']}
    result = {'cases':count,'metrics':metrics,'scheduler':aggregate,'rows':ordered,'runtime':{name:{'load_seconds':saved['load_seconds'][name],
        'median_ms':statistics.median(r['runtime_seconds'][name] for r in rows)*1000,
        'max_ms':max(r['runtime_seconds'][name] for r in rows)*1000} for name in ['ltr','clairvoyant']}}
    Path('runs/ltr-comparison-results.json').write_text(json.dumps(result,indent=2))
    text = render(result,ranks)
    Path(args.report).write_text(text)
    print('Saved',args.report,';',count,'offline scenarios; no Gemma inference.')


def render(result,ranks):
    text = '''# Pretrained LTR versus Clairvoyant

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
'''
    for source,backends in result['metrics'].items():
        for backend,targets in backends.items():
            for target,m in targets.items():
                text+=f"| {source} | {target} | {backend} | {m['spearman']:.3f} | {m['pairwise_accuracy']:.1%} | {m['pairs']} |\n"
    text+='''
`tokens` is the direct output-length target. `proxy_units` matches Clairvoyant’s original class unit. `seconds` is a secondary check against corrected end-to-end service durations, not a latency calibration.

## Every request

Ranks are relative to these 14 prompts (1 = shortest expected output), computed for display only. They are not used as scheduler inputs; the scheduler sees raw continuous scores, so no future cohort or measured outcome is needed. Observed columns are output tokens, not seconds.

| Request | Local tokens | Hosted tokens | Clairvoyant class | Clairvoyant rank | LTR rank | LTR raw score |
|---|---:|---:|---|---:|---:|---:|
'''
    for i,r in enumerate(result['rows']):
        text+=f"| {r['job_id']} | {r['observed']['local']['tokens']} | {r['observed']['hosted']['tokens']} | {r['clairvoyant']['length_class']} | {ranks['clairvoyant_score'][i]:.1f} | {ranks['ltr'][i]:.1f} | {r['ltr']['raw_score']:.4f} |\n"
    text+='''
## Scheduler comparison

We reuse the same 4,056 scenarios (2,184 pair cases and 1,872 occupied-lane cases), unchanged arrivals/dependencies/capacities, and corrected durations. The only new optional scheduler input is a continuous length-order score with a declared source; it replaces the short/medium/long comparison in the existing overlap heuristic. It does not add shortest-job-first queue reordering. Existing callers without scores retain the original behavior. Scores are compared only on a matching scale; context-only discards them. No actual durations or future completion timestamps enter dispatch decisions.

- `*_length_only`: all acceleration preferences are neutral. This directly compares length signals without Clairvoyant-based nominations.
- `clairvoyant_classes_combined`: the existing combined policy, unchanged.
- `*_frozen_nominations`: keep the existing combined policy’s NVIDIA/Clairvoyant nominations, replacing only the length comparison. **The LTR hybrid still uses Clairvoyant through those nominations; it is not a complete Clairvoyant replacement.** LTR has no calibrated class probabilities, so we do not manufacture them for the original confidence rules.
- `clairvoyant_score_*`: controls whether any improvement comes simply from retaining Clairvoyant’s probability ordering instead of coarse argmax classes.

All scenario durations are fixed standalone observations. Classifier prediction overhead, thermal state, provider variability, concurrency interference, retries, quality and monetary cost are excluded from simulated finish times. The scenarios repeat these same observations and are not independent new measurements. Calls below mean simulated placements; hosted calls avoided equal local placements. There are no deadlines here, so budget misses are undefined.

| Variant | Mean foreground finish | Hosted placements | Local placements / hosted calls avoided |
|---|---:|---:|---:|
'''
    for name,a in result['scheduler'].items():text+=f"| {name} | {a['mean_finish']:.2f}s | {a['hosted']:,} | {a['local']:,} |\n"
    text+='\n| Variant vs context only | Better | Equal | Worse | Tradeoff |\n|---|---:|---:|---:|---:|\n'
    for name,a in result['scheduler'].items():
        if name in ['always_local','always_hosted','occupancy','context_only']:continue
        c=a['categories'];text+=f"| {name} | {c['better']:,} | {c['equal']:,} | {c['worse']:,} | {c['tradeoff']:,} |\n"
    text+='\n“Better” finishes no later with no more hosted placements and one strict improvement; “worse” reverses that relation. “Tradeoff” exchanges time for hosted use.\n\n'
    text+='| Variant | Pair-case mean finish | Occupied-lane mean finish |\n|---|---:|---:|\n'
    for name,a in result['scheduler'].items():text+=f"| {name} | {a['families']['pair_matrix']['mean_finish']:.2f}s | {a['families']['occupied_lanes']['mean_finish']:.2f}s |\n"
    text+='\n## Predictor overhead\n\nOne warmup per predictor is excluded. Each prompt is timed three times, alternating predictor order; per-prompt medians are summarized below. LTR is CPU-only; Clairvoyant uses one-thread CPU ONNX. These are observed timings on this Mac, not universal speed claims. Load includes local asset validation and model initialization. A live scheduler would need to account for this overhead, especially for sub-second requests.\n\n| Predictor | Load | Median prompt prediction | Maximum prompt median |\n|---|---:|---:|---:|\n'
    for name,r in result['runtime'].items():text+=f"| {name} | {r['load_seconds']:.3f}s | {r['median_ms']:.3f}ms | {r['max_ms']:.3f}ms |\n"
    baseline = result['scheduler']['clairvoyant_classes_combined']
    hybrid = result['scheduler']['ltr_frozen_nominations']
    text+='\n## Findings on this fixture\n\n'
    for backend in ['local','hosted']:
        before = result['metrics']['clairvoyant_classes'][backend]['tokens']
        after = result['metrics']['ltr'][backend]['tokens']
        text+=f"- **{backend.title()} output ordering:** pair accuracy {before['pairwise_accuracy']:.1%} → {after['pairwise_accuracy']:.1%}; Spearman {before['spearman']:.3f} → {after['spearman']:.3f}.\n"
    text+=f"- **Scheduler, frozen nominations:** LTR mean foreground finish {hybrid['mean_finish']:.2f}s versus {baseline['mean_finish']:.2f}s for existing Clairvoyant classes ({hybrid['mean_finish']-baseline['mean_finish']:+.3f}s); hosted placements {hybrid['hosted']:,} versus {baseline['hosted']:,} ({hybrid['hosted']-baseline['hosted']:+,}).\n"
    text+='\nDirect comparison of the LTR hybrid with the existing combined policy:\n\n| Better | Equal | Worse | Tradeoff |\n|---:|---:|---:|---:|\n'
    c=hybrid['vs_existing'];text+=f"| {c['better']:,} | {c['equal']:,} | {c['worse']:,} | {c['tradeoff']:,} |\n"
    text+='\nThe LTR checkpoint improves response-length ordering on this small sample but does not establish a scheduler improvement. Only the overlap branch consumes the new scores; dependency rules and frozen nominations still dominate decisions. Its pair-case mean finish is unchanged, with differences concentrated in occupied-lane cases. It also makes mistakes: architecture review is ranked longest, despite several saved reviews/investigations producing longer outputs. Keep LTR as an optional experimental signal rather than changing the default policy based on these 14 prompts. A future comparison could change queue ordering or nomination rules, but that would be a separate policy experiment, not an isolated predictor replacement.\n'
    text+='''
## Use and reproduce

```bash
# Reuse the existing optional model environment; no Gemma or GPU is needed.
python -m pip install -e '.[ltr,estimator]'
python scripts/download_ltr.py
python -m inference_router rank 'Return only one category for this support ticket.'
python scripts/evaluate_ltr.py --run --allow-legacy-no-hash \\
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \\
  --local-log runs/gemma-validation-20261110-local.jsonl \\
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \\
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl
```

The optional scheduler profile fields are `length_order_score` (higher means longer) and `length_order_source` (a shared, named predictor scale). The `rank` command emits both fields; merge them into an existing request profile before simulation. They do not replace NVIDIA task labels or produce a new acceleration preference.

Omit `--run` to replay cached predictor results without loading either predictor. `rank --input PATH --output PATH` supports JSONL request files. LTR is optional and does not replace the default Clairvoyant/NVIDIA profiler. Full responses, private logs, weights, and JSON prediction caches remain ignored. A fresh clone needs the original saved outputs/profiles to reproduce these exact historical results; it does not include those private artifacts. No model selection or thresholds were tuned on these 14 outcomes.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Run only the small frozen predictors on CPU; never run Gemma')
    parser.add_argument('--predictions',default='runs/ltr-predictions.json')
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--timings',default='examples/corrected-gemma-timings.csv')
    parser.add_argument('--local-log',action='append',required=True)
    parser.add_argument('--hosted-log',action='append',required=True)
    parser.add_argument('--allow-legacy-no-hash',action='store_true')
    parser.add_argument('--report',default='docs/ltr-comparison.md')
    args=parser.parse_args()
    if args.run:predict_all(args.predictions)
    evaluate(args)


if __name__=='__main__':main()
