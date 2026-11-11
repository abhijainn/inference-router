"""Summarize completed BF16 pairs and replay frozen scheduling policies separately."""
import argparse
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inference_router.precision import aggregate_pairs,verify_bf16_pair,normalized_precision,ensure_timing_valid
from inference_router.request_profile import combine_predictions,ProfileRules
from scripts.evaluate_combined_policy import JOBS,read,request_hash
from scripts.evaluate_profile_value import VARIANTS,real_cases,run_variant,summary,routing_difference


def render(manifest,table,cases,stats):
    evidence=manifest['precision_check'];local=manifest['identities']['local'];hosted=manifest['identities']['hosted']
    text='''# Matched reported-BF16 Gemma rerun

## Why this rerun exists

The earlier measurements used Ollama Gemma 3 4B Q4_K_M and DeepInfra Gemma 3 4B BF16. Those runs do **not** have matching precision and must not be described as if they did. Their tables and synthetic replays remain historical comparisons of two differently configured deployments. This experiment creates a separate BF16-versus-BF16 baseline and reruns the frozen scheduler against new observations.

The precision preflight checks actual Ollama metadata/digest and the current DeepInfra catalog, refusing Q4, FP16, missing dtype, or a different model label. BF16 and IEEE FP16 are distinct formats; equal bit width is insufficient.

## Precision evidence and scope

'''
    text+=f"| Field | Local | Hosted |\n|---|---|---|\n| Model | `{local['model']}` | `{hosted['model']}` |\n| Reported weight format | `{local['details']['quantization_level']}` | `{hosted['details']['quantization_level']}` |\n| Evidence | Ollama model metadata | DeepInfra public catalog |\n| Artifact digest | `{local['digest']}` | Not exposed/verified |\n"
    text+='''
This verifies a **match in reported weight precision**, not identical compute kernels, accumulation precision, checkpoint revisions, tokenizer/chat templates, serving stacks, or hardware-only causality. The hosted catalog is a provider statement, not a per-request runtime attestation. The local runtime also reports loaded-model residency, captured privately after warmup and each local request. The two deployments can still produce different outputs and output lengths. Answer quality equivalence has not been established.

## Protocol

'''
    text+=f"Started: `{manifest['started_at']}`. Completed: `{manifest['completed_at']}`.\n\n"
    text+=f"All 14 original prompts were run **{manifest['repeats']} times per backend**, giving {len(table)*manifest['repeats']*2} measured requests. Local model loading was warmed separately and excluded. Prompts are shuffled with a fixed seed per repetition; local-first and hosted-first pair order alternate. Only one measured request is active at a time. Temperature is 0, seed 42, streaming disabled, and local context is 4,096. Identical messages and hashes are sent to both backends. Output generation runs to natural stop; hosted provider limits remain. Final aggregation requires all successful naturally stopped pairs. No automatic retries, model substitutions, or overwriting of historical runs occur. Explicit DNS-failure recovery preserves interrupted logs and a separate failed-attempt audit before resuming missing requests.\n\n"
    text+='''These are client-observed end-to-end durations. The table uses per-prompt medians and retains min/max ranges; a small repetition count is insufficient for reliable tail-latency statistics. Prompt caching, thermal state, network/provider load, and concurrent user applications are not controlled. Memory pressure and swap can affect local results even when Ollama reports full GPU residency. A matched weight format removes one confound, not all deployment differences.

## Paired timing results

| Prompt | Local median [min–max] | Hosted median [min–max] | Local output tokens (median) | Hosted output tokens (median) |
|---|---:|---:|---:|---:|
'''
    for item in table:
        l,h=item['local'],item['hosted']
        text+=f"| {item['job_id']} | {l['median_seconds']:.2f}s [{l['min_seconds']:.2f}–{l['max_seconds']:.2f}] | {h['median_seconds']:.2f}s [{h['min_seconds']:.2f}–{h['max_seconds']:.2f}] | {l['median_output_tokens']} | {h['median_output_tokens']} |\n"
    text+='\n## Frozen-policy scheduler replay\n\n'
    text+='''NVIDIA/Clairvoyant predictions and combined-profile rules are unchanged. New per-prompt median service times replace the old Q4/BF16 single-observation fixtures. The same 4,056 synthetic cases and seven strategy variants are evaluated. Observed service times remain private to the engine and are consulted only after dispatch. This is not a live concurrent benchmark: independent measured medians are reused as fixed service times, classifier overhead is excluded, and the same 14 prompts are repeatedly reused. No training or threshold tuning is performed.

| Strategy | Mean foreground finish | Hosted executions | Local executions | Better / equal / worse / tradeoff versus context-only |
|---|---:|---:|---:|---|
'''
    for variant in VARIANTS:
        values=stats[variant];counts=values['categories']
        h=sum(case['results'][variant]['metrics']['hosted_calls'] for case in cases)
        l=sum(case['results'][variant]['metrics']['hosted_calls_avoided'] for case in cases)
        text+=f"| {variant} | {values['mean_finish']:.2f}s | {h:,} | {l:,} | {' / '.join(str(counts[k]) for k in ['better','equal','worse','tradeoff'])} |\n"
    text+='''
Better means no later foreground finish and no more hosted calls, with at least one improvement; worse reverses that relation. Tradeoffs are faster with more hosted use or slower with less hosted use. Counts are simulated placements, not additional API requests or dollar savings. Identical call counts between old and new replays are possible because the policy uses profiles and context, not actual future service times; completion-event ordering can nevertheless change placements.

'''
    old_path=Path('runs/profile-value-results.json')
    if old_path.exists():
        previous=json.loads(old_path.read_text())['summary']
        text+='### Earlier mixed-precision replay versus this rerun\n\n| Strategy | Earlier mean foreground finish | New BF16/BF16 mean foreground finish |\n|---|---:|---:|\n'
        for variant in ['context_only','nomination_only','combined']:
            text+=f"| {variant} | {previous[variant]['mean_finish']:.2f}s | {stats[variant]['mean_finish']:.2f}s |\n"
        text+='\nThis before/after comparison also changes date, runtime conditions, sample count, aggregation, and generated outputs. It cannot isolate the causal effect of quantization. Earlier blog percentages must not be reused as if they were findings from this BF16 rerun.\n'
    text+='\n## Hosted usage and environment\n\n'
    totals=manifest.get('hosted_usage_summary',{})
    if totals:
        text+=f"Measured hosted calls: {totals['requests']}; input tokens: {totals['input_tokens']}; output tokens: {totals['output_tokens']}. Estimated cost at the catalog's standard list rates: ${totals['estimated_list_cost_usd']:.6f}. This is not a billing invoice and does not model cached-token discounts.\n\n"
    if manifest.get('failed_attempt_summary'):
        failures=manifest['failed_attempt_summary']
        text+=f"There were {len(failures)} failed attempts recorded separately from the completed pairs. This run was explicitly resumed after public connectivity and precision were rechecked. Successful measurements were retained rather than rerun; interruptions and resume warmups remain in the private manifest. The service-time fixtures model successful completions, not the outage/recovery gap or retry latency.\n\n"
    if manifest.get('environment_notes'):
        notes=manifest['environment_notes']
        text+=f"A diagnostic observation during warmup/first complex request reported swap usage `{notes['swapusage_observation']}` and system free-memory percentage {notes['system_memory_free_percent_observation']}. Swap allocation alone is not a measurement of active paging. Other applications were not stopped, and this condition was not experimentally isolated.\n\n"
    text+='''
## Interpretation limits

The original mismatch mattered to experimental design. It was not legitimate to label the old runs same-precision. The new logs support saying both deployments **reported BF16 weight precision** during preflight and repetition checks. They do not support saying every difference is due only to hardware or that hosted internally used an independently verified artifact/compute dtype. Old results should stay labeled Q4_K_M versus BF16; this report is a separate matched-reported-format experiment.

For stricter hardware isolation, the same exact checkpoint, tokenizer/template, runtime configuration, and numerical computation would need verification on both sides, ideally with control of the serving environment. A public API does not expose all those details. That limitation should be explicit rather than filled in by assumption.

## Reproduce

```bash
ollama pull gemma3:4b-it-bf16
python scripts/run_matched_precision.py --check-only
python scripts/run_matched_precision.py \\
  --output-dir runs/gemma-bf16-paired-NEW-DATE --repeat 3
python scripts/evaluate_matched_precision.py \\
  --measurements runs/gemma-bf16-paired-NEW-DATE
```

A measured run makes billable hosted requests using the locally configured key; `--check-only` reads metadata without generation. If interrupted specifically by DNS lookup failure, first verify connectivity, then use the same output directory and `--resume` to preserve completed work and the failed-attempt audit. Other failure types are not automatically retried or silently excluded. The evaluator is offline and refuses incomplete pairs, missing/different prompt hashes, truncated requests, or precision mismatch. Model downloads, private logs, provider IDs, responses, and credentials remain ignored. Only lightweight source/tests and aggregate documentation should be published.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements',required=True)
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--report',default='docs/matched-precision.md')
    args=parser.parse_args();root=Path(args.measurements)
    manifest=json.loads((root/'manifest.json').read_text())
    ensure_timing_valid(manifest)
    if not manifest.get('completed_at'):raise ValueError('Paired run is incomplete')
    verify_bf16_pair(manifest['identities']['local'],manifest['identities']['hosted'])
    local_rows,hosted_rows=read(root/'local.jsonl'),read(root/'hosted.jsonl')
    if (root/'failed-attempts.jsonl').exists():
        manifest['failed_attempt_summary']=[{key:row.get(key) for key in ['job_id','repeat','backend','error','client_elapsed_seconds']} for row in read(root/'failed-attempts.jsonl')]
    for row in local_rows+hosted_rows:
        job=next(job for job in JOBS if job['id']==row['job_id'])
        if row.get('request_features',{}).get('request_sha256')!=request_hash(job):raise ValueError('Prompt differs from original catalog')
    for row in local_rows:
        running=(row.get('local_runtime_after_request') or {}).get('models',[])
        loaded=next((model for model in running if model.get('name')=='gemma3:4b-it-bf16'),None)
        if not loaded or normalized_precision(loaded)!='BF16' or loaded.get('digest')!=row['model_identity']['digest']:
            raise ValueError('Local runtime did not confirm the expected BF16 artifact after a measured request')
    table=aggregate_pairs(local_rows,hosted_rows,[job['id'] for job in JOBS],manifest['repeats'])
    prices=manifest['identities']['hosted']['pricing_usd_per_million']
    if all(isinstance(row.get('input_tokens'),int) and isinstance(row.get('output_tokens'),int) for row in hosted_rows):
        inputs=sum(row['input_tokens'] for row in hosted_rows);outputs=sum(row['output_tokens'] for row in hosted_rows)
        manifest['hosted_usage_summary']={'requests':len(hosted_rows),'input_tokens':inputs,'output_tokens':outputs,
                                          'estimated_list_cost_usd':(inputs*prices['input']+outputs*prices['output'])/1e6}
    if (root/'environment-notes.json').exists():manifest['environment_notes']=json.loads((root/'environment-notes.json').read_text())
    rows=read(args.profiles);catalog={job['id']:job for job in JOBS}
    if len(rows)!=len(catalog) or {row['job_id'] for row in rows}!=set(catalog):raise ValueError('Expected original 14 profiles')
    for row in rows:
        if row.get('request_sha256')!=request_hash(catalog[row['job_id']]):raise ValueError('Profile hash mismatch')
        if row['profile']!=combine_predictions(row['nvidia'],row['clairvoyant'],ProfileRules(**row['profile']['rules'])):raise ValueError('Profiles/rules were modified')
    profiles={row['job_id']:row['profile'] for row in rows}
    paired={item['job_id']:{backend:item[backend]['median_seconds'] for backend in ['local','hosted']} for item in table}
    cases=[]
    for case in real_cases(profiles,paired):
        results={variant:run_variant(case['spec'],variant,case['slots']) for variant in VARIANTS}
        cases.append({**case,'results':results})
    stats=summary(cases)
    compact=[{key:value for key,value in case.items() if key not in {'spec','results'}} | {'results':{variant:{'metrics':value['metrics'],'tasks':value['tasks']} for variant,value in case['results'].items()}} for case in cases]
    (root/'scheduler-results.json').write_text(json.dumps({'summary':stats,'cases':compact,'timing_table':table}))
    report_path=Path(args.report);report_path.parent.mkdir(exist_ok=True,parents=True)
    report_path.write_text(render(manifest,table,cases,stats))
    print('Saved matched-precision report:',report_path)
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
