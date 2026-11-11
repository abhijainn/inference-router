"""Profile the existing prompts and replay three explicit offline workflows."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.make_requests import CALIBRATION_JOBS, VALIDATION_JOBS
from examples.scheduling_scenarios import scenarios
from inference_router.scheduler import STRATEGIES,simulate

JOBS=CALIBRATION_JOBS+VALIDATION_JOBS


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def request_hash(job):
    return hashlib.sha256(json.dumps(job['messages'],sort_keys=True).encode()).hexdigest()


def measurements(local_paths,hosted_paths,allow_legacy_no_hash=False):
    catalog={job['id']:job for job in JOBS}
    paired={key:{} for key in catalog}
    for name,paths,backend,model in [('local',local_paths,'local_ollama','gemma3:4b-it-q4_K_M'),('hosted',hosted_paths,'deepinfra','google/gemma-3-4b-it')]:
        seen=set()
        for path in paths:
            for row in read(path):
                key=row['job_id']
                if key not in catalog or key in seen or row.get('repeat') != 0:
                    raise ValueError('Expected one measurement per existing prompt/backend; do not combine duplicate logs')
                if row.get('backend')!=backend or row.get('model_identity',{}).get('model')!=model or row.get('success') is not True or row.get('done_reason')!='stop':
                    raise ValueError('Only naturally completed paired Gemma measurements are supported')
                saved_hash=row.get('request_features',{}).get('request_sha256')
                if saved_hash is None and not allow_legacy_no_hash:
                    raise ValueError('Legacy log has no request hash; explicitly use --allow-legacy-no-hash for known historical logs')
                if saved_hash is not None and saved_hash!=request_hash(catalog[key]):
                    raise ValueError('Measurement prompt does not match catalog')
                paired[key][name]=row['client_elapsed_seconds'];seen.add(key)
        if seen!=set(catalog):
            raise ValueError('Supply both calibration-combined and validation logs for each backend')
    return paired


def render(rows,results):
    counts={key:sum(row['profile']['preference']==key for row in rows) for key in ['prefer_local','consider_hosted','uncertain']}
    text='''# Combined request profiles and dynamic scheduling

> **Precision scope:** These historical results use local Q4_K_M versus hosted BF16, not a matched-precision hardware comparison. See [the precision audit](precision-audit.md) before citing them as hardware evidence.

## What is implemented

`profile` combines frozen NVIDIA task/complexity outputs with frozen Clairvoyant output-length probabilities. It does not convert them into seconds. The policy preserves each signal and returns a provisional `prefer_local`, `consider_hosted`, or `uncertain` result with reasons. GLiClass is not loaded. The previous fixed-budget policy remains a separate experiment.

`simulate` is an offline event-driven scheduler, not a live Ollama/DeepInfra dispatcher. At every arrival or completion it recomputes readiness, visible dependency depth, and backend occupancy, then dispatches more work if appropriate. Tasks are non-preemptive. It has one local worker and two hosted slots in this experiment. Future arrivals are hidden from decisions. Declared dependency nodes become visible only when submitted; submitted blocked tasks can influence priorities. No additional paid inference calls were made.

## Profile rules, fixed before inspecting combined results

Default thresholds: NVIDIA selected-task probability at least 0.40, Clairvoyant maximum class probability at least 0.55, and reasoning score at least 0.30 for a higher-reasoning flag. These are explicit experimental settings, not trained thresholds or calibrated confidence levels. NVIDIA Other/Unknown, truncated inputs, low scores, or classification/extraction with a non-short length prediction produce `uncertain`.

For remaining cases: short output prefers local; medium/long output with the higher-reasoning flag becomes `consider_hosted`; long output with lower reasoning prefers local as a background candidate. Medium output without a stronger signal remains uncertain. Constraints are retained as a feature but are not multiplied into an arbitrary composite suitability score. No rule guarantees quality or a deadline, and hosted remains the same Gemma family rather than a more capable model.

## All 14 combined profiles

'''
    text+=f"Counts: {counts['prefer_local']} prefer local, {counts['consider_hosted']} consider hosted, {counts['uncertain']} uncertain.\n\n"
    text+='| Prompt | NVIDIA task | Short / medium / long | Reasoning | Preference | Issues |\n|---|---|---|---:|---|---|\n'
    for row in rows:
        p=row['profile'];probs=p['length_probabilities']
        text+=f"| {row['job_id']} | {p['task_type']} | {' / '.join(f'{probs[k]:.2f}' for k in ['short','medium','long'])} | {p['reasoning']:.3f} | {p['preference']} | {', '.join(p['issues']) or p['reason']} |\n"
    text+='''
The high uncertain count is an observed limitation, not a failed software check. The simple classification prompts still receive medium-length forecasts; the combined policy now exposes that conflict rather than claiming accurate durations. These profiles have not been validated as true fast/slow labels. The raw distributions remain available even when a provisional preference is uncertain.

## Dynamic scheduling rules

1. Re-evaluate after each submitted arrival or completion. Run tasks only after all dependencies finish; respect capacity and never preempt work already running.
2. Favor foreground results and their ancestors over unrelated background work. Use the longest visible dependency chain, counted in tasks, as a structural priority. This is not a duration-based critical-path calculation.
3. Consider hosted for a ready task that unlocks a foreground chain, unless an equal-or-deeper chain is already running on hosted. Among equal structural priorities, a `consider_hosted` profile breaks ties. Such a profile can also nominate independent foreground work for hosted execution.
4. Use available local capacity for other ready work. When local is occupied, send waiting foreground work to hosted unless hosted is already running equal-or-deeper foreground work with an equal-or-larger predicted output class. In that case, retain the task for local and reconsider at the next event.
5. Background work uses local capacity unless it is an ancestor of a submitted foreground result. An uncertain profile is neutral, not an automatic hosted escalation; its raw length class is still a weak input to the overlap heuristic.

Rule 4 is an overlap heuristic, not a proof that local work will finish before hosted work. It uses only the ordering short < medium < long, with no assumed token lengths, speeds, seconds, or monetary costs. It does not estimate remaining service time or account for how long a running task has already executed. It can wait too long if predictions are wrong, and hosted work can finish first; the next completion event then revises the decision. Finite replay tasks eventually finish, but ongoing-stream fairness, deadlines, and starvation protection are not established.

The scheduler must receive foreground/background intent and dependencies from a caller/workflow. The classifiers do not invent those facts. A background task that later becomes necessary is not migrated or restarted. This prototype uses queued ready tasks and completion events; it does not implement retries, cancellation, live provider limits, or persistent state.

## Replay design and baselines

The three hand-authored scenarios reuse the previously measured standalone Gemma durations, with no new generation. Profiles were generated from prompt text only. Durations live in a separate outcome mapping consulted by the simulator only after a dispatch decision; the decision function receives no service times or future event timestamps. A decision's trace records which tasks were visible, its backend, and its reason.

- **Background and arrivals:** code review starts as background at time zero; classification arrives at 1s, extraction at 2s. The objective is prompt foreground completion while the background task continues.
- **Parallel join:** code review, classification, and extraction are submitted together; a final classification-like task waits for all three. This represents an AND-join.
- **Dependency chain with background:** code review, code generation, and final classification form a chain, alongside an independent background runbook.

These are synthetic workflows using measured service-time fixtures, not real dependencies or data flow between the original prompts. For example, the join does not consume newly generated review text. Duplicate task use across scenarios is not new independent evidence. The nine earliest calibration prompts predate request hashes in both logs. For these known historical logs we explicitly enable `--allow-legacy-no-hash`, associating outcomes by job ID and model rather than verified message hash; this limits provenance assurance. The other five pairs and all new profiles have matching hashes. Mismatched nonempty hashes are always rejected. The snapshots assume local and hosted executions do not interfere and service time does not change under hosted concurrency. Quality equivalence is assumed, as before.

Baselines share tasks, outcomes, capacity, and dependency enforcement. `always_local` and `always_hosted` use one backend. `occupancy` uses FIFO by arrival/id, local first then hosted, without task/dependency priority. `context_only` uses the same dynamic rules with every profile replaced by neutral preference and medium length. It isolates the additional effect of classifier features. `profile_dynamic` uses the combined profiles. Ties are deterministic; conclusions may change with tie order or other workloads.

Profiles are precomputed in replay. Profile inference overhead is excluded from the timeline and included neither as service nor queue time. The separate classifier report measures that overhead; a live system would need to include it. The following results describe these exact fixtures, not validated production savings.

## Replay results

| Scenario | Strategy | Foreground finished by | All tasks finished by | Mean foreground flow | Hosted calls | Hosted calls avoided | Queue wait total |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    for name,strategies in results.items():
        for strategy,result in strategies.items():
            m=result['metrics']
            text+=f"| {name} | {strategy} | {m['foreground_finish_seconds']:.2f}s | {m['all_tasks_finish_seconds']:.2f}s | {m['mean_foreground_flow_seconds']:.2f}s | {m['hosted_calls']} | {m['hosted_calls_avoided']} | {m['total_queue_wait_seconds']:.2f}s |\n"
    text+='''
Foreground finish is the timestamp of the last non-background task, relative to time zero; mean foreground flow is finish minus arrival, averaged across those tasks. All-task finish includes background work. Queue wait begins when a task is both submitted and dependency-ready. Hosted calls avoided counts local executions relative to always-hosted, including background tasks. All strategies finish every task—there are no abstention credits or deadline guarantees. No deadline/budget-miss metric is reported because these scenarios specify no deadlines.

## Dispatch traces for the profile policy

'''
    for name,strategies in results.items():
        text+=f'### {name}\n\n| Time | Task | Backend | Reason |\n|---:|---|---|---|\n'
        for event in strategies['profile_dynamic']['trace']:
            if event['event']=='dispatched':
                text+=f"| {event['time']:.2f}s | {event['task']} | {event['backend']} | {event['reason']} |\n"
        text+='\n'
    equal=all(results[name]['profile_dynamic']['metrics']==results[name]['context_only']['metrics'] for name in results)
    text+='\n## Interpretation\n\n'
    text+=('The profile-informed and context-only scheduler produce identical metrics in all three scenarios. The observed benefits relative to simple baselines come from workflow context, not demonstrated incremental classifier value. ' if equal else 'Profile-informed and context-only results differ in at least one scenario; compare the table and traces rather than attributing all improvements to classifiers. ')
    text+='''The simulator demonstrates the intended behaviors: foreground arrivals can bypass occupied local background work, and other tasks can remain local while hosted execution is already advancing a foreground path. It also exposes that the current profiles are mostly uncertain. A functioning simulator does not establish an optimal or generally useful scheduling algorithm.

Next evaluation should use fresh workflows, varied arrival order, capacity and service-time variability, and profile-inference overhead. Keep context-only as an ablation. Before live dispatch, add real completion notifications, explicit service errors and cancellation semantics, quality eligibility, and a persistent task queue. No live scheduling or automatic spending was enabled here.

## Verification

All 68 tests pass with the optional NVIDIA, GLiClass, and Clairvoyant runtimes installed. Checks include invalid probabilities, rule boundaries, conflicting/truncated profiles, no dependence on duration fields, dependency readiness, capacity, future-arrival isolation, overlap behavior, context-only ablation, outcome-independent initial dispatch, and offline CLI execution. The 14 profiles were also generated using the real frozen NVIDIA and Clairvoyant weights. Passing these tests verifies implementation behavior; it does not validate scheduling optimality or profile accuracy.

## Commands

```bash
# In the optional classifier environment, add Clairvoyant support:
python -m pip install -r requirements-profiles.txt
python scripts/download_estimator.py
python scripts/download_classifiers.py
HF_HUB_OFFLINE=1 python -m inference_router profile "Your prompt"

# Replay the measured fixtures locally (raw logs remain ignored):
HF_HUB_OFFLINE=1 python scripts/evaluate_combined_policy.py --run --allow-legacy-no-hash \\
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \\
  --local-log runs/gemma-validation-20261110-local.jsonl \\
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \\
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl

# The evaluator writes ignored workflow specs with separate task/outcome fields:
python -m inference_router simulate examples/generated/parallel_join.json --strategy all
```

`profile --input PATH --output PATH` profiles a JSONL job file. `simulate --hosted-slots N --output PATH` controls capacity and saves detailed traces. The simulator requires no model dependencies or API key. The evaluator without `--run` uses its saved profiles but still verifies their request hashes against the catalog. Thresholds are configurable via the Python `ProfileRules` API and recorded in every profile. GLiClass remains an explicit optional `classify --classifiers gliclass` choice; `classify` now defaults to NVIDIA only.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--allow-legacy-no-hash',action='store_true')
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--local-log',action='append',required=True)
    parser.add_argument('--hosted-log',action='append',required=True)
    parser.add_argument('--report',default='docs/combined-policy.md')
    args=parser.parse_args()
    if args.run:
        from inference_router.request_profile import RequestProfiler
        profiler=RequestProfiler()
        profiler.predict([{'role':'user','content':'Reply with a greeting.'}])
        path=Path(args.profiles);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as output:
            for job in JOBS:
                row={'job_id':job['id'],'request_sha256':request_hash(job),**profiler.predict(job['messages'])}
                output.write(json.dumps(row)+'\n');output.flush()
    rows=read(args.profiles)
    catalog={job['id']:job for job in JOBS}
    if len(rows)!=len(catalog) or {row['job_id'] for row in rows}!=set(catalog):
        raise ValueError('Expected exactly one profile per existing prompt')
    for row in rows:
        if row.get('request_sha256')!=request_hash(catalog[row['job_id']]):
            raise ValueError('Profile prompt hash mismatch')
    profiles={row['job_id']:row['profile'] for row in rows}
    specs=scenarios(profiles,measurements(args.local_log,args.hosted_log,args.allow_legacy_no_hash))
    results={name:{strategy:simulate(spec['tasks'],spec['outcomes'],strategy) for strategy in STRATEGIES} for name,spec in specs.items()}
    generated=Path('examples/generated');generated.mkdir(parents=True,exist_ok=True)
    for name,spec in specs.items():
        (generated/f'{name}.json').write_text(json.dumps(spec,indent=2))
    Path('runs').mkdir(exist_ok=True)
    Path('runs/scheduling-results.json').write_text(json.dumps(results,indent=2))
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(render(rows,results))
    for name,strategies in results.items():
        for strategy,result in strategies.items():print(name,strategy,result['metrics'])


if __name__=='__main__':
    main()
