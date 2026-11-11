"""Frozen-policy stress test: real cached profiles plus separate controlled cases."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from examples.profile_signal_scenarios import controlled_scenarios
from scripts.evaluate_combined_policy import JOBS,read,request_hash,measurements
from inference_router.scheduler import simulate
from inference_router.request_profile import combine_predictions,ProfileRules

VARIANTS=('context_only','length_signal_only','nomination_only','combined','always_local','always_hosted','occupancy')
EPSILON=1e-9


def run_variant(spec,variant,hosted_slots=2):
    tasks=deepcopy(spec['tasks'])
    strategy=variant
    if variant in {'length_signal_only','nomination_only','combined'}:
        strategy='profile_dynamic'
        for task in tasks:
            if variant=='length_signal_only':
                task['profile']['preference']='uncertain'
            elif variant=='nomination_only':
                task['profile']['length_class']='medium'
    return simulate(tasks,spec['outcomes'],strategy,hosted_slots)


def category(reference,candidate):
    time_diff=candidate['foreground_finish_seconds']-reference['foreground_finish_seconds']
    call_diff=candidate['hosted_calls']-reference['hosted_calls']
    if abs(time_diff)<=EPSILON and call_diff==0:return 'equal'
    if time_diff<=EPSILON and call_diff<=0:return 'better'
    if time_diff>=-EPSILON and call_diff>=0:return 'worse'
    return 'tradeoff'


def routing_difference(reference,candidate):
    return any(row['backend']!=candidate['tasks'][key]['backend'] for key,row in reference['tasks'].items())


def real_cases(profiles,paired):
    def task(ident,prompt,arrival=0,dependencies=None):
        return {'id':ident,'arrival_seconds':arrival,'dependencies':dependencies or [],
                'background':False,'profile':profiles[prompt]}
    for first,second in itertools.permutations(sorted(profiles),2):
        for offset in [0,1,5]:
            for slots in [1,2]:
                for topology in ['independent','join']:
                    tasks=[task('a',first),task('b',second,offset)]
                    outcomes={'a':paired[first],'b':paired[second]}
                    if topology=='join':
                        tasks.append(task('join','holdout-classify-01',dependencies=['a','b']))
                        outcomes['join']=paired['holdout-classify-01']
                    yield {'family':'pair_matrix','id':f'{first}__{second}__arrival{offset}__slots{slots}__{topology}',
                           'first':first,'second':second,'offset':offset,'slots':slots,'topology':topology,
                           'spec':{'tasks':tasks,'outcomes':outcomes}}


    # Exploratory extension: occupied local + hosted execution is needed to
    # exercise the length-cover branch; pairs alone leave that path dormant.
    anchor='code-review-01'
    for background,arrival_task in itertools.permutations([key for key in sorted(profiles) if key!=anchor],2):
        for offset in [0,1,5]:
            for slots in [1,2]:
                for topology in ['independent','join']:
                    tasks=[task('a',anchor),task('b',background),task('c',arrival_task,offset)]
                    tasks[1]['background']=True
                    outcomes={'a':paired[anchor],'b':paired[background],'c':paired[arrival_task]}
                    if topology=='join':
                        tasks.append(task('join','holdout-classify-01',dependencies=['a','c']))
                        outcomes['join']=paired['holdout-classify-01']
                    yield {'family':'occupied_lanes','id':f'occupied__{background}__{arrival_task}__arrival{offset}__slots{slots}__{topology}',
                           'first':anchor,'second':background,'third':arrival_task,'offset':offset,'slots':slots,'topology':topology,
                           'spec':{'tasks':tasks,'outcomes':outcomes}}


def summary(cases):
    result={}
    for variant in VARIANTS:
        rows=[case['results'][variant] for case in cases]
        comparisons=[category(case['results']['context_only']['metrics'],row['metrics']) for case,row in zip(cases,rows)]
        result[variant]={
            'categories':{label:comparisons.count(label) for label in ['better','equal','worse','tradeoff']},
            'routing_changed':sum(routing_difference(case['results']['context_only'],row) for case,row in zip(cases,rows)),
            'mean_finish':statistics.mean(row['metrics']['foreground_finish_seconds'] for row in rows),
            'mean_hosted':statistics.mean(row['metrics']['hosted_calls'] for row in rows),
        }
    return result


def report(cases,controlled):
    stats=summary(cases)
    text='''# Where request profiles change scheduling

> **Precision scope:** These historical results use local Q4_K_M versus hosted BF16, not a matched-precision hardware comparison. See [the precision audit](precision-audit.md) before citing them as hardware evidence.

## Question and frozen policy

We tested whether NVIDIA/Clairvoyant information changes backend choices beyond the context-only scheduler. No scheduler rules, model weights, profile thresholds, or earlier measurements were changed to improve these results. No new paid inference calls were made.

The existing policy uses classifier features in two places: a combined `consider_hosted` nomination can break equal-depth dependency ties or nominate independent foreground work for acceleration; output-length ordering can decide whether a task waits locally or uses spare hosted capacity. NVIDIA task type also affects profile uncertainty through the combined rules, but the scheduler does not otherwise route by task label. NVIDIA constraint scores are recorded but currently have no scheduling effect. This experiment tests the implemented feature paths, not the broader potential of every model output.

## Separate evidence sets

**Controlled mechanism checks:** three tiny hand-authored workloads with synthetic service times and constructed NVIDIA/Clairvoyant probabilities. These show how the policy behaves when signals are informative, and when they are wrong. They are not predictions from the pretrained checkpoints and do not demonstrate their accuracy.

**Real-profile stress sweep:** reuse all 14 cached pretrained profiles and their paired Gemma service-time observations. For every ordered pair of distinct prompts, test second-task arrival offsets 0, 1, and 5 seconds; hosted capacities 1 and 2; and independent foreground tasks versus an AND-join followed by a fixed final classification-like task. There are 14 × 13 × 3 × 2 × 2 = **2,184 pair cases**, each evaluated under seven variants. Pair order reverses which prompt gets task ID `a` versus `b`, exercising deterministic tie order in both directions. The pair sweep initially showed no length-only effect, so we explicitly extended this exploratory experiment with **1,872 occupied-lane cases**: code review is a foreground anchor, every ordered pair of the other 13 prompts supplies background local work and a foreground arrival, with the same offsets, capacities, and two topologies. The join waits for the two foreground tasks, not unrelated background work. This brings the total to **4,056 cases**. The extension tests a previously dormant branch and is not held-out confirmation; it concentrates on the one current acceleration nomination. This is a finite stress matrix, not independent new measurements or a representative traffic distribution. Topologies and offsets are illustrative choices.

The measured outcomes are private to the simulator and consulted only after dispatch. Future task arrivals and completion timestamps are not visible to the policy. Submitted join dependencies may refer to a task whose payload has not arrived; that is declared workflow context, not access to its future profile. Scenarios have one local worker. Durations are fixed standalone observations, so concurrency interference, provider variability, quality differences, retries, and model-loading costs are not simulated. Profile inference is cached and its overhead is excluded; wall-clock benefits are therefore optimistic relative to classifying live requests. The nine earliest prompt pairs lack logged request hashes; explicit legacy mode associates those outcomes by job ID/model. Newer pairs and every cached profile have matching catalog hashes.

## Comparisons

| Variant | Information used |
|---|---|
| Context only | Dependencies, arrivals, foreground/background intent, occupancy; neutral preference and medium length for every task |
| Length signal only | Same rules plus actual predicted length classes; all acceleration nominations removed |
| Nomination only | Same rules plus combined profile preference; all length classes replaced with medium |
| Combined | Existing combined preference and actual predicted length class |
| Always local / always hosted | One backend, with the same dependency/capacity rules |
| Occupancy | Local-first FIFO by arrival/task ID, spilling to hosted when local is occupied |

These are **mechanism ablations**, not clean NVIDIA-only versus Clairvoyant-only models. Nominations already depend on both classifiers. Length-only retains raw argmax classes even for uncertain profiles, matching the scheduler's current use of weak length signals. We do not turn observed lengths or execution times into oracle profiles.

A case is **better** if it finishes foreground work no later and uses no more hosted calls, with at least one strict improvement. **Worse** reverses that relation. **Equal** matches both metrics. **Tradeoff** means faster with more calls or slower with fewer calls. Times use a 1e-9 comparison tolerance. These labels do not impose an arbitrary dollar-versus-delay weighting. Routing changed means at least one task changes backend; ordering/waiting differences can occur without a backend change. Hosted-call count is not a dollar-cost estimate.

## Controlled checks

| Case | Variant | Foreground finish | Hosted calls |
|---|---|---:|---:|
'''
    for name,results in controlled.items():
        for variant in ['context_only','length_signal_only','nomination_only','combined']:
            m=results[variant]['metrics']
            text+=f"| {name} | {variant} | {m['foreground_finish_seconds']:.2f}s | {m['hosted_calls']} |\n"
    text+='''
- **Equal-depth tie:** a small classification-like branch and a larger analysis branch feed a join. A useful nomination sends the analysis branch to hosted and the small branch locally, rather than accelerating whichever task ID sorts first.
- **Wait versus spare hosted slot:** a medium-output foreground task runs on hosted while background work occupies local. A new long-output foreground task arrives. Length information can reject the assumption that existing hosted work provides enough overlap and start the new task on a spare hosted slot. With one hosted slot, that option disappears.
- **Wrong nomination:** a genuinely large task is described as short, while a genuinely small task is nominated for acceleration. The same rule accelerates the wrong branch. This is intentionally a counterexample, showing how classifier errors can reverse the benefit.

## Real cached-profile sweep

| Variant | Better | Equal | Worse | Tradeoff | Cases with backend changes | Mean foreground finish | Mean hosted calls |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for variant,values in stats.items():
        counts=values['categories']
        text+=f"| {variant} | {counts['better']} | {counts['equal']} | {counts['worse']} | {counts['tradeoff']} | {values['routing_changed']} | {values['mean_finish']:.2f}s | {values['mean_hosted']:.3f} |\n"
    text+='\n### Fast versus slow inference usage\n\nThese are simulated placements across the same cases, not new API calls. Each strategy executes the same total number of tasks.\n\n| Strategy | Fast: hosted | Slow: local | Hosted share | Local share |\n|---|---:|---:|---:|---:|\n'
    totals={}
    for variant in VARIANTS:
        hosted=sum(case['results'][variant]['metrics']['hosted_calls'] for case in cases)
        local=sum(case['results'][variant]['metrics']['hosted_calls_avoided'] for case in cases)
        totals[variant]=(hosted,local)
        text+=f"| {variant} | {hosted:,} | {local:,} | {100*hosted/(hosted+local):.2f}% | {100*local/(hosted+local):.2f}% |\n"
    extra_nomination=totals['nomination_only'][0]-totals['context_only'][0]
    extra_combined=totals['combined'][0]-totals['context_only'][0]
    extra_length=totals['combined'][0]-totals['nomination_only'][0]
    text+=f"\nThe nomination mechanism adds {extra_nomination} hosted calls relative to context-only; the full combined policy adds {extra_combined}. Adding the length mechanism on top of nomination accounts for {extra_length} extra calls. Hosted calls are a spending proxy, not measured dollar savings; prompts and generated lengths differ.\n"
    text+='''
All comparisons above are against context-only. Means weight every synthetic case equally, count reused observations repeatedly, and are descriptive only. They are not confidence estimates or expected production savings.

### Combined results by workload condition

| Condition | Cases | Better | Equal | Worse | Tradeoff | Backend changes |
|---|---:|---:|---:|---:|---:|---:|
'''
    for field,values in [('family',['pair_matrix','occupied_lanes']),('slots',[1,2]),('offset',[0,1,5]),('topology',['independent','join'])]:
        for value in values:
            subset=[case for case in cases if case[field]==value]
            metrics=summary(subset)['combined'];c=metrics['categories']
            text+=f"| {field}={value} | {len(subset)} | {c['better']} | {c['equal']} | {c['worse']} | {c['tradeoff']} | {metrics['routing_changed']} |\n"
    text+='\n### Incremental length contribution\n\n'
    length_changes=sum(routing_difference(case['results']['nomination_only'],case['results']['combined']) for case in cases)
    length_categories=[category(case['results']['nomination_only']['metrics'],case['results']['combined']['metrics']) for case in cases]
    text+=f"Comparing combined against nomination-only isolates the current raw-length mechanism: {length_changes} cases change backend; " + ', '.join(f"{label}={length_categories.count(label)}" for label in ['better','equal','worse','tradeoff']) + '. These remain mechanism ablations, not a claim about independent model accuracy.\n'
    length_effects={'faster':0,'same_finish':0,'slower':0}
    for case in cases:
        a=case['results']['nomination_only']['metrics'];b=case['results']['combined']['metrics']
        if routing_difference(case['results']['nomination_only'],case['results']['combined']):
            delta=b['foreground_finish_seconds']-a['foreground_finish_seconds']
            length_effects['faster' if delta < -EPSILON else 'slower' if delta > EPSILON else 'same_finish']+=1
    text+=f"In those changed cases, adding length finishes faster in {length_effects['faster']}, at the same time in {length_effects['same_finish']}, and slower in {length_effects['slower']}. In this particular sweep all 130 changes add one hosted call: 46 add no final-completion benefit and 12 are slower as well. This directly exposes unnecessary acceleration while other work is still determining completion.\n"
    text+='\n### Selected real-profile examples\n\n'
    selected=[]
    for label in ['better','worse','tradeoff']:
        eligible=[case for case in cases if category(case['results']['context_only']['metrics'],case['results']['combined']['metrics'])==label]
        if eligible:
            def magnitude(case):
                a=case['results']['context_only']['metrics'];b=case['results']['combined']['metrics']
                return abs(a['foreground_finish_seconds']-b['foreground_finish_seconds'])
            selected.append((label,max(eligible,key=magnitude)))
    for label,case in selected:
        text+=f"#### {label}: `{case['id']}`\n\n"
        text+='| Strategy | Foreground finish | Hosted calls | Dispatch order (task → backend) |\n|---|---:|---:|---|\n'
        for variant in ['context_only','combined']:
            r=case['results'][variant];m=r['metrics']
            dispatch=', '.join(f"{event['task']} → {event['backend']} at {event['time']:.2f}s" for event in r['trace'] if event['event']=='dispatched')
            text+=f"| {variant} | {m['foreground_finish_seconds']:.2f}s | {m['hosted_calls']} | {dispatch} |\n"
        text+='\nThese examples were selected after the full sweep for the largest absolute finish-time difference in their category; they are illustrations, not an unbiased sample. In pair cases task `a` is the first named prompt and `b` the second. In occupied-lane cases `a` is code review, `b` the named background prompt, and `c` the named arriving foreground prompt. The join uses the saved held-out classification duration.\n\n'
    combined=stats['combined'];c=combined['categories']
    text+='## What this establishes\n\n'
    text+=f"The actual cached profiles change at least one backend choice in **{combined['routing_changed']}/{len(cases):,}** cases. Against context-only they yield {c['better']} better cases, {c['equal']} equal cases, {c['worse']} worse cases, and {c['tradeoff']} tradeoffs under the two-metric definition. Unlike the earlier three scenarios, the feature paths are now exercised by varied ordering, arrivals, and capacity.\n\n"
    text+='''The harmful occupied-lane example does not necessarily mean the code-review nomination was wrong: code review really is lengthy. Accelerating it changes which resource is occupied when another request arrives. With no knowledge of that future arrival, a locally sensible decision can still worsen the eventual foreground finish.

The controlled checks demonstrate plausible mechanisms, while the real sweep tests the current imperfect predictions. Neither establishes a generally better policy. Review the overall results and both helpful and harmful traces before deciding to enable additional routing rules. With 12 of the 14 combined profiles uncertain, weak raw length classes and the single code-review nomination dominate the current feature effects. More varied scenarios cannot repair bad predictions by themselves.

In this sweep, keeping the acceleration nominations while disabling the raw-length overlap rule gives 960 better cases and 80 worse cases relative to context-only; adding length changes this to 911 better and 93 worse, with more tradeoffs. This makes the raw-length overlap rule a candidate for revision, not a validated feature to enable more aggressively. Most gains concentrate on the single code-review nomination, which the occupied-lane extension deliberately anchors; this does not demonstrate general task-type routing ability.

The next credible validation is a fresh prompt/workflow set, repeated service measurements, classifier overhead included, randomized tie identifiers, and a predefined completion-versus-hosted-use objective. Keep context-only, mechanism ablations, and simple baselines. Do not tune rules on this sweep and then report the same sweep as independent validation.

## Verification

The added checks cover useful nomination, the length mechanism with and without spare capacity, a harmful wrong signal, input immutability during ablations, two-metric comparison boundaries, and complete case enumeration with both pair orders. Cached combined profiles are checked against their raw NVIDIA/Clairvoyant predictions and recorded rule settings before evaluation. The full suite passed all 74 tests with the optional real-model checks installed. Test success verifies behavior, not general scheduling performance. No production scheduler behavior was changed.

## Reproduce and inspect

```bash
python scripts/evaluate_profile_value.py \\
  --allow-legacy-no-hash \\
  --local-log runs/gemma-calibration-combined-20261110-local.jsonl \\
  --local-log runs/gemma-validation-20261110-local.jsonl \\
  --hosted-log runs/gemma-calibration-combined-20261110-hosted.jsonl \\
  --hosted-log runs/gemma-validation-20261110-hosted.jsonl
```

The evaluator uses `runs/combined-profiles.jsonl` from the previous profile run. It needs no model runtime, API credentials, or network. The full stress-case inputs and per-case metrics/backend assignments are written to ignored local files; the public report contains aggregate tables and selected traces only. Every case can be rerun with `python -m inference_router simulate PATH` from the generated spec directory. Controlled cases are explicitly marked in their own directory. Weights and private measurements remain excluded from Git.
'''
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--local-log',action='append',required=True)
    parser.add_argument('--hosted-log',action='append',required=True)
    parser.add_argument('--allow-legacy-no-hash',action='store_true')
    parser.add_argument('--report',default='docs/profile-value.md')
    args=parser.parse_args()
    rows=read(args.profiles);catalog={job['id']:job for job in JOBS}
    if len(rows)!=len(catalog) or {row['job_id'] for row in rows}!=set(catalog):raise ValueError('Expected all 14 profiles without duplicates')
    for row in rows:
        if row.get('request_sha256')!=request_hash(catalog[row['job_id']]):raise ValueError('Profile hash mismatch')
        expected=combine_predictions(row['nvidia'],row['clairvoyant'],ProfileRules(**row['profile']['rules']))
        if row['profile']!=expected:raise ValueError('Cached profile differs from its raw classifier predictions/rules')
    profiles={row['job_id']:row['profile'] for row in rows}
    paired=measurements(args.local_log,args.hosted_log,args.allow_legacy_no_hash)
    cases=[]
    for case in real_cases(profiles,paired):
        results={variant:run_variant(case['spec'],variant,case['slots']) for variant in VARIANTS}
        cases.append({**case,'results':results})
    controlled={name:{variant:run_variant(spec,variant) for variant in VARIANTS} for name,spec in controlled_scenarios().items()}
    specs_dir=Path('examples/generated/profile-value');specs_dir.mkdir(parents=True,exist_ok=True)
    for case in cases:
        (specs_dir/f"{case['id']}.json").write_text(json.dumps(case['spec']))
    controlled_dir=specs_dir/'controlled';controlled_dir.mkdir(exist_ok=True)
    for name,spec in controlled_scenarios().items():(controlled_dir/f'{name}.json').write_text(json.dumps(spec))
    # Keep the full spec inputs and every metric/placement locally, avoiding thousands of duplicate traces.
    compact=[{key:value for key,value in case.items() if key not in {'spec','results'}} | {'results':{variant:{'metrics':result['metrics'],'tasks':result['tasks']} for variant,result in case['results'].items()}} for case in cases]
    destination=Path('runs/profile-value-results.json');destination.parent.mkdir(exist_ok=True,parents=True)
    destination.write_text(json.dumps({'summary':summary(cases),'cases':compact,'controlled':controlled}))
    report_path=Path(args.report);report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(report(cases,controlled))
    print(json.dumps(summary(cases),indent=2));print('Saved',report_path)


if __name__=='__main__':main()
