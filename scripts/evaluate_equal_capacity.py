"""Compare two hosted slots with one hosted plus one local, using saved fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_combined_policy import JOBS, read, request_hash
from scripts.evaluate_profile_value import real_cases, run_variant
from scripts.replay_corrected_timings import load_fixture
from inference_router.request_profile import combine_predictions, ProfileRules
from inference_router.scheduler import simulate

VARIANTS = ('always_hosted', 'hosted_context', 'hosted_profile', 'occupancy',
            'context_only', 'length_signal_only', 'nomination_only', 'combined', 'always_local')
LABELS = {'always_hosted':'Always-hosted (FIFO)', 'hosted_context':'All-hosted + context priority',
          'hosted_profile':'All-hosted + context/profile priority', 'occupancy':'Occupancy-based',
          'context_only':'Context-only', 'length_signal_only':'Context + length',
          'nomination_only':'Context + acceleration flag', 'combined':'Full combined policy',
          'always_local':'Always-local (one-slot reference)'}


def cases_once(profiles, paired):
    """Capacity was a sweep dimension before; retain each workload exactly once."""
    for case in real_cases(profiles, paired):
        if case['slots'] == 1:
            yield case


def run_capacity_variant(spec, variant):
    if variant in {'hosted_context','hosted_profile'}:
        return simulate(spec['tasks'],spec['outcomes'],variant,hosted_slots=2)
    return run_variant(spec,variant,hosted_slots=2 if variant=='always_hosted' else 1)


def summarize(cases):
    result={}
    for variant in VARIANTS:
        metrics=[c['results'][variant]['metrics'] for c in cases]
        wins=ties=losses=0
        for case,m in zip(cases,metrics):
            delta=m['foreground_finish_seconds']-case['results']['always_hosted']['metrics']['foreground_finish_seconds']
            if abs(delta)<=1e-9:ties+=1
            elif delta<0:wins+=1
            else:losses+=1
        result[variant]={'mean_foreground':statistics.mean(m['foreground_finish_seconds'] for m in metrics),
                         'mean_all_tasks':statistics.mean(m['all_tasks_finish_seconds'] for m in metrics),
                         'hosted_calls':sum(m['hosted_calls'] for m in metrics),
                         'local_calls':sum(m['hosted_calls_avoided'] for m in metrics),
                         'faster_than_hosted_fifo':wins, 'equal_to_hosted_fifo':ties, 'slower_than_hosted_fifo':losses}
    return result


def render(result):
    text='''# Equal-capacity replay: two hosted versus one hosted plus one local

## Question and inputs

This replay holds total execution slots at two for the main comparison: **two hosted slots** for all-hosted strategies versus **one hosted and one local slot** for mixed strategies. Each slot executes at most one non-preemptive task. Always-local is retained separately as a one-slot reference, not an equal-capacity competitor.

The earlier sweep gave every strategy the same hosted limit, allowing mixed policies an additional local worker. Its always-hosted comparison therefore confounded scheduling with additional total execution capacity. Use the results here for the two-slot comparison. Comparisons among mixed policies already shared identical capacity.

No new Gemma inference, classifier inference, API calls, model training, or policy-threshold changes were performed. Outcomes use the [14-row owner-corrected timing fixture](corrected-timings.md); profiles are the original cached NVIDIA/Clairvoyant predictions, verified against prompt hashes and the routing-relevant fields derived from their saved rules. Original cached preferences remain frozen; diagnostic wording/version metadata is not regenerated. Local remains Q4_K_M and hosted remains BF16. These are heterogeneous deployment comparisons, not hardware-isolated measurements. The later interrupted BF16 run is excluded.

## Workloads and strategies

The old 4,056-case matrix repeated each arrival/dependency workload under hosted capacities one and two. Capacity is now assigned by strategy, so retaining both copies would double-count identical workloads. This replay uses **2,028 unique workloads: 1,092 pair cases and 936 occupied-lane cases**, with **6,006 task placements per strategy**. Arrival offsets remain 0/1/5 seconds, with independent or join topologies. Occupied-lane cases retain the exploratory code-review anchor and one background task.

| Strategy | Local slots used | Hosted slots used | Selection rule |
|---|---:|---:|---|
| Always-hosted (FIFO) | 0 | 2 | Earliest ready arrival, task-ID tie break |
| All-hosted + context priority | 0 | 2 | Foreground dependency depth before background, then arrival/ID |
| All-hosted + context/profile priority | 0 | 2 | Same structural priority, plus frozen acceleration nomination as a tie break |
| Occupancy-based | 1 | 1 | Local-first FIFO, spill to hosted |
| Context-only | 1 | 1 | Existing dependency/occupancy rules, neutral profiles |
| Context + length | 1 | 1 | Predicted length classes; acceleration nominations disabled |
| Context + acceleration flag | 1 | 1 | Combined nominations; overlap lengths replaced with medium |
| Full combined policy | 1 | 1 | Existing combined nominations and length classes |
| Always-local reference | 1 | 0 | Local FIFO; shown separately |

The two additional all-hosted controls distinguish placement effects from queue priority. Plain always-hosted is FIFO and does not prioritize foreground work; it is not an optimal all-hosted scheduler. The priority controls use the same visible dependency-depth calculation as the mixed policy. They are new evaluation baselines, not changes to the mixed policies. No strategy sees actual service durations or future arrival payloads before dispatch.

## Main results

Completion is measured from each scenario's start. “Foreground finish” is when all foreground tasks complete; “all tasks finish” also includes unrelated background work. Each scenario receives equal weight. Calls are simulated placements, not new paid requests. Hosted calls avoided relative to all-hosted equal local placements.

'''
    text+=table(result['overall'],include_reference=False)
    text+='\n### One-slot reference\n\nThis row has fewer execution slots and is not part of the equal-capacity comparison.\n\n'
    text+=table({'always_local':result['overall']['always_local']},include_reference=True)
    text+='\n## Results by scenario family\n'
    for name,stats in result['families'].items():
        text+=f"\n### {name} ({result['family_counts'][name]:,} workloads)\n\n"
        text+=table(stats,include_reference=False)
    text+='\n## Per-workload comparison with two-slot hosted FIFO\n\n| Strategy | Faster foreground finish | Same | Slower |\n|---|---:|---:|---:|\n'
    for name,s in result['overall'].items():
        if name in {'always_hosted','always_local'}:continue
        text+=f"| {LABELS[name]} | {s['faster_than_hosted_fifo']:,} | {s['equal_to_hosted_fifo']:,} | {s['slower_than_hosted_fifo']:,} |\n"
    text+='\nThese comparisons count foreground completion only; they do not combine time and hosted use into a single success score.\n'
    text+='\n## Interpretation\n\n'
    base=result['overall']['always_hosted'];full=result['overall']['combined'];priority=result['overall']['hosted_profile']
    text+=f"The combined policy has mean foreground finish **{full['mean_foreground']:.2f}s** versus **{base['mean_foreground']:.2f}s** for two-slot hosted FIFO and **{priority['mean_foreground']:.2f}s** for the all-hosted context/profile-priority control. It uses **{full['hosted_calls']:,} hosted calls** versus **{base['hosted_calls']:,}**, avoiding **{full['local_calls']:,} ({full['local_calls']/base['hosted_calls']:.1%})** of those calls.\n\n"
    text+='''With only one hosted slot in each mixed strategy, the length-overlap branch cannot affect dispatch: if a hosted task is running, there is no spare hosted slot to use. Accordingly, context + length exactly matches context-only, and the full combined policy exactly matches nomination-only in every one of the 2,028 workloads. This is a structural property of the implemented rule, not evidence that length prediction is useless in all schedulers.

With equal total slot counts, local execution is no longer extra free parallel capacity. Mixed routing must justify its completion tradeoff through reduced hosted use. A foreground win over hosted FIFO can reflect task ordering and background prioritization, rather than faster local service. The added hosted-priority controls help expose that distinction. Neither control is proven optimal for non-preemptive online arrivals, so this is not an optimal-scheduler bound.

All hosted fixture durations are smaller than their local counterparts. Under identical task-to-lane assignments and execution order, replacing a local lane with hosted service cannot increase completion time in this fixed-service model. Different online policies can nevertheless produce different orders and background interference.

These remain synthetic replays of 14 supplied timing pairs. Equal slot counts do not establish equal hardware cost, memory, utilization, provider capacity or performance under real concurrency. Durations are assumed fixed under concurrency; quality equivalence, classifier overhead, retries, thermal/provider variability, deadlines and billed cost are not evaluated. Hosted-call count is a usage proxy, not dollar savings. The matrices are not independent new measurements or a representative traffic distribution.

## Reproduce

```bash
python3 scripts/evaluate_equal_capacity.py \\
  --timings examples/corrected-gemma-timings.csv \\
  --profiles runs/combined-profiles.jsonl
```

This uses cached profiles without loading models or credentials. Public output is this lightweight report; detailed local results remain ignored. A fresh clone needs the private cached profiles or must generate them separately from the original prompts. The original logs and earlier reports are preserved.
'''
    return text


def table(stats,include_reference):
    text='| Strategy | Mean foreground finish | Mean all-tasks finish | Hosted calls | Local calls / hosted calls avoided |\n|---|---:|---:|---:|---:|\n'
    for name,s in stats.items():
        if name=='always_local' and not include_reference:continue
        text+=f"| {LABELS[name]} | {s['mean_foreground']:.2f}s | {s['mean_all_tasks']:.2f}s | {s['hosted_calls']:,} | {s['local_calls']:,} |\n"
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings',default='examples/corrected-gemma-timings.csv')
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--report',default='docs/equal-capacity.md')
    args=parser.parse_args()
    fixture=load_fixture(args.timings);rows=read(args.profiles);catalog={j['id']:j for j in JOBS}
    if len(rows)!=len(catalog) or {r['job_id'] for r in rows}!=set(catalog):raise ValueError('Require all 14 cached profiles without duplicates')
    for row in rows:
        if row['request_sha256']!=request_hash(catalog[row['job_id']]):raise ValueError('Profile prompt hash mismatch')
        expected=combine_predictions(row['nvidia'],row['clairvoyant'],ProfileRules(**row['profile']['rules']))
        # Keep the original policy snapshot. Newer local policy versions can
        # change diagnostic wording; every routing-relevant field must match.
        fields=('preference','length_class','length_probabilities','task_type','task_probability','reasoning','constraints','rules')
        if any(row['profile'].get(key)!=expected.get(key) for key in fields):
            raise ValueError('Cached routing fields differ from saved predictions/rules')
    profiles={r['job_id']:r['profile'] for r in rows}
    paired={r['job_id']:{'local':float(r['slow_seconds']),'hosted':float(r['fast_seconds'])} for r in fixture}
    cases=[]
    for case in cases_once(profiles,paired):
        cases.append({'id':case['id'].replace('__slots1__','__equal_capacity__'),'family':case['family'],
                      'results':{v:run_capacity_variant(case['spec'],v) for v in VARIANTS}})
    counts={name:sum(c['family']==name for c in cases) for name in ['pair_matrix','occupied_lanes']}
    result={'source_timing_sha256':hashlib.sha256(Path(args.timings).read_bytes()).hexdigest(),
            'source_profiles_sha256':hashlib.sha256(Path(args.profiles).read_bytes()).hexdigest(),
            'cases':len(cases),'family_counts':counts,'overall':summarize(cases),
            'families':{name:summarize([c for c in cases if c['family']==name]) for name in counts},
            'cases_detail':[{**{k:v for k,v in c.items() if k!='results'},'metrics':{v:r['metrics'] for v,r in c['results'].items()}} for c in cases]}
    if len(cases)!=2028 or counts!={'pair_matrix':1092,'occupied_lanes':936}:raise ValueError('Unexpected workload count')
    if any(s['hosted_calls']+s['local_calls']!=6006 for s in result['overall'].values()):raise ValueError('Placement accounting mismatch')
    output=Path('runs/equal-capacity-results.json');output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result))
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(render(result))
    print(table(result['overall'],include_reference=True))
    print('Saved',report,'; no inference or classifier calls.')


if __name__=='__main__':main()
