"""Replay the owner's corrected timing fixture without inference or raw-log edits."""
import argparse
import csv
from decimal import Decimal
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_combined_policy import JOBS, read, request_hash
from scripts.evaluate_profile_value import VARIANTS, real_cases, run_variant, summary
from inference_router.request_profile import combine_predictions, ProfileRules


def load_fixture(path):
    with Path(path).open(newline='') as source:
        rows=list(csv.DictReader(source))
    expected={job['id'] for job in JOBS}
    if len(rows)!=len(expected) or {row['job_id'] for row in rows}!=expected:
        raise ValueError('Require exactly the 14 known prompts without duplicates')
    for row in rows:
        for key in ('slow_seconds','fast_seconds'):
            value=float(row[key])
            if not math.isfinite(value) or value<=0:
                raise ValueError('Durations must be positive finite seconds')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timings',default='examples/corrected-gemma-timings.csv')
    parser.add_argument('--profiles',default='runs/combined-profiles.jsonl')
    parser.add_argument('--report',default='docs/corrected-timings.md')
    args=parser.parse_args()
    rows=load_fixture(args.timings)
    saved=read(args.profiles);catalog={job['id']:job for job in JOBS}
    if len(saved)!=len(catalog) or {row['job_id'] for row in saved}!=set(catalog):
        raise ValueError('Require one cached profile per original prompt')
    for row in saved:
        if row['request_sha256']!=request_hash(catalog[row['job_id']]):
            raise ValueError('Cached profile prompt hash mismatch')
        if row['profile']!=combine_predictions(row['nvidia'],row['clairvoyant'],ProfileRules(**row['profile']['rules'])):
            raise ValueError('Cached profile differs from frozen rules')
    profiles={row['job_id']:row['profile'] for row in saved}
    paired={row['job_id']:{'local':float(row['slow_seconds']),'hosted':float(row['fast_seconds'])} for row in rows}
    cases=[]
    for case in real_cases(profiles,paired):
        cases.append({**case,'results':{variant:run_variant(case['spec'],variant,case['slots']) for variant in VARIANTS}})
    stats=summary(cases)
    slow=sum(Decimal(row['slow_seconds']) for row in rows)
    fast=sum(Decimal(row['fast_seconds']) for row in rows)
    wins=sum(float(row['fast_seconds'])<float(row['slow_seconds']) for row in rows)
    text='''# Corrected Gemma timings and offline scheduler replay

## Source and scope

The project owner supplied this corrected 14-row table on November 11, 2026. It is the current timing fixture for this replay. Four local values changed from the original logs: extraction 1.03 → 2.03 seconds; classification 0.27 → 1.27; constraint planning 32.21 → 52.21; evidence synthesis 32.48 → 52.48. All other supplied values are unchanged. These are owner-supplied corrections, not independently verified new measurements. Original raw logs, tokens, phase timings, and saved predictions were preserved.

“Slow inference” means local Ollama `gemma3:4b-it-q4_K_M`; “fast inference” means DeepInfra `google/gemma-3-4b-it`, reported BF16. These are heterogeneous deployment choices, not a matched-precision hardware benchmark. The interrupted local BF16 run is excluded. The original study ran each prompt once per backend, with natural stopping and no automated answer-quality grading. The supplied corrections contain no new outputs or repetitions.

## All 14 requests

| Request | Slow Inference | Fast Inference |
|---|---:|---:|
'''
    for row in rows:text+=f"| {row['request']} | {row['slow_seconds']}s | {row['fast_seconds']}s |\n"
    text+=f'\nSum of displayed values: **{slow:.2f}s slow/local** and **{fast:.2f}s fast/hosted**. Hosted is faster in **{wins}/14 pairs**; its summed duration is **{(1-fast/slow)*100:.1f}% lower**. Sums are not workflow makespans.\n'
    text+='''
## Regenerated scheduler comparison

The unchanged scheduler rules and the original cached NVIDIA/Clairvoyant profiles are replayed against the corrected CSV. Profiles are verified against prompt hashes and frozen combination rules. This recomputes the same 2,184 pair cases and 1,872 occupied-lane cases (4,056 total), with one local worker, one or two hosted slots, arrival offsets 0/1/5 seconds, and independent/join topologies. Case construction and strategy definitions are documented in [the historical mechanism report](profile-value.md); its old numeric tables are superseded for this corrected fixture.

These cases reuse 14 pairs; they are not 4,056 independent measurements. Decisions receive task profiles, dependencies, arrivals and occupancy, not actual durations. Actual durations are consulted only after dispatch. Classifier runtime overhead, concurrency interference, network variability, retries, energy cost, and answer-quality differences are excluded. No inference API calls were made for this replay. “Calls” below are simulated placements, not newly billed requests. There are no workflow deadlines in these scenarios, so budget misses are not defined here.

| Strategy | Mean foreground finish | Hosted placements | Local placements | Hosted calls avoided vs always-hosted |
|---|---:|---:|---:|---:|
'''
    total=sum(len(case['spec']['tasks']) for case in cases)
    for variant in VARIANTS:
        hosted=sum(case['results'][variant]['metrics']['hosted_calls'] for case in cases)
        text+=f"| {variant} | {stats[variant]['mean_finish']:.2f}s | {hosted:,} | {total-hosted:,} | {total-hosted:,} |\n"
    text+='\n| Strategy vs context only | Better | Equal | Worse | Tradeoff |\n|---|---:|---:|---:|---:|\n'
    for variant in ['length_signal_only','nomination_only','combined']:
        c=stats[variant]['categories']
        text+=f"| {variant} | {c['better']:,} | {c['equal']:,} | {c['worse']:,} | {c['tradeoff']:,} |\n"
    text+='''
“Better” means no later foreground completion and no more hosted placements, with at least one strict improvement; “worse” reverses this comparison. “Tradeoff” changes completion time and hosted use in opposite directions. Hosted calls avoided are a usage proxy, not measured dollar savings. Results describe this finite fixture and do not establish production benefit or equal answer quality.

## Reproduce

```bash
python3 scripts/replay_corrected_timings.py \\
  --timings examples/corrected-gemma-timings.csv \\
  --profiles runs/combined-profiles.jsonl
```

The CSV is public and small. Cached real classifier profiles remain private/ignored as in the original experiment, so a fresh clone must generate those profiles before reproducing the replay. This command loads no model and requires no API key when the cached profiles already exist. Historical duration-estimator and budget-policy reports remain explicitly labeled with their original inputs; they were not recalibrated using end-to-end corrections without replacement phase data.
'''
    Path(args.report).write_text(text)
    print(f'Saved {args.report}; {len(cases)} cases; slow={slow}, fast={fast}; no inference calls.')


if __name__=='__main__':main()
