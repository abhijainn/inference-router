# Equal-capacity replay: two hosted versus one hosted plus one local

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

| Strategy | Mean foreground finish | Mean all-tasks finish | Hosted calls | Local calls / hosted calls avoided |
|---|---:|---:|---:|---:|
| Always-hosted (FIFO) | 30.67s | 30.88s | 6,006 | 0 |
| All-hosted + context priority | 30.32s | 30.88s | 6,006 | 0 |
| All-hosted + context/profile priority | 30.32s | 30.88s | 6,006 | 0 |
| Occupancy-based | 40.45s | 40.45s | 2,782 | 3,224 |
| Context-only | 36.94s | 41.95s | 2,169 | 3,837 |
| Context + length | 36.94s | 41.95s | 2,169 | 3,837 |
| Context + acceleration flag | 34.09s | 35.76s | 2,224 | 3,782 |
| Full combined policy | 34.09s | 35.76s | 2,224 | 3,782 |

### One-slot reference

This row has fewer execution slots and is not part of the equal-capacity comparison.

| Strategy | Mean foreground finish | Mean all-tasks finish | Hosted calls | Local calls / hosted calls avoided |
|---|---:|---:|---:|---:|
| Always-local (one-slot reference) | 63.09s | 63.09s | 0 | 6,006 |

## Results by scenario family

### pair_matrix (1,092 workloads)

| Strategy | Mean foreground finish | Mean all-tasks finish | Hosted calls | Local calls / hosted calls avoided |
|---|---:|---:|---:|---:|
| Always-hosted (FIFO) | 23.50s | 23.50s | 2,730 | 0 |
| All-hosted + context priority | 23.50s | 23.50s | 2,730 | 0 |
| All-hosted + context/profile priority | 23.50s | 23.50s | 2,730 | 0 |
| Occupancy-based | 29.83s | 29.83s | 910 | 1,820 |
| Context-only | 28.69s | 28.69s | 1,105 | 1,625 |
| Context + length | 28.69s | 28.69s | 1,105 | 1,625 |
| Context + acceleration flag | 27.95s | 27.95s | 1,112 | 1,618 |
| Full combined policy | 27.95s | 27.95s | 1,112 | 1,618 |

### occupied_lanes (936 workloads)

| Strategy | Mean foreground finish | Mean all-tasks finish | Hosted calls | Local calls / hosted calls avoided |
|---|---:|---:|---:|---:|
| Always-hosted (FIFO) | 39.04s | 39.49s | 3,276 | 0 |
| All-hosted + context priority | 38.27s | 39.48s | 3,276 | 0 |
| All-hosted + context/profile priority | 38.27s | 39.48s | 3,276 | 0 |
| Occupancy-based | 52.84s | 52.84s | 1,872 | 1,404 |
| Context-only | 46.56s | 57.42s | 1,064 | 2,212 |
| Context + length | 46.56s | 57.42s | 1,064 | 2,212 |
| Context + acceleration flag | 41.26s | 44.87s | 1,112 | 2,164 |
| Full combined policy | 41.26s | 44.87s | 1,112 | 2,164 |

## Per-workload comparison with two-slot hosted FIFO

| Strategy | Faster foreground finish | Same | Slower |
|---|---:|---:|---:|
| All-hosted + context priority | 92 | 1,936 | 0 |
| All-hosted + context/profile priority | 92 | 1,936 | 0 |
| Occupancy-based | 0 | 188 | 1,840 |
| Context-only | 68 | 222 | 1,738 |
| Context + length | 68 | 222 | 1,738 |
| Context + acceleration flag | 64 | 604 | 1,360 |
| Full combined policy | 64 | 604 | 1,360 |

These comparisons count foreground completion only; they do not combine time and hosted use into a single success score.

## Interpretation

The combined policy has mean foreground finish **34.09s** versus **30.67s** for two-slot hosted FIFO and **30.32s** for the all-hosted context/profile-priority control. It uses **2,224 hosted calls** versus **6,006**, avoiding **3,782 (63.0%)** of those calls.

With only one hosted slot in each mixed strategy, the length-overlap branch cannot affect dispatch: if a hosted task is running, there is no spare hosted slot to use. Accordingly, context + length exactly matches context-only, and the full combined policy exactly matches nomination-only in every one of the 2,028 workloads. This is a structural property of the implemented rule, not evidence that length prediction is useless in all schedulers.

With equal total slot counts, local execution is no longer extra free parallel capacity. Mixed routing must justify its completion tradeoff through reduced hosted use. A foreground win over hosted FIFO can reflect task ordering and background prioritization, rather than faster local service. The added hosted-priority controls help expose that distinction. Neither control is proven optimal for non-preemptive online arrivals, so this is not an optimal-scheduler bound.

All hosted fixture durations are smaller than their local counterparts. Under identical task-to-lane assignments and execution order, replacing a local lane with hosted service cannot increase completion time in this fixed-service model. Different online policies can nevertheless produce different orders and background interference.

These remain synthetic replays of 14 supplied timing pairs. Equal slot counts do not establish equal hardware cost, memory, utilization, provider capacity or performance under real concurrency. Durations are assumed fixed under concurrency; quality equivalence, classifier overhead, retries, thermal/provider variability, deadlines and billed cost are not evaluated. Hosted-call count is a usage proxy, not dollar savings. The matrices are not independent new measurements or a representative traffic distribution.

## Reproduce

```bash
python3 scripts/evaluate_equal_capacity.py \
  --timings examples/corrected-gemma-timings.csv \
  --profiles runs/combined-profiles.jsonl
```

This uses cached profiles without loading models or credentials. Public output is this lightweight report; detailed local results remain ignored. A fresh clone needs the private cached profiles or must generate them separately from the original prompts. The original logs and earlier reports are preserved.
