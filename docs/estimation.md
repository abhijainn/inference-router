# Estimation and the eventual routing policy

## Length prediction is separate from backend suitability

Clairvoyant Model B reads the prompt using 19 lexical features and returns
probabilities for short, medium and long output. It does not read hardware state,
dependencies, deadlines, queue pressure or costs. This project keeps its weights
frozen. It is a lightweight baseline, not a demonstrated best predictor for this
workload.

Features: input character count divided by four; code keywords; requested-length
phrases; whether the prompt ends in a question; format keywords; clause count;
and 13 one-hot instruction-verb categories. For conversations, the adapter joins
message contents with blank lines. This lexical representation can miss task
intent; classification was a particularly poor case in the pilot.

Length units are `len(response) // 4`, not actual model-token counts. Short is
below 200 units, medium is [200, 800), and long is 800 or more. The long category
has no finite upper bound.

## Empirical conversion to seconds

For each backend b and actual output-length category k, calculate:

- `r[b,k]`: mean observed length among that category's completed responses.
- `P[k]`: the new prompt's category probability from frozen Clairvoyant.
- `L[b] = sum(P[k] * r[b,k])`: expected length approximation.
- `T[b] = A[b] + L[b] / R[b]`: duration approximation.

| Quantity | Local profile | Hosted profile |
|---|---|---|
| Category means | Measured local response lengths | Measured hosted response lengths |
| Rate R | Sum of lengths / sum of server generation times | Sum of lengths / sum of client elapsed times |
| Overhead A | Mean(client elapsed - generation time) | Zero; already included in end-to-end rate |
| Backend identity | Local URL, model digest/version, precision and settings | API URL, model name, precision, settings and standard tier |

No fixed 100/500/1200 category representatives remain. If a category has no
observations, the estimator explicitly falls back to measured overall mean
duration and reports the missing categories. With no observations, duration is
null. Neither category probabilities nor this mean conversion provide calibrated
confidence or an upper-duration guarantee.

Profiles use only successful, naturally stopped requests without a client output
cap. Local samples must follow warmup. Hosted samples must use the standard tier.
Responses with separately reported reasoning are excluded because that reasoning
text is not saved. New logs save a response-length scalar before optionally
removing answer text. Failures/truncations remain in logs but are not learned as
full completion times. API success does not establish answer correctness.

Every prediction is made before execution. Signed error is actual minus predicted
seconds; positive means underestimation. Optional updating happens only after
scoring, with a 100-observation window and duplicate-run protection. Exact output
token counts and observed reasoning are never used as pre-execution inputs.
Profiles require matching configuration; schema v1 profiles need rebuilding from
their original response logs. Hidden hosted changes without a digest remain an
unobservable source of drift.

## What the pilot taught us

The [validation table](experiments.md#duration-prediction-validation) shows that
measured category means improve the basis of the conversion but do not fix wrong
length predictions. A one-word classification was estimated at 14-16 seconds,
versus about 0.25-0.34 seconds in reality. Adding more measurements of backend
speed alone cannot fix that mismatch.

Possible next experiments, not implemented here:

1. Prefer measured history for known workflow steps, using task identity supplied
   by the workflow rather than inventing urgency from prompt wording.
2. Compare similar-request retrieval using task, input size and output constraints.
3. Compare each method with a task-average baseline on fresh prompts and repeated
   runs. Repeating an exact prompt checks repeatability, not generalization.
4. Measure underestimation and empirical uncertainty before deadline decisions.

## Proposed policy objective

For this POC, hosted is treated as the fast/paid option and local as the
slower/cheaper option. The measured table contains exceptions; the assumption is
a policy convention rather than a guarantee that hosted wins every request.
Local is cheaper in incremental API spend, not necessarily free in electricity
or hardware opportunity cost. No local energy measurement was made.

A task fits local execution when the API cost saved is useful and its extra
execution time can be absorbed without unacceptable delay to the workflow.
Task length alone does not determine this.

A proposed margin is:

    margin[b] = latest_acceptable_finish -
                (current_time + additional_queue_wait[b] + estimated_duration[b] + safety_allowance[b])

A comfortably positive margin supports local execution; a near-zero margin is
uncertain; a negative margin suggests accelerating the task if hosted can meet
the budget. Queue time already included in hosted duration must not be added
again. This margin is not a probability.

Initially a caller can supply a task budget. A later dependency scheduler can
compute budgets using the workflow deadline, remaining downstream work and
backend contention, and revise them as tasks complete. The score should be judged
by workflow completion time and hosted cost saved, not just classifier accuracy.
A standalone budget policy and paired replay are now implemented; see
[policy and evaluation](policy.md). It prefers eligible local inference when
its adjusted estimate fits and otherwise selects fitting hosted inference.
There is no dependency deadline engine, persistent queue, or automatic dispatch yet.
