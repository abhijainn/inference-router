"""Offline paired replay: decision inputs are separate from observed outcomes."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .policy import BackendEstimate, choose_backend

DEFAULT_BUDGETS = (5.0, 20.0, 60.0)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _index(rows, backend):
    result = {}
    for row in rows:
        if row.get('backend') != backend:
            raise ValueError(f'Unexpected backend in paired log; expected {backend}')
        if row.get('success') is not True or row.get('done_reason') != 'stop':
            raise ValueError('Replay requires successful, naturally stopped paired measurements; failed/truncated rows cannot be silently excluded.')
        duration = row.get('client_elapsed_seconds')
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('Paired outcome duration must be finite and positive')
        key = (row.get('job_id'), row.get('repeat', 0))
        if not isinstance(key[0], str) or not isinstance(key[1], int) or isinstance(key[1], bool):
            raise ValueError('Each paired row needs job_id and integer repeat')
        if key in result:
            raise ValueError('Duplicate job/repeat in replay input; choose a single paired run or distinct repeat indices.')
        if 'estimated_seconds' not in row.get('duration_prediction', {}):
            raise ValueError('Missing pre-execution prediction. Use recorded validation logs or explicit leave-one-out mode, not an in-sample profile.')
        result[key] = row
    if not result:
        raise ValueError('No paired measurements supplied')
    return result


def replay(local_rows, hosted_rows, budgets=DEFAULT_BUDGETS, safety_factor=1.25):
    local, hosted = _index(local_rows, 'local_ollama'), _index(hosted_rows, 'deepinfra')
    if local.keys() != hosted.keys():
        raise ValueError('Local and hosted logs must contain exactly matching job/repeat pairs')
    for key in local:
        left = local[key].get('request_features', {}).get('request_sha256')
        right = hosted[key].get('request_features', {}).get('request_sha256')
        if left and right and left != right:
            raise ValueError('Same job ID has different request hashes across backends')
    # Validate settings even when no route can be selected.
    for budget in budgets:
        choose_backend(BackendEstimate(None), BackendEstimate(None), budget, safety_factor)
    if not budgets or len(set(budgets)) != len(budgets):
        raise ValueError('Supply at least one distinct budget')
    scenarios = []
    for budget in budgets:
        metrics = {name: {'requests': len(local), 'local_calls': 0, 'hosted_calls': 0,
                         'hosted_calls_avoided': 0, 'late_completions': 0,
                         'not_routed': 0, 'budget_misses': 0, 'unsafe_local_choices': 0,
                         'unnecessary_hosted_calls': 0}
                   for name in ('policy', 'always_local', 'always_hosted')}
        decisions = []
        for key in sorted(local):
            lrow, hrow = local[key], hosted[key]
            # Only forecasts, budget, eligibility and allowance enter the policy.
            decision = choose_backend(
                BackendEstimate(lrow['duration_prediction']['estimated_seconds']),
                BackendEstimate(hrow['duration_prediction']['estimated_seconds']),
                budget, safety_factor)
            # Actual durations are consulted only AFTER the decision exists.
            actual = {'local': lrow['client_elapsed_seconds'], 'hosted': hrow['client_elapsed_seconds']}
            for strategy, selected in [('policy', decision['backend']), ('always_local', 'local'), ('always_hosted', 'hosted')]:
                score = metrics[strategy]
                if selected is None:
                    score['not_routed'] += 1
                    score['budget_misses'] += 1
                    continue
                score[selected + '_calls'] += 1
                if selected == 'local':
                    score['hosted_calls_avoided'] += 1
                    score['unsafe_local_choices'] += int(actual['local'] > budget and actual['hosted'] <= budget)
                else:
                    score['unnecessary_hosted_calls'] += int(actual['local'] <= budget)
                late = actual[selected] > budget
                score['late_completions'] += int(late)
                score['budget_misses'] += int(late)
            decisions.append({'job_id': key[0], 'repeat': key[1], 'decision': decision,
                              'actual_seconds': actual,
                              'prediction_origin': lrow.get('prediction_origin', 'recorded_before_execution')})
        for score in metrics.values():
            score['budget_miss_fraction'] = score['budget_misses'] / score['requests']
            score['hosted_calls_avoided_fraction'] = score['hosted_calls_avoided'] / score['requests']
        scenarios.append({'budget_seconds': budget, 'metrics': metrics, 'decisions': decisions})
    return {'request_pairs': len(local), 'safety_factor': safety_factor,
            'queue_assumption': 'No additional queue outside measured request duration; not a scheduler simulation.',
            'metric_definitions': {'hosted_calls_avoided': 'Requests routed locally, relative to always hosted. Abstentions earn no avoided-call credit.',
                                   'budget_misses': 'Late selected completions plus requests not routed.',
                                   'not_routed': 'Policy uncertain or neither fits; no real request was executed in replay.'},
            'scenarios': scenarios}


def leave_one_out_predictions(local_rows, hosted_rows, calibration_local, calibration_hosted, jobs, estimator=None):
    """Retrospective replay only: remove every same-prompt calibration sample.

    Existing validation predictions are preserved. Missing calibration predictions
    are regenerated from OTHER request outcomes; this is not chronological testing.
    """
    from .estimator import DurationEstimator, TimingProfile
    catalog = {}
    for job in jobs:
        if job['id'] in catalog and job['messages'] != catalog[job['id']]['messages']:
            raise ValueError('Conflicting prompt catalog entries')
        catalog[job['id']] = job
    def prompt_hash(job):
        return hashlib.sha256(json.dumps(job['messages'], sort_keys=True).encode()).hexdigest()
    hashes = {ident: prompt_hash(job) for ident, job in catalog.items()}
    for row in calibration_local + calibration_hosted:
        if row['job_id'] not in hashes:
            raise ValueError('Prompt catalog is incomplete for calibration')
    estimator = estimator or DurationEstimator()
    outputs = []
    for rows, calibration in [(local_rows, calibration_local), (hosted_rows, calibration_hosted)]:
        output = []
        for original in rows:
            row = deepcopy(original)
            if 'estimated_seconds' in row.get('duration_prediction', {}):
                row['prediction_origin'] = 'recorded_before_execution'
                output.append(row)
                continue
            ident = row['job_id']
            if ident not in hashes:
                raise ValueError('No prompt definition for retrospective prediction')
            recorded_hash = row.get('request_features', {}).get('request_sha256')
            if recorded_hash and recorded_hash != hashes[ident]:
                raise ValueError('Prompt catalog does not match recorded request')
            others = [r for r in calibration if hashes[r['job_id']] != hashes[ident]]
            if not others:
                raise ValueError('No independent calibration observations remain')
            estimator.profile = TimingProfile.from_rows(others)
            estimator.profile.check_configuration(row['model_identity'])
            row['duration_prediction'] = estimator.predict(catalog[ident]['messages'], -1)
            row['prediction_origin'] = 'retrospective_leave_one_prompt_out'
            row['calibration_samples_excluded'] = len(calibration) - len(others)
            output.append(row)
        outputs.append(output)
    return tuple(outputs)


def markdown_report(report):
    lines = ['| Budget | Strategy | Hosted calls avoided | Hosted calls | Budget misses | Late completions | Not routed |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for scenario in report['scenarios']:
        for name, score in scenario['metrics'].items():
            lines.append(f"| {scenario['budget_seconds']:g} sec | {name} | {score['hosted_calls_avoided']}/{score['requests']} | {score['hosted_calls']} | {score['budget_misses']}/{score['requests']} | {score['late_completions']} | {score['not_routed']} |")
    return '\n'.join(lines) + '\n'
