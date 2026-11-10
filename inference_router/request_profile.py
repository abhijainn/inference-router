"""Interpretable NVIDIA + Clairvoyant profiles, without duration conversion."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ProfileRules:
    min_task_probability: float = .4
    min_length_probability: float = .55
    high_reasoning: float = .3

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite probability')


def distribution(values, labels=None):
    if not isinstance(values, dict) or not values or (labels is not None and set(values) != set(labels)):
        raise ValueError('Missing or invalid probability distribution')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()) or abs(sum(values.values()) - 1) > .001:
        raise ValueError('Invalid probability values or sum')
    return values


def combine_predictions(nvidia, clairvoyant, rules=None):
    rules = rules or ProfileRules()
    rules.validate()
    probabilities = distribution(clairvoyant.get('length_class_probabilities'), ['short', 'medium', 'long'])
    tasks = distribution(nvidia.get('task_probabilities'))
    task = nvidia.get('task_type')
    if task not in tasks or tasks[task] != max(tasks.values()):
        raise ValueError('NVIDIA selected task does not match distribution')
    reasoning = nvidia.get('reasoning')
    constraints = nvidia.get('constraint_ct')
    for value in [reasoning, constraints]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid NVIDIA reasoning/constraint score')
    length = max(probabilities, key=probabilities.get)
    issues = []
    if nvidia.get('truncated'):
        issues.append('nvidia_input_truncated')
    if task in {'Other', 'Unknown'} or tasks[task] < rules.min_task_probability:
        issues.append('uncertain_task_type')
    if probabilities[length] < rules.min_length_probability:
        issues.append('uncertain_output_length')
    if task in {'Classification', 'Extraction'} and length != 'short':
        issues.append('task_and_length_disagree')
    preference, reason = 'uncertain', 'Predictions are uncertain or disagree; retain both backend options.'
    if not issues:
        if length == 'short':
            preference, reason = 'prefer_local', 'Likely short output; test local before paying for acceleration.'
        elif reasoning >= rules.high_reasoning:
            preference, reason = 'consider_hosted', 'Substantial output and reasoning signal; accelerate only when workload context justifies it.'
        elif length == 'long':
            preference, reason = 'prefer_local', 'Likely long output with lower reasoning signal; candidate for background local execution.'
        else:
            reason = 'Medium output alone does not establish a backend preference.'
    return {'schema_version': 1, 'preference': preference, 'reason': reason, 'issues': issues,
            'task_type': task, 'task_probability': tasks[task], 'length_class': length,
            'length_probabilities': dict(probabilities), 'reasoning': reasoning, 'constraints': constraints,
            'rules': asdict(rules), 'validation_status': 'experimental_rules_not_calibrated',
            'requires_workload_context': True}


class RequestProfiler:
    def __init__(self, rules=None):
        from .classifiers import build_classifiers
        from .estimator import DurationEstimator
        self.nvidia = build_classifiers(['nvidia'])['nvidia']
        self.clairvoyant = DurationEstimator()  # No deployment timing profile.
        self.rules = rules or ProfileRules()

    def predict(self, messages):
        import time
        started = time.perf_counter()
        nvidia = self.nvidia.predict(messages)
        clairvoyant = self.clairvoyant.predict(messages)
        from .classifier_assets import MODELS
        return {'nvidia_model': {'repo': MODELS['nvidia'][0], 'revision': MODELS['nvidia'][1]},
                'profile': combine_predictions(nvidia, clairvoyant, self.rules),
                'nvidia': nvidia, 'clairvoyant': clairvoyant,
                'profile_elapsed_seconds': time.perf_counter() - started}
