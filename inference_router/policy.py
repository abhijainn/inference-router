"""Budget-based backend selection; contains no execution or outcome lookup."""
from dataclasses import asdict, dataclass
import math


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be a finite number >= {minimum}')
    return float(value)


@dataclass(frozen=True)
class BackendEstimate:
    seconds: float | None
    additional_queue_seconds: float = 0
    eligible: bool = True

    def adjusted(self, safety_factor):
        queue = _number(self.additional_queue_seconds, 'additional queue wait')
        if self.seconds is None:
            return None
        return queue + _number(self.seconds, 'estimated duration') * safety_factor


def choose_backend(local, hosted, budget_seconds, safety_factor=1.25):
    """Prefer eligible local inference if it fits, otherwise eligible hosted.

    An allowance is a configured scenario assumption, not a calibrated quantile.
    Additional queue wait must exclude waiting already inside duration estimates.
    Unknown duration is never treated as a fitting backend. No estimate of
    dollar cost is needed: local is preferred when predicted to meet the budget.
    """
    budget = _number(budget_seconds, 'budget', minimum=0.000000001)
    factor = _number(safety_factor, 'safety factor', minimum=1)
    adjusted = {'local': local.adjusted(factor), 'hosted': hosted.adjusted(factor)}
    eligible = {'local': local.eligible, 'hosted': hosted.eligible}
    if not all(isinstance(v, bool) for v in eligible.values()):
        raise ValueError('Eligibility must be boolean')
    choice = None
    reason = None
    for backend in ('local', 'hosted'):
        if eligible[backend] and adjusted[backend] is not None and adjusted[backend] <= budget:
            choice = backend
            reason = 'Local fits the budget and is preferred for lower API spend.' if backend == 'local' else 'Local does not have a known fitting estimate; hosted fits the budget.'
            break
    if choice:
        status = 'routed'
    elif any(eligible[k] and adjusted[k] is None for k in eligible):
        status, reason = 'uncertain', 'An eligible backend has no duration estimate, and no known estimate fits.'
    else:
        status, reason = 'neither_fits', 'No eligible backend is predicted to fit the budget with the configured allowance.'
    return {'backend': choice, 'status': status, 'reason': reason,
            'budget_seconds': budget, 'safety_factor': factor,
            'allowance_source': 'configured_multiplier_not_calibrated_confidence',
            'estimates': {'local': asdict(local), 'hosted': asdict(hosted)},
            'adjusted_completion_seconds': adjusted,
            'margin_seconds': {k: budget - v if v is not None else None for k, v in adjusted.items()}}
