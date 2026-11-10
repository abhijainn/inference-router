"""Synthetic workflows over measured Gemma requests; not new concurrent trials."""


def scenarios(profiles, measurements):
    def task(ident, prompt, dependencies=None, arrival=0, background=False):
        return {'id': ident, 'arrival_seconds': arrival, 'dependencies': dependencies or [],
                'background': background, 'profile': profiles[prompt]}, measurements[prompt]
    definitions = {
        'background_and_arrivals': [
            task('background', 'code-review-01', background=True),
            task('ticket', 'classify-01', arrival=1),
            task('extract', 'extract-01', arrival=2),
        ],
        'parallel_join': [
            task('a-review', 'code-review-01'),
            task('b-ticket', 'classify-01'),
            task('c-extract', 'extract-01'),
            task('join', 'holdout-classify-01', ['a-review','b-ticket','c-extract']),
        ],
        'dependency_chain_with_background': [
            task('a-review', 'code-review-01'),
            task('b-code', 'holdout-code-01', ['a-review']),
            task('c-label', 'holdout-classify-01', ['b-code']),
            task('z-background', 'runbook-medium-01', background=True),
        ],
    }
    return {name: {'tasks': [item[0] for item in items],
                   'outcomes': {item[0]['id']: item[1] for item in items}} for name, items in definitions.items()}
