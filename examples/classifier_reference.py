"""Author-reviewed prompt labels, fixed before the 14-prompt classifier run.

Exploratory annotations, not independent or Gemma routing ground truth.
Task labels allow genuine ambiguity for analysis/design tasks.
"""
REFERENCES = {
    'extract-01': (['Extraction'], ['tiny_answer', 'structured_format']),
    'classify-01': (['Classification'], ['tiny_answer']),
    'summarize-01': (['Summarization'], ['brief_answer', 'structured_format']),
    'explain-01': (['Open QA'], ['brief_answer']),
    'code-01': (['Code Generation'], ['source_code']),
    'incident-01': (['Closed QA'], ['detailed_answer', 'evidence_or_tradeoffs']),
    'code-review-01': (['Code Generation', 'Closed QA'], ['source_code', 'detailed_answer']),
    'constraint-plan-01': (['Closed QA'], ['detailed_answer']),
    'evidence-synthesis-01': (['Closed QA'], ['detailed_answer', 'evidence_or_tradeoffs']),
    'runbook-medium-01': (['Text Generation', 'Open QA'], ['detailed_answer', 'structured_format']),
    'architecture-medium-01': (['Text Generation', 'Open QA'], ['detailed_answer', 'evidence_or_tradeoffs']),
    'holdout-incident-01': (['Closed QA'], ['detailed_answer', 'evidence_or_tradeoffs']),
    'holdout-code-01': (['Code Generation'], ['source_code', 'detailed_answer']),
    'holdout-classify-01': (['Classification'], ['tiny_answer']),
}
