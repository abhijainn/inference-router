"""Generate lightweight input files locally; generated JSONL is ignored by Git."""
import json
from pathlib import Path

CALIBRATION_JOBS = [{'id': 'extract-01',
  'messages': [{'role': 'user',
                'content': 'Extract the company and city from this sentence: Maya works at Acme '
                           'Robotics in Austin. Return only a JSON object with keys company and '
                           'city.'}],
  'max_tokens': -1},
 {'id': 'classify-01',
  'messages': [{'role': 'user',
                'content': 'Classify this support ticket into exactly one category: billing, '
                           'technical, or account. Ticket: I was charged twice for my monthly '
                           'subscription. Return only the category.'}],
  'max_tokens': -1},
 {'id': 'summarize-01',
  'messages': [{'role': 'user',
                'content': 'Summarize this update in three bullet points, using no more than 60 '
                           'words total: The team launched a document search pilot on Monday for '
                           '40 employees. By Friday, 28 employees had tried it and submitted 93 '
                           'searches. Feedback praised the clear citations but identified slow '
                           'responses for long documents. The team will improve document indexing '
                           'next week and postpone expanding the pilot until response times '
                           'improve.'}],
  'max_tokens': -1},
 {'id': 'explain-01',
  'messages': [{'role': 'user',
                'content': 'Explain the difference between a queue and a stack in at most 100 '
                           'words. Include one practical example of each.'}],
  'max_tokens': -1},
 {'id': 'code-01',
  'messages': [{'role': 'user',
                'content': 'Write a Python function named deduplicate(items) that removes '
                           'duplicate hashable items while preserving their original order. Return '
                           'only the function, without explanation or Markdown fences.'}],
  'max_tokens': -1},
 {'id': 'incident-01',
  'messages': [{'role': 'user',
                'content': 'Investigate this fictional service incident.\n'
                           'Timeline:\n'
                           '09:00: A deployment increases worker concurrency from 8 to 32. '
                           'Database connection pool size remains 20.\n'
                           '09:03: Job queue length falls from 600 to 120.\n'
                           '09:05: Database connection acquisition p95 rises from 20 ms to 4 '
                           'seconds.\n'
                           '09:06: API error rate rises from 0.2% to 8%. CPU utilization is 45%; '
                           'memory utilization is 60%.\n'
                           '09:08: Worker logs show connection acquisition timeouts.\n'
                           '09:10: A dashboard reports database query execution time unchanged at '
                           '15 ms.\n'
                           '09:12: Worker concurrency is restored to 8.\n'
                           '09:15: API error rate returns to 0.3%, but the queue begins growing.\n'
                           'Identify the most likely cause, explain why the unchanged query '
                           'execution time does or does not contradict it, and distinguish '
                           'observations from hypotheses. Propose an immediate mitigation and two '
                           'experiments that could test your diagnosis. Explain the tradeoffs of '
                           'each experiment.'}],
  'max_tokens': -1},
 {'id': 'code-review-01',
  'messages': [{'role': 'user',
                'content': 'Review this fictional payment-processing code:\n'
                           '\n'
                           'def process(job, db, payments, queue):\n'
                           '    if db.is_completed(job.id):\n'
                           '        return\n'
                           '    payments.charge(job.customer_id, job.amount)\n'
                           '    db.mark_completed(job.id)\n'
                           '    queue.ack(job.id)\n'
                           '\n'
                           'Queue delivery is at least once. Two workers may receive the same job '
                           'concurrently. A worker may crash after any operation. The payment API '
                           'supports an idempotency key, but its result may be lost if the network '
                           'connection drops. The database supports transactions and unique '
                           'constraints.\n'
                           'Identify the failure cases. Propose revised pseudocode and the '
                           'database fields or constraints it requires. Explain how recovery works '
                           'after an ambiguous payment timeout. Specify tests for duplicate '
                           'delivery, concurrent workers, and crashes. State any guarantees that '
                           "depend on the payment provider's idempotency behavior."}],
  'max_tokens': -1},
 {'id': 'constraint-plan-01',
  'messages': [{'role': 'user',
                'content': 'Schedule six non-preemptive tasks on two identical workers. Each '
                           'worker can execute only one task at a time. Tasks may start at time '
                           'zero when their dependencies permit:\n'
                           'A: duration 4; no dependencies.\n'
                           'B: duration 7; no dependencies.\n'
                           'C: duration 3; depends on A.\n'
                           'D: duration 5; depends on A.\n'
                           'E: duration 2; depends on B and C.\n'
                           'F: duration 4; depends on D and E.\n'
                           'Minimize the completion time of F. Return a schedule with worker '
                           'assignments, start times, and finish times. Verify that every '
                           'dependency and worker constraint is satisfied. Explain whether your '
                           'schedule is optimal using a lower bound. Then determine how the best '
                           'completion time changes if B takes 10 instead of 7.'}],
  'max_tokens': -1},
 {'id': 'evidence-synthesis-01',
  'messages': [{'role': 'user',
                'content': 'Assess whether this fictional document-search pilot is ready to '
                           'expand:\n'
                           'Product report: 40 employees invited; 28 used the product; 93 searches '
                           'recorded; average satisfaction 4.5/5 from 12 respondents.\n'
                           'Engineering report: median response time 3 seconds; p95 24 seconds; 8 '
                           'searches timed out. Timed-out searches were excluded from latency '
                           'statistics.\n'
                           'Quality review: 20 randomly sampled completed searches; 16 answers '
                           'fully supported by their citations, 3 partially supported, and 1 '
                           'unsupported.\n'
                           'Sales report: two prospective customers require p95 response time '
                           'below 10 seconds and reliable citations.\n'
                           'Operations report: long-document indexing failed for 6 of 50 uploaded '
                           'documents. Users were not notified of these failures.\n'
                           'Recommend expansion, a limited extension, or a pause. Reconcile the '
                           'evidence, identify misleading or incomplete metrics, and calculate the '
                           'rates that can be calculated from the supplied data. Propose '
                           'measurable release criteria and the next three experiments. Do not '
                           'invent missing observations.'}],
  'max_tokens': -1},
 {'id': 'runbook-medium-01',
  'messages': [{'role': 'user',
                'content': 'Write an approximately 300-word operational runbook for a worker that '
                           'loses its database connection midway through processing a job. Cover '
                           'detection, immediate containment, checking whether side effects '
                           'already happened, safe recovery, and verification. Use five numbered '
                           'sections. Keep it practical and state assumptions.'}],
  'max_tokens': -1},
 {'id': 'architecture-medium-01',
  'messages': [{'role': 'user',
                'content': 'Write an approximately 300-word design review comparing a single '
                           'durable queue with per-customer queues for an asynchronous report '
                           'generator. Address fairness, overload, retries, operational '
                           'complexity, and observability. Recommend one approach for a small '
                           'service and explain when you would reconsider it.'}],
  'max_tokens': -1}]

VALIDATION_JOBS = [{'id': 'holdout-incident-01',
  'messages': [{'role': 'user',
                'content': 'Investigate this fictional incident: at 14:00 a cache TTL changed from '
                           '60 seconds to 1 second. At 14:02 cache hit rate fell from 95% to 20%, '
                           'database queries per second rose fourfold, and API p95 increased from '
                           '200 ms to 3 seconds. CPU on API workers remained 40%. At 14:10 '
                           'restoring the TTL returned latency to 250 ms. Explain the likely '
                           'cause, one alternative hypothesis, what the rollback establishes and '
                           'what it does not, and two checks before trying the change again.'}],
  'max_tokens': -1},
 {'id': 'holdout-code-01',
  'messages': [{'role': 'user',
                'content': 'Write a Python function merge_intervals(intervals) for pairs of '
                           'integer start/end values. Merge overlapping or touching closed '
                           'intervals, return a sorted list of tuples, do not mutate input, and '
                           'reject reversed intervals with ValueError. Explain the algorithm and '
                           'its complexity, then provide six assert-based tests including empty '
                           'input, negatives, duplicates, nesting, touching intervals, and invalid '
                           'input.'}],
  'max_tokens': -1},
 {'id': 'holdout-classify-01',
  'messages': [{'role': 'user',
                'content': 'Return exactly one label, billing, technical, or account, for this '
                           'support ticket: I can no longer sign in after changing my email '
                           'address.'}],
  'max_tokens': -1}]

COMPLEX_JOBS = [job for job in CALIBRATION_JOBS if job['id'] in {
    'incident-01', 'code-review-01', 'constraint-plan-01', 'evidence-synthesis-01'}]


def write_jobs(path, jobs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(job) + '\n' for job in jobs), encoding='utf-8')


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent / 'generated'
    write_jobs(folder / 'calibration.jsonl', CALIBRATION_JOBS)
    write_jobs(folder / 'validation.jsonl', VALIDATION_JOBS)
    write_jobs(folder / 'complex.jsonl', COMPLEX_JOBS)
    print(f'Generated 11 calibration, 3 validation, and 4 complex prompts in {folder}')
