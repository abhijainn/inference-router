import copy
import unittest
from inference_router.policy import BackendEstimate as E, choose_backend
from inference_router.policy_replay import replay


def row(backend, ident, estimate, actual):
    return dict(backend=backend, job_id=ident, repeat=0, success=True, done_reason='stop',
                client_elapsed_seconds=actual, duration_prediction={'estimated_seconds': estimate})


class PolicyTests(unittest.TestCase):
    def test_prefer_local_and_inclusive_budget(self):
        self.assertEqual(choose_backend(E(4), E(1), 5)['backend'], 'local')

    def test_hosted_fallback(self):
        self.assertEqual(choose_backend(E(9), E(3), 5)['backend'], 'hosted')

    def test_unknown_and_neither_fit(self):
        self.assertEqual(choose_backend(E(None), E(9), 5)['status'], 'uncertain')
        self.assertEqual(choose_backend(E(8), E(9), 5)['status'], 'neither_fits')

    def test_eligibility_and_queue(self):
        self.assertEqual(choose_backend(E(1, eligible=False), E(2), 5)['backend'], 'hosted')
        result = choose_backend(E(2, 3), E(8), 5.5)
        self.assertEqual(result['adjusted_completion_seconds']['local'], 5.5)
        self.assertEqual(result['backend'], 'local')

    def test_invalid_inputs(self):
        for value in [-1, float('nan'), float('inf'), True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                choose_backend(E(value), E(1), 5)
        for budget, factor in [(0, 1), (5, .9), (float('nan'), 1)]:
            with self.assertRaises(ValueError):
                choose_backend(E(1), E(1), budget, factor)

    def test_replay_counts_abstentions_as_misses_without_avoidance_credit(self):
        local = [row('local_ollama', 'a', 1, 8), row('local_ollama', 'b', 9, 2), row('local_ollama', 'c', 9, 9)]
        hosted = [row('deepinfra', 'a', 1, 2), row('deepinfra', 'b', 1, 1), row('deepinfra', 'c', 9, 8)]
        metrics = replay(local, hosted, [5])['scenarios'][0]['metrics']
        policy = metrics['policy']
        self.assertEqual((policy['hosted_calls_avoided'], policy['hosted_calls'], policy['budget_misses'], policy['late_completions'], policy['not_routed']), (1, 1, 2, 1, 1))
        self.assertEqual(policy['unsafe_local_choices'], 1)
        self.assertEqual(policy['unnecessary_hosted_calls'], 1)
        self.assertEqual(metrics['always_local']['budget_misses'], 2)
        self.assertEqual(metrics['always_hosted']['budget_misses'], 1)

    def test_outcomes_do_not_change_decisions(self):
        local = [row('local_ollama', 'a', 1, 8)]
        hosted = [row('deepinfra', 'a', 1, 2)]
        before = replay(local, hosted, [5])['scenarios'][0]['decisions'][0]['decision']
        local[0]['client_elapsed_seconds'] = .1
        hosted[0]['client_elapsed_seconds'] = 100
        after = replay(local, hosted, [5])['scenarios'][0]['decisions'][0]['decision']
        self.assertEqual(before, after)

    def test_reject_bad_pairs(self):
        local, hosted = [row('local_ollama', 'a', 1, 1)], [row('deepinfra', 'a', 1, 1)]
        cases = [(local * 2, hosted), (local, []), (local, [row('deepinfra', 'b', 1, 1)])]
        truncated = copy.deepcopy(local)
        truncated[0]['done_reason'] = 'length'
        cases.append((truncated, hosted))
        no_prediction = copy.deepcopy(local)
        no_prediction[0].pop('duration_prediction')
        cases.append((no_prediction, hosted))
        for left, right in cases:
            with self.assertRaises(ValueError):
                replay(left, right)
        local[0]['request_features'] = {'request_sha256': 'a'}
        hosted[0]['request_features'] = {'request_sha256': 'b'}
        with self.assertRaises(ValueError):
            replay(local, hosted)

    def test_leave_one_out_excludes_same_prompt_aliases_and_preserves_recorded(self):
        from inference_router.policy_replay import leave_one_out_predictions
        from test_estimator import example_row
        jobs = [{'id': name, 'messages': [{'role': 'user', 'content': text}]} for name, text in [('a', 'same'), ('alias', 'same'), ('b', 'other')]]
        calibration = [example_row(length, job_id=name, repeat=0) for name, length in [('a', 100), ('alias', 200), ('b', 900)]]
        class FakeEstimator:
            def predict(self, messages, max_tokens):
                lengths = [sample['length'] for sample in self.profile.samples]
                if messages[0]['content'] == 'same':
                    self_outer.assertEqual(lengths, [900])
                else:
                    self_outer.assertEqual(lengths, [100, 200])
                return {'estimated_seconds': 12}
        self_outer = self
        recorded = copy.deepcopy(calibration[0])
        recorded['duration_prediction'] = {'estimated_seconds': 123}
        left, right = leave_one_out_predictions(calibration + [recorded], [], calibration, [], jobs, FakeEstimator())
        self.assertEqual(left[0]['calibration_samples_excluded'], 2)
        self.assertEqual(left[-1]['duration_prediction']['estimated_seconds'], 123)
        self.assertEqual(left[-1]['prediction_origin'], 'recorded_before_execution')
        self.assertNotIn('duration_prediction', calibration[0])

    def test_policy_and_replay_cli_work_without_api_calls(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        result = subprocess.run([sys.executable, '-m', 'inference_router', '--base-url', 'http://invalid.invalid', 'policy', '--budget-seconds', '5', '--local-seconds', '2', '--hosted-seconds', '1'], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)['backend'], 'local')
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ['local.jsonl', 'hosted.jsonl']]
            for path, backend in zip(paths, ['local_ollama', 'deepinfra']):
                path.write_text(json.dumps(row(backend, 'a', 1, 2))+'\n')
            result = subprocess.run([sys.executable, '-m', 'inference_router', 'replay-policy', '--local-log', str(paths[0]), '--hosted-log', str(paths[1]), '--budgets', '5', '--format', 'markdown'], capture_output=True, text=True, check=True)
            self.assertIn('| 5 sec | policy | 1/1 | 0 | 0/1 |', result.stdout)
