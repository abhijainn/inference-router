import importlib.util
from inference_router.model_assets import MODEL_PATH
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inference_router.estimator import DurationEstimator, TimingProfile, fingerprint, timing_sample, prediction_error

IDENTITY = {'model': 'gemma-local', 'digest': 'test-digest', 'ollama_version': 'test-server'}
HOSTED = {'backend': 'deepinfra', 'model': 'google/gemma-3-4b-it', 'base_url': 'https://api.deepinfra.com/v1/openai', 'details': {'quantization_level': 'bfloat16'}}
HAS_RUNTIME = importlib.util.find_spec('onnxruntime') is not None and importlib.util.find_spec('numpy') is not None and MODEL_PATH.is_file()


def example_row(length=100, hosted=False, **changes):
    row = {'backend': 'deepinfra' if hosted else 'local_ollama', 'success': True,
           'done_reason': 'stop', 'concurrency': 1, 'max_output_tokens': -1,
           'requested_max_output_tokens': -1, 'warmup_performed': not hosted,
           'response': 'A' * (length * 4), 'generation_seconds': None if hosted else 10,
           'client_elapsed_seconds': 11, 'model_identity': HOSTED if hosted else IDENTITY,
           'num_ctx': None if hosted else 4096, 'temperature': 0, 'seed': 42,
           'thinking_requested': None if hosted else False}
    row.update(changes)
    return row


class EstimatorTests(unittest.TestCase):
    def test_empirical_representatives_and_rates(self):
        profile = TimingProfile.from_rows([example_row(100), example_row(500), example_row(1200)])
        summary = profile.summary()
        self.assertEqual(summary['category_mean_lengths'], {'short': 100, 'medium': 500, 'long': 1200})
        self.assertEqual(summary['proxy_units_per_second'], 60)
        self.assertEqual(summary['overhead_seconds'], 1)
        result = profile.convert({'short': 0.6, 'medium': 0.3, 'long': 0.1})
        self.assertAlmostEqual(result['expected_output_proxy_units'], 330)
        self.assertAlmostEqual(result['estimated_seconds'], 6.5)
        self.assertIsNone(result['upper_estimate_seconds'])

    def test_hosted_uses_wall_rate_without_double_counting_overhead(self):
        profile = TimingProfile.from_rows([example_row(100, hosted=True), example_row(500, hosted=True), example_row(1200, hosted=True)])
        result = profile.convert({'short': 0.6, 'medium': 0.3, 'long': 0.1})
        self.assertEqual(profile.summary()['overhead_seconds'], 0)
        self.assertAlmostEqual(result['estimated_seconds'], 330 / (1800 / 33))
        with self.assertRaises(ValueError):
            profile.check_configuration(IDENTITY)

    def test_missing_categories_use_measured_mean_without_invented_lengths(self):
        profile = TimingProfile.from_rows([example_row(100)])
        result = profile.convert({'short': 0.1, 'medium': 0.2, 'long': 0.7})
        self.assertEqual(result['estimated_seconds'], 11)
        self.assertIsNone(result['expected_output_proxy_units'])
        self.assertEqual(result['missing_categories'], ['medium', 'long'])
        self.assertEqual(result['timing_status'], 'pooled_mean_fallback_missing_categories')
        self.assertEqual(TimingProfile(fingerprint(IDENTITY)).convert({'short':1,'medium':0,'long':0})['timing_status'], 'no_completed_observations')

    def test_excludes_bad_capped_cold_and_unavailable_reasoning_lengths(self):
        for changes in ({'success': False}, {'done_reason': 'length'}, {'warmup_performed': False},
                        {'response': None}, {'thinking_field_observed': True},
                        {'generation_seconds': float('nan')}, {'max_output_tokens': 128}):
            self.assertIsNone(timing_sample(example_row(**changes)))
        self.assertIsNone(timing_sample(example_row(hosted=True, service_tier_returned='flex')))
        self.assertIsNotNone(timing_sample(example_row(response=None, observed_output_proxy_units=100)))

    def test_configuration_and_deduplication(self):
        profile = TimingProfile.from_rows([example_row(run_id='one', job_id='x', repeat=0)])
        self.assertFalse(profile.update(example_row(run_id='one', job_id='x', repeat=0)))
        with self.assertRaises(ValueError):
            TimingProfile.from_rows([example_row(), example_row(seed=99)])
        with self.assertRaises(ValueError):
            profile.check_configuration(dict(IDENTITY, digest='other'))

    def test_window_roundtrip_and_legacy_rejection(self):
        profile = TimingProfile.from_rows([example_row()])
        for _ in range(110):
            profile.update(example_row(500))
        self.assertEqual(len(profile.samples), 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            profile.save(path)
            self.assertEqual(TimingProfile.load(path).summary(), profile.summary())
            path.write_text('{"schema_version":1}')
            with self.assertRaisesRegex(ValueError, 'Legacy'):
                TimingProfile.load(path)

    def test_error_scoring(self):
        result = prediction_error({'estimated_seconds': 10}, 15)
        self.assertEqual(result['absolute_error_seconds'], 5)
        self.assertTrue(result['underestimated'])
        self.assertIsNone(prediction_error({'estimated_seconds': None}, 15))

    @unittest.skipUnless(HAS_RUNTIME, 'optional estimator dependencies not installed')
    def test_frozen_model_and_empirical_conversion(self):
        estimator = DurationEstimator(TimingProfile.from_rows([example_row(100), example_row(500), example_row(1200)]))
        result = estimator.predict([{'role':'user','content':'Return only the word OK.'}])
        self.assertTrue(result['classifier_frozen'])
        self.assertAlmostEqual(result['length_class_probabilities']['short'], 0.7436023, places=5)
        expected = 1 + sum(result['length_class_probabilities'][k] * n for k,n in result['timing_profile']['category_mean_lengths'].items()) / 60
        self.assertAlmostEqual(result['estimated_seconds'], expected)
        self.assertNotIn('assumed_class_representatives', result)
        self.assertIsNone(estimator.predict([{'content':'Hello'}],128)['estimated_seconds'])

    @unittest.skipUnless(HAS_RUNTIME, 'optional estimator dependencies not installed')
    def test_compare_profiles_offline_and_no_ollama_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            profiles=[]
            for hosted in [False, True]:
                path=Path(directory)/('hosted.json' if hosted else 'local.json')
                TimingProfile.from_rows([example_row(n,hosted=hosted) for n in [100,500,1200]]).save(path)
                profiles += ['--timing-profile',str(path)]
            process=subprocess.run([sys.executable,'-m','inference_router','--base-url','http://127.0.0.1:1','estimate','Return OK',*profiles],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads(process.stdout)
            self.assertEqual(len(result['backend_predictions']),2)
            self.assertNotEqual(result['backend_predictions'][0]['estimated_seconds'],result['backend_predictions'][1]['estimated_seconds'])
