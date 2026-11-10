"""Frozen length classifier with per-deployment empirical duration conversion."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from .vendor.clairvoyant.feature_extractor import extract_features

from .model_assets import MODEL_PATH, MODEL_REVISION, MODEL_SHA256, FEATURE_SHA256
CLASSES = ('short', 'medium', 'long')


def length_class(length):
    return 'short' if length < 200 else 'medium' if length < 800 else 'long'


def fingerprint(identity, num_ctx=4096, temperature=0, seed=42, thinking_requested=False):
    backend = identity.get('backend', 'local_ollama')
    hosted = backend == 'deepinfra'
    return {'backend': backend, 'model': identity.get('model'),
            'digest': identity.get('digest'), 'ollama_version': identity.get('ollama_version'),
            'base_url': identity.get('base_url', 'http://127.0.0.1:11434' if not hosted else 'https://api.deepinfra.com/v1/openai'),
            'quantization': (identity.get('details') or {}).get('quantization_level'),
            'num_ctx': None if hosted else num_ctx, 'temperature': temperature,
            'seed': seed, 'thinking_requested': None if hosted else thinking_requested,
            'concurrency': 1, 'service_tier': 'standard' if hosted else None}


def row_fingerprint(row):
    return fingerprint(row.get('model_identity', {}), row.get('num_ctx'),
                       row.get('temperature'), row.get('seed'), row.get('thinking_requested'))


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def timing_sample(row):
    backend = row.get('backend', 'local_ollama')
    if (row.get('success') is not True or row.get('done_reason') != 'stop'
            or row.get('concurrency') != 1 or row.get('thinking_field_observed') is True
            or (row.get('thinking_detection_method') is None and row.get('thinking_observed') is True)):
        return None
    if backend == 'local_ollama' and row.get('warmup_performed') is not True:
        return None
    # Only natural-completion requests with no CLIENT output cap are comparable.
    if backend == 'deepinfra':
        if row.get('requested_max_output_tokens') != -1 or row.get('service_tier_requested', 'standard') != 'standard':
            return None
        if row.get('service_tier_returned') not in (None, 'standard', 'default'):
            return None
    elif row.get('max_output_tokens') != -1:
        return None
    length = row.get('observed_output_proxy_units')
    if length is None and isinstance(row.get('response'), str):
        length = len(row['response']) // 4
    elapsed = row.get('client_elapsed_seconds')
    if not isinstance(length, int) or isinstance(length, bool) or length < 0 or not finite_positive(elapsed):
        return None
    if backend == 'deepinfra':
        generation, overhead = elapsed, 0.0
    else:
        generation = row.get('generation_seconds')
        if not finite_positive(generation) or elapsed < generation:
            return None
        overhead = elapsed - generation
    return {'length': length, 'length_class': length_class(length),
            'duration_seconds': elapsed, 'processing_seconds': generation,
            'overhead_seconds': overhead,
            'key': [row.get('run_id'), row.get('job_id'), row.get('repeat')]}


class TimingProfile:
    def __init__(self, configuration, samples=None, source=None):
        self.configuration = configuration
        self.samples = list(samples or [])[-100:]
        self.source = source

    @classmethod
    def from_rows(cls, rows, source=None):
        profile = None
        for row in rows:
            if timing_sample(row) is None:
                continue
            config = row_fingerprint(row)
            if not config.get('model') or (config['backend'] == 'local_ollama' and not config.get('digest')):
                continue
            if profile is None:
                profile = cls(config, source=source)
            if config != profile.configuration:
                raise ValueError('Logs mix backend/model/settings configurations; build separate profiles.')
            profile.update(row)
        if profile is None or not profile.samples:
            raise ValueError('No eligible naturally stopped, uncapped rows with saved response lengths.')
        return profile

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get('schema_version') != 2:
            raise ValueError('Legacy timing profile lacks measured category lengths. Rebuild with build-timing-profile from the original measurements.')
        obj = cls(data['configuration'], data['samples'], data.get('source'))
        for sample in obj.samples:
            if (not isinstance(sample.get('length'), int) or isinstance(sample['length'], bool)
                    or sample['length'] < 0 or sample.get('length_class') != length_class(sample['length'])
                    or not finite_positive(sample.get('duration_seconds'))
                    or not finite_positive(sample.get('processing_seconds'))
                    or not isinstance(sample.get('overhead_seconds'), (int, float))
                    or not math.isfinite(sample['overhead_seconds']) or sample['overhead_seconds'] < 0):
                raise ValueError('Invalid timing profile samples')
        return obj

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        temporary = p.with_name(p.name + '.tmp')
        temporary.write_text(json.dumps({'schema_version': 2, 'configuration': self.configuration,
            'source': self.source, 'samples': self.samples}, indent=2) + '\n')
        temporary.replace(p)

    def check_configuration(self, identity):
        if self.configuration != fingerprint(identity):
            raise ValueError('Timing profile does not match this backend/model/version/settings. Use a separate profile.')

    def update(self, row):
        sample = timing_sample(row)
        if sample is None:
            return False
        if row_fingerprint(row) != self.configuration:
            raise ValueError('Cannot update a profile with a different backend or configuration.')
        if any(v is not None for v in sample['key']) and any(s.get('key') == sample['key'] for s in self.samples):
            return False
        self.samples = (self.samples + [sample])[-100:]
        return True

    def summary(self):
        groups = {k: [s['length'] for s in self.samples if s['length_class'] == k] for k in CLASSES}
        processing = sum(s['processing_seconds'] for s in self.samples)
        return {'sample_count': len(self.samples),
                'category_counts': {k: len(v) for k, v in groups.items()},
                'category_mean_lengths': {k: statistics.mean(v) if v else None for k, v in groups.items()},
                'proxy_units_per_second': sum(s['length'] for s in self.samples) / processing if processing else None,
                'overhead_seconds': statistics.mean(s['overhead_seconds'] for s in self.samples) if self.samples else None,
                'pooled_mean_seconds': statistics.mean(s['duration_seconds'] for s in self.samples) if self.samples else None,
                'speed_basis': 'end_to_end_including_network_and_provider_wait' if self.configuration['backend'] == 'deepinfra' else 'server_generation_plus_separate_client_overhead'}

    def convert(self, probabilities):
        summary = self.summary()
        result = {'estimated_seconds': None, 'expected_output_proxy_units': None,
                  'upper_estimate_seconds': None, 'timing_profile': summary,
                  'timing_configuration': self.configuration,
                  'category_representatives_source': 'observed_completed_response_means',
                  'confidence_status': 'not_calibrated', 'timing_status': 'no_completed_observations'}
        if not self.samples:
            return result
        representatives = summary['category_mean_lengths']
        missing = [k for k in CLASSES if representatives[k] is None]
        if missing or not summary['proxy_units_per_second']:
            result.update(estimated_seconds=summary['pooled_mean_seconds'],
                          timing_status='pooled_mean_fallback_missing_categories', missing_categories=missing)
            return result
        length = sum(probabilities[k] * representatives[k] for k in CLASSES)
        result.update(expected_output_proxy_units=length,
                      estimated_seconds=summary['overhead_seconds'] + length / summary['proxy_units_per_second'],
                      timing_status='empirical_category_length_conversion')
        return result


def prediction_error(prediction, actual_seconds):
    estimate = prediction.get('estimated_seconds')
    if not finite_positive(estimate) or not finite_positive(actual_seconds):
        return None
    return {'predicted_seconds': estimate, 'actual_seconds': actual_seconds,
            'signed_error_seconds': actual_seconds - estimate,
            'absolute_error_seconds': abs(actual_seconds - estimate),
            'absolute_percentage_error': abs(actual_seconds - estimate) / actual_seconds,
            'underestimated': actual_seconds > estimate}
class DurationEstimator:
    def __init__(self, profile=None):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise ValueError('Estimator requires its environment: use .venv-estimator/bin/python instead of python3.') from exc
        self.np = np
        self.profile = profile
        self.provenance = {'revision': MODEL_REVISION}
        if not MODEL_PATH.is_file():
            raise ValueError('Predictor weights are missing. Run: python scripts/download_estimator.py')
        feature_path = Path(__file__).parent / 'vendor' / 'clairvoyant' / 'feature_extractor.py'
        for path, expected in [(MODEL_PATH, MODEL_SHA256), (feature_path, FEATURE_SHA256)]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Pretrained artifact integrity check failed: {path.name}')
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL_PATH),
            sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape[-1] != 19:
            raise ValueError('Unexpected pretrained model input schema')
        self.input_name = inputs[0].name
        # Warm the tiny predictor once; this never calls the language model.
        self._probabilities('Reply with OK.')

    def _probabilities(self, prompt):
        result = self.session.run(['probabilities'], {self.input_name:
            self.np.asarray([extract_features(prompt)], dtype=self.np.float32)})[0][0]
        probabilities = [float(v) for v in result]
        if len(probabilities) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in probabilities) or abs(sum(probabilities) - 1) > 0.001:
            raise ValueError('Invalid classifier probabilities')
        return probabilities

    def predict(self, messages, max_tokens=-1):
        started = time.perf_counter()
        if not messages or any(not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('Estimator requires text messages')
        prompt = '\n\n'.join(m['content'] for m in messages)
        probs = self._probabilities(prompt)
        output = {'estimator': 'clairvoyant_model_b', 'model_revision': self.provenance['revision'],
                  'classifier_frozen': True, 'length_class': CLASSES[max(range(3), key=probs.__getitem__)],
                  'length_class_probabilities': dict(zip(CLASSES, probs)),
                  'length_unit': 'characters_divided_by_4_proxy_not_actual_model_tokens',
                  'estimated_seconds': None, 'upper_estimate_seconds': None,
                  'timing_status': 'no_timing_profile',
                  'warnings': ['Class probabilities and duration estimates are not calibrated confidence levels.',
                               'No finite upper bound or deadline guarantee is available.']}
        if max_tokens != -1:
            output['timing_status'] = 'finite_output_cap_not_supported_for_duration_conversion'
        elif self.profile is not None:
            output.update(self.profile.convert(output['length_class_probabilities']))
            if output['timing_profile']['sample_count'] < 30:
                output['warnings'].append('Small pilot profile; 30 is a display threshold, not a guarantee of accuracy.')
            if output['timing_status'] == 'pooled_mean_fallback_missing_categories':
                output['warnings'].append('Some categories have no observations. Duration uses the measured overall mean, not invented category lengths.')
        output['prediction_elapsed_seconds'] = time.perf_counter() - started
        return output
