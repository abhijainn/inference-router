import json
import importlib.util
from inference_router.model_assets import MODEL_PATH
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from inference_router.deepinfra import DeepInfraClient, DeepInfraError


class DeepInfraTests(unittest.TestCase):
    def setUp(self):
        from examples.make_requests import CALIBRATION_JOBS, COMPLEX_JOBS, write_jobs
        self.job_directory = tempfile.TemporaryDirectory()
        self.jobs_path = str(Path(self.job_directory.name) / 'requests.jsonl')
        write_jobs(self.jobs_path, COMPLEX_JOBS)
        self.calls = []
        self.mode = 'ok'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body, status=200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                owner.calls.append(('GET', self.path, dict(self.headers), None))
                if self.path != '/models/list':
                    return self.reply({}, 404)
                self.reply([{'model_name': 'google/gemma-3-4b-it', 'quantization': 'bfloat16',
                    'max_tokens': 131072, 'deprecated': 1 if owner.mode == 'deprecated' else None,
                    'pricing': {'cents_per_input_token': 0.000005, 'cents_per_output_token': 0.00001}}])

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.calls.append(('POST', self.path, dict(self.headers), body))
                if owner.mode == 'unauthorized':
                    return self.reply({'error': 'bad secret-test-key'}, 401)
                self.reply({'id': 'test-request', 'model': 'another-model' if owner.mode == 'wrong-model' else body['model'],
                    'choices': [{'finish_reason': 'length' if owner.mode == 'length' else 'stop',
                        'message': {'role': 'assistant', 'content': 'OK'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 1}})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1/openai'
        self.client = DeepInfraClient(self.url, api_key='secret-test-key')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.job_directory.cleanup()

    def test_catalog_does_not_claim_authenticated_access_or_send_key(self):
        identity = self.client.inspect()
        self.assertFalse(identity['credentials_verified'])
        self.assertFalse(identity['artifact_digest_verified'])
        self.assertEqual(identity['details']['quantization_level'], 'bfloat16')
        self.assertAlmostEqual(identity['pricing_usd_per_million']['input'], 0.05)
        self.assertAlmostEqual(identity['pricing_usd_per_million']['output'], 0.10)
        self.assertNotIn('Authorization', self.calls[0][2])
        self.mode = 'deprecated'
        with self.assertRaisesRegex(DeepInfraError, 'deprecated'):
            self.client.inspect()

    def test_chat_auth_payload_and_honest_missing_server_timings(self):
        result = self.client.chat([{'role': 'user', 'content': 'Return OK'}], -1)
        method, path, headers, body = self.calls[-1]
        self.assertEqual(headers['Authorization'], 'Bearer secret-test-key')
        self.assertEqual(path, '/v1/openai/chat/completions')
        self.assertNotIn('max_tokens', body)
        self.assertNotIn('think', body)
        self.assertNotIn('num_ctx', body)
        self.assertEqual(body['temperature'], 0)
        self.assertEqual(body['seed'], 42)
        self.assertIsNone(result['generation_seconds'])
        self.assertIsNone(result['generation_tokens_per_second'])
        self.assertIsNone(result['max_output_tokens'])
        self.assertEqual(result['output_limit_mode'], 'provider_default')
        self.assertTrue(result['naturally_stopped'])

    def test_explicit_limit_and_truncation_recorded(self):
        self.mode = 'length'
        result = self.client.chat([{'role': 'user', 'content': 'Hello'}], 64)
        self.assertEqual(self.calls[-1][3]['max_tokens'], 64)
        self.assertEqual(result['done_reason'], 'length')
        self.assertFalse(result['naturally_stopped'])

    def test_error_redaction_no_retry_and_model_mismatch(self):
        self.mode = 'unauthorized'
        with self.assertRaises(DeepInfraError) as error:
            self.client.chat([{'role': 'user', 'content': 'Hello'}])
        self.assertNotIn('secret-test-key', str(error.exception))
        self.assertEqual(len(self.calls), 1)
        self.mode = 'wrong-model'
        with self.assertRaisesRegex(DeepInfraError, 'differs'):
            self.client.chat([{'role': 'user', 'content': 'Hello'}])

    def test_missing_key_and_insecure_remote_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DeepInfraError, 'DEEPINFRA_API_KEY'):
                DeepInfraClient()
        with self.assertRaises(ValueError):
            DeepInfraClient('http://example.com', api_key='test')

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime") is not None and importlib.util.find_spec("numpy") is not None and MODEL_PATH.is_file(), "optional estimator dependencies not installed")
    def test_hosted_profile_bootstraps_then_scores_before_updating(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'predicted.jsonl'
            profile = Path(directory) / 'hosted-profile.json'
            process = subprocess.run([sys.executable, '-m', 'inference_router',
                '--backend', 'deepinfra', '--base-url', self.url, 'run',
                self.jobs_path, '--max-tokens', '-1',
                '--estimate', '--timing-profile', str(profile), '--update-timing-profile',
                '--output', str(output)], env=dict(os.environ, DEEPINFRA_API_KEY='secret-test-key'),
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([r['duration_prediction']['timing_profile']['sample_count'] for r in rows], [0,1,2,3])
            self.assertIsNone(rows[0]['duration_prediction_error'])
            self.assertTrue(all(r['duration_prediction_error'] is not None for r in rows[1:]))
            self.assertTrue(all('response' not in r and r['observed_output_proxy_units'] == 0 for r in rows))
            data = json.loads(profile.read_text())
            self.assertEqual(data['schema_version'], 2)
            self.assertEqual(data['configuration']['backend'], 'deepinfra')
            self.assertEqual(len(data['samples']), 4)

    def test_cli_hosted_run_skips_warmup_and_keeps_separate_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'hosted.jsonl'
            env = dict(os.environ, DEEPINFRA_API_KEY='secret-test-key')
            process = subprocess.run([sys.executable, '-m', 'inference_router',
                '--backend', 'deepinfra', '--base-url', self.url, '--timeout', '0',
                'run', self.jobs_path, '--max-tokens', '-1',
                '--save-responses', '--output', str(output)], env=env, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len([c for c in self.calls if c[0] == 'POST']), 4)
            self.assertTrue(all(r['success'] and r['backend'] == 'deepinfra' and not r['warmup_performed'] for r in rows))
            self.assertTrue(all(r['requested_max_output_tokens'] == -1 and r['response'] == 'OK' for r in rows))
            self.assertNotIn('secret-test-key', output.read_text())
