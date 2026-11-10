import json
import importlib.util
from inference_router.model_assets import MODEL_PATH
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from inference_router.ollama import OllamaClient, OllamaError


class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        from examples.make_requests import CALIBRATION_JOBS, COMPLEX_JOBS, write_jobs
        self.job_directory = tempfile.TemporaryDirectory()
        self.jobs_path = str(Path(self.job_directory.name) / 'requests.jsonl')
        write_jobs(self.jobs_path, CALIBRATION_JOBS[:3])
        self.calls = []
        self.mode = "ok"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, body, status=200):
                encoded = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                if self.path == "/api/version":
                    self.reply({"version": "test-server"})
                elif self.path == "/api/tags":
                    self.reply({"models": [{"name": "gemma3:4b-it-q4_K_M", "digest": "test-digest", "details": {"quantization_level": "Q4_K_M"}}]})
                else:
                    self.reply({"error": "not found"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    self.reply({"thinking": {"values": [True, False]}})
                    return
                owner.calls.append(body)
                if owner.mode == "error":
                    self.reply({"error": "insufficient memory"}, 500)
                    return
                self.reply({
                    "model": "gemma3:4b-it-q4_K_M", "done": owner.mode != "incomplete",
                    "message": {"role": "assistant", "content": "Example response"},
                    "total_duration": 1_500_000_000,
                    "load_duration": 100_000_000,
                    "prompt_eval_count": 20, "prompt_eval_duration": 200_000_000,
                    "eval_count": 10, "eval_duration": 1 if owner.mode == "timing" else 1_000_000_000,
                    "done_reason": "stop",
                })

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.client = OllamaClient(self.url)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.job_directory.cleanup()

    def test_wire_payload_and_metrics(self):
        self.assertEqual(self.client.inspect()["digest"], "test-digest")
        result = self.client.chat([{"role": "user", "content": "hello"}], 64)
        self.assertFalse(self.calls[0]["stream"])
        self.assertFalse(self.calls[0]["think"])
        self.assertEqual(self.calls[0]["options"]["num_predict"], 64)
        self.assertEqual(result["server_total_seconds"], 1.5)
        self.assertEqual(result["generation_tokens_per_second"], 10)
        self.assertEqual(result["output_tokens"], 10)
        self.assertGreater(result["client_elapsed_seconds"], 0)

    def test_uncapped_request_and_invalid_limits(self):
        result = self.client.chat([{"role": "user", "content": "hello"}], -1)
        self.assertEqual(self.calls[-1]["options"]["num_predict"], -1)
        self.assertEqual(result["max_output_tokens"], -1)
        self.assertEqual(result["done_reason"], "stop")
        for invalid in (0, -2, True):
            with self.assertRaises(ValueError):
                self.client.chat([{"role": "user", "content": "hello"}], invalid)

    def test_zero_timeout_disables_socket_timeout(self):
        client = OllamaClient(self.url, timeout=0)
        self.assertIsNone(client.timeout)
        self.assertEqual(client.chat([{"role": "user", "content": "hello"}], -1)["done_reason"], "stop")

    def test_cli_uncapped_override_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "uncapped.jsonl"
            process = subprocess.run([
                sys.executable, "-m", "inference_router", "--base-url", self.url,
                "--timeout", "0", "run", self.jobs_path,
                "--max-tokens", "-1", "--output", str(output), "--save-responses",
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(self.calls[0]["options"]["num_predict"], 8)
            self.assertTrue(all(call["options"]["num_predict"] == -1 for call in self.calls[1:]))
            self.assertTrue(all(row["max_output_tokens"] == -1 and row["done_reason"] == "stop" and "response" in row for row in rows))

    @unittest.skipUnless(importlib.util.find_spec("onnxruntime") is not None and importlib.util.find_spec("numpy") is not None and MODEL_PATH.is_file(), "optional estimator dependencies not installed")
    def test_predictions_are_logged_before_timing_profile_updates(self):
        self.mode = "timing"
        from inference_router.estimator import TimingProfile, fingerprint
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            profile_path = Path(directory) / "profile.json"
            TimingProfile.from_rows([{
                "backend": "local_ollama", "model_identity": self.client.inspect(),
                "num_ctx": 4096, "temperature": 0, "seed": 42, "thinking_requested": False,
                "success": True, "done_reason": "stop", "concurrency": 1,
                "max_output_tokens": -1, "warmup_performed": True,
                "response": "x" * 400, "generation_seconds": 10, "client_elapsed_seconds": 11,
            }]).save(profile_path)
            process = subprocess.run([
                sys.executable, "-m", "inference_router", "--base-url", self.url,
                "run", self.jobs_path, "--max-tokens", "-1",
                "--estimate", "--timing-profile", str(profile_path),
                "--update-timing-profile", "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["duration_prediction"]["timing_profile"]["sample_count"] for row in rows], [1, 2, 3])
            self.assertTrue(all(row["success"] and "response" not in row and row["duration_prediction_error"] is not None for row in rows))
            self.assertEqual(TimingProfile.load(profile_path).summary()["sample_count"], 4)

    def test_errors_are_not_silently_retried(self):
        self.mode = "error"
        with self.assertRaisesRegex(OllamaError, "insufficient memory"):
            self.client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_result_rejected(self):
        self.mode = "incomplete"
        with self.assertRaisesRegex(OllamaError, "incomplete"):
            self.client.chat([{"role": "user", "content": "hello"}])

    def test_missing_model_message(self):
        with self.assertRaisesRegex(OllamaError, "ollama pull"):
            OllamaClient(self.url, "not-installed").inspect()

    def test_cli_worker_writes_measurements_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurements.jsonl"
            process = subprocess.run([
                sys.executable, "-m", "inference_router", "--base-url", self.url,
                "run", self.jobs_path, "--repeat", "2", "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(self.calls), 7)
            self.assertTrue(all(row["success"] and row["concurrency"] == 1 for row in rows))
            self.assertTrue(all("response" not in row and "messages" not in row for row in rows))

    def test_worker_logs_failure_and_stops(self):
        self.mode = "error"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failures.jsonl"
            process = subprocess.run([
                sys.executable, "-m", "inference_router", "--base-url", self.url,
                "run", self.jobs_path, "--no-warmup", "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(process.returncode, 1)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["success"])
            self.assertEqual(len(self.calls), 1)


if __name__ == "__main__":
    unittest.main()
