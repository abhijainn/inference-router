import unittest
from unittest.mock import patch

from inference_router.ollama import OllamaClient, thinking_evidence


class ThinkingEvidenceTests(unittest.TestCase):
    def test_dedicated_field(self):
        labels = thinking_evidence({"thinking": "Reasoning", "content": "Answer"})
        self.assertTrue(labels["thinking_observed"])
        self.assertTrue(labels["thinking_field_observed"])
        self.assertFalse(labels["thinking_markers_in_response"])
        self.assertEqual(labels["thinking_evidence"], ["thinking_field"])

    def test_inline_and_unclosed_markers(self):
        for content in ("<think>Reasoning</think>Answer", "Reasoning</think>Answer", "<think>unfinished"):
            with self.subTest(content=content):
                labels = thinking_evidence({"content": content})
                self.assertTrue(labels["thinking_observed"])
                self.assertFalse(labels["thinking_field_observed"])
                self.assertEqual(labels["thinking_evidence"], ["response_markers"])

    def test_both_sources(self):
        labels = thinking_evidence({"thinking": "Reasoning", "content": "</think>Answer"})
        self.assertEqual(labels["thinking_evidence"], ["thinking_field", "response_markers"])

    def test_no_evidence_does_not_guess_from_prose(self):
        labels = thinking_evidence({"thinking": "  ", "content": "Let me think about this."})
        self.assertFalse(labels["thinking_observed"])
        self.assertEqual(labels["thinking_evidence"], [])

    def test_chat_preserves_raw_content_and_timings(self):
        raw = {"done": True, "model": "gemma3:4b-it-q4_K_M", "message": {
            "content": "Reasoning</think>\nAnswer"},
            "total_duration": 2000000000, "eval_duration": 1000000000,
            "eval_count": 10, "done_reason": "stop"}
        with patch.object(OllamaClient, "_request", return_value=raw):
            result = OllamaClient().chat([{"role": "user", "content": "Question"}], -1)
        self.assertTrue(result["thinking_observed"])
        self.assertFalse(result["thinking_requested"])
        self.assertEqual(result["response"], raw["message"]["content"])
        self.assertEqual(result["server_total_seconds"], 2)
        self.assertEqual(result["output_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
