"""Small Ollama adapter using only the Python standard library."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def thinking_evidence(message):
    """Report observable evidence, not whether hidden reasoning occurred.

    Response markers are a heuristic: quoted tags can also trigger detection.
    In particular a closing-only tag can occur when the opening tag was part
    of the generation prompt. Never delete or reinterpret the raw content.
    """
    content = message.get("content", "")
    field = message.get("thinking")
    field_observed = isinstance(field, str) and bool(field.strip())
    markers = isinstance(content, str) and any(
        marker in content for marker in ("<think>", "</think>")
    )
    evidence = []
    if field_observed:
        evidence.append("thinking_field")
    if markers:
        evidence.append("response_markers")
    return {
        "thinking_observed": field_observed or markers,
        "thinking_field_observed": field_observed,
        "thinking_markers_in_response": markers,
        "thinking_evidence": evidence,
        "thinking_detection_method": "field_or_markers_v1",
    }


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", model="gemma3:4b-it-q4_K_M", timeout=180):
        if urlparse(base_url).scheme not in {"http", "https"}:
            raise ValueError("base_url must start with http:// or https://")
        if timeout < 0:
            raise ValueError("timeout must be nonnegative (0 disables the socket timeout)")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = None if timeout == 0 else timeout

    def _request(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.base_url + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaError(
                f"Cannot complete request to {self.base_url}: {exc}. "
                "Check Ollama is running on this machine (ollama serve). "
                "A timeout does not guarantee server-side generation stopped."
            ) from exc
        except (ValueError, UnicodeError) as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise OllamaError("Expected an Ollama JSON object")
        if result.get("error"):
            raise OllamaError(str(result["error"]))
        return result

    def inspect(self):
        version = self._request("/api/version").get("version")
        models = self._request("/api/tags").get("models", [])
        selected = next((m for m in models if m.get("name") == self.model or m.get("model") == self.model), None)
        if selected is None:
            raise OllamaError(f"Model {self.model!r} is not installed. Run: ollama pull {self.model}")
        info = self._request("/api/show", {"model": self.model})
        return {
            "backend": "local_ollama",
            "base_url": self.base_url,
            "ollama_version": version,
            "model": self.model,
            "digest": selected.get("digest"),
            "details": selected.get("details"),
            "size_bytes": selected.get("size"),
            "thinking_controls": info.get("thinking"),
        }

    def chat(self, messages, max_tokens=128, num_ctx=4096):
        if not messages or any(
            not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}
            or not isinstance(m.get("content"), str) for m in messages
        ):
            raise ValueError("messages must be a nonempty list of role/content text objects")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or (max_tokens != -1 and max_tokens <= 0):
            raise ValueError("max_tokens must be -1 (no output cap) or a positive integer")
        if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx <= 0 or (max_tokens != -1 and num_ctx <= max_tokens):
            raise ValueError("num_ctx must be positive and larger than a finite max_tokens")
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"num_predict": max_tokens, "num_ctx": num_ctx, "temperature": 0, "seed": 42},
        }
        started = time.perf_counter()
        raw = self._request("/api/chat", payload)
        elapsed = time.perf_counter() - started
        message = raw.get("message")
        if raw.get("done") is not True or not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaError("Ollama returned an incomplete or malformed chat response")

        def seconds(key):
            value = raw.get(key)
            return value / 1_000_000_000 if isinstance(value, (int, float)) else None

        generation = seconds("eval_duration")
        count = raw.get("eval_count")
        return {
            "backend": "local_ollama",
            "model_requested": self.model,
            "model_returned": raw.get("model"),
            "client_elapsed_seconds": elapsed,
            "server_total_seconds": seconds("total_duration"),
            "load_seconds": seconds("load_duration"),
            "prompt_eval_seconds": seconds("prompt_eval_duration"),
            "generation_seconds": generation,
            "input_tokens": raw.get("prompt_eval_count"),
            "cached_input_tokens": raw.get("prompt_eval_cached_count"),
            "output_tokens": count,
            "generation_tokens_per_second": count / generation if generation and isinstance(count, int) else None,
            "max_output_tokens": max_tokens,
            "num_ctx": num_ctx,
            "temperature": 0,
            "seed": 42,
            "thinking_requested": False,
            **thinking_evidence(message),
            "done_reason": raw.get("done_reason"),
            "response": message["content"],
        }
