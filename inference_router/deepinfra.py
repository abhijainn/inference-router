"""DeepInfra text chat adapter; standard library only, no automatic retries."""
import json
import os
from pathlib import Path
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, HTTPSHandler, build_opener

from .ollama import OllamaError, thinking_evidence


class DeepInfraError(OllamaError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DeepInfraClient:
    def __init__(self, base_url='https://api.deepinfra.com/v1/openai',
                 model='google/gemma-3-4b-it', timeout=180, api_key=None):
        parsed = urlsplit(base_url)
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('DeepInfra requires HTTPS (HTTP localhost is allowed for tests).')
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Base URL must not contain credentials, query, or fragment.')
        if timeout < 0:
            raise ValueError('timeout must be nonnegative; 0 disables it')
        self.api_key = api_key if api_key is not None else os.getenv('DEEPINFRA_API_KEY', '')
        if not self.api_key.strip():
            raise DeepInfraError('Set DEEPINFRA_API_KEY in this terminal first. Do not paste the key into chat.')
        if '\n' in self.api_key or '\r' in self.api_key:
            raise ValueError('API key contains a newline; re-enter it in the terminal.')
        self.base_url = base_url.rstrip('/')
        self.catalog_url = f'{parsed.scheme}://{parsed.netloc}/models/list'
        self.model = model
        self.timeout = None if timeout == 0 else timeout
        # Python.org installs may not yet have their bundled CA file configured.
        # Use macOS's existing verified CA bundle; never disable verification.
        cafile = '/etc/ssl/cert.pem' if sys.platform == 'darwin' and Path('/etc/ssl/cert.pem').exists() and not os.getenv('SSL_CERT_FILE') else None
        context = ssl.create_default_context(cafile=cafile)
        self.opener = build_opener(NoRedirect(), HTTPSHandler(context=context))
        self.identity = None

    def _request(self, url, payload=None, authenticated=False):
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer ' + self.api_key
        body = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(url, data=body, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace').replace(self.api_key, '[REDACTED]')[:500]
            exc.close()
            hint = {401: 'Check your API key.', 402: 'Check account credits/billing.',
                    403: 'Check model access and account permissions.',
                    429: 'Rate limited or capacity unavailable; no automatic retry was made.'}.get(exc.code, '')
            raise DeepInfraError(f'DeepInfra HTTP {exc.code}. {hint} {detail}') from None
        except (URLError, TimeoutError, OSError) as exc:
            detail = str(exc).replace(self.api_key, '[REDACTED]')
            raise DeepInfraError(f'DeepInfra request failed: {detail}. No retry was made; a timed-out generation may still be running.') from None
        except (ValueError, UnicodeError):
            raise DeepInfraError('DeepInfra returned invalid JSON.') from None

    def inspect(self):
        models = self._request(self.catalog_url)
        if not isinstance(models, list):
            raise DeepInfraError('Unexpected public model catalog response.')
        selected = next((m for m in models if m.get('model_name') == self.model), None)
        if selected is None:
            raise DeepInfraError(f'Model {self.model!r} is not listed in the current DeepInfra catalog.')
        if selected.get('deprecated') or selected.get('replaced_by'):
            raise DeepInfraError('This model is marked deprecated/replaced; no substitution was made.')
        pricing = selected.get('pricing') or {}
        def dollars_per_million(key):
            value = pricing.get(key)
            return value * 10000 if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        self.identity = {
            'backend': 'deepinfra', 'base_url': self.base_url, 'model': self.model,
            'digest': None, 'artifact_digest_verified': False,
            'details': {'quantization_level': selected.get('quantization'),
                        'context_limit': selected.get('max_tokens')},
            'pricing_usd_per_million': {
                'input': dollars_per_million('cents_per_input_token'),
                'output': dollars_per_million('cents_per_output_token'),
            },
            'pricing_scope': 'standard public list rates, excluding special discounts and cached-token adjustments',
            'catalog_checked_at_unix': time.time(),
            'api_key_present': True, 'credentials_verified': False,
            'check_scope': 'Public catalog only. Run chat to verify authenticated inference; chat is billable.',
        }
        return self.identity

    def chat(self, messages, max_tokens=128, num_ctx=None):
        if not messages or any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) for m in messages):
            raise ValueError('messages must contain text system/user/assistant objects')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or (max_tokens != -1 and max_tokens <= 0):
            raise ValueError('max_tokens must be -1 or a positive integer')
        payload = {'model': self.model, 'messages': messages, 'stream': False,
                   'temperature': 0, 'seed': 42}
        if max_tokens != -1:
            payload['max_tokens'] = max_tokens
        started = time.perf_counter()
        raw = self._request(self.base_url + '/chat/completions', payload, authenticated=True)
        elapsed = time.perf_counter() - started
        try:
            choice = raw['choices'][0]
            message = choice['message']
            content = message['content']
            finish = choice['finish_reason']
        except (KeyError, IndexError, TypeError):
            raise DeepInfraError('DeepInfra returned a malformed chat response.') from None
        if not isinstance(content, str) or not finish:
            raise DeepInfraError('DeepInfra returned an incomplete or non-text chat response.')
        if raw.get('model') and raw['model'] != self.model:
            raise DeepInfraError('Returned model differs from the requested model; response was not accepted as a matched measurement.')
        usage = raw.get('usage') or {}
        def count(key):
            value = usage.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        prompt_tokens, output_tokens = count('prompt_tokens'), count('completion_tokens')
        detail = usage.get('prompt_tokens_details') or {}
        cached = detail.get('cached_tokens')
        if not isinstance(cached, int) or isinstance(cached, bool) or cached < 0:
            cached = None
        field = message.get('reasoning_content') or message.get('reasoning')
        labels = thinking_evidence({'content': content, 'thinking': field})
        # No server phase timings are invented from non-streaming wall time.
        return {
            'backend': 'deepinfra', 'model_requested': self.model,
            'model_returned': raw.get('model'), 'provider_request_id': raw.get('id'),
            'client_elapsed_seconds': elapsed, 'server_total_seconds': None,
            'load_seconds': None, 'prompt_eval_seconds': None, 'generation_seconds': None,
            'generation_tokens_per_second': None,
            'end_to_end_output_tokens_per_second': output_tokens / elapsed if output_tokens is not None and elapsed > 0 else None,
            'input_tokens': prompt_tokens, 'output_tokens': output_tokens,
            'cached_input_tokens': cached,
            'max_output_tokens': None if max_tokens == -1 else max_tokens,
            'requested_max_output_tokens': max_tokens,
            'output_limit_mode': 'provider_default' if max_tokens == -1 else 'explicit',
            'provider_default_output_limit_known': False if max_tokens == -1 else None,
            'num_ctx': None, 'temperature': 0, 'seed': 42,
            'thinking_requested': None, **labels,
            'done_reason': finish, 'naturally_stopped': finish == 'stop',
            'service_tier_requested': 'standard', 'service_tier_returned': raw.get('service_tier'),
            'response': content,
        }
