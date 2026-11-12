"""Optional frozen OPT-125M length ranker; no generation or network at inference."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

REPO = 'LLM-ltr/OPT-Predictors'
REVISION = '39df2b41ffe88d5ed967c6035d3838b5b5960379'
SUBDIR = 'opt-125m-llama3-8b-lmsys-score-trainbucket10-b32'
TOKENIZER_REPO = 'facebook/opt-125m'
TOKENIZER_REVISION = '27dcfa74d334bc871f3234de431e71c6eeba5dd6'
WEIGHT_SHA256 = 'e7739d26217ee23eb14b6282ee5088f9c79bdc477f456c027b7ebcffacde61ab'
ROOT = Path(os.environ.get('INFERENCE_ROUTER_LTR_DIR', Path(__file__).resolve().parent.parent / '.models/ltr'))


def prompt_text(messages):
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get('role') not in {'user', 'system', 'assistant'}
        or not isinstance(m.get('content'), str) for m in messages
    ):
        raise ValueError('Require nonempty text messages')
    # Our 14 prompts each contain one user message, matching plain-prompt input.
    if len(messages) == 1 and messages[0]['role'] == 'user':
        return messages[0]['content']
    return '\n\n'.join(f"{m['role']}: {m['content']}" for m in messages)


def length_order_score(raw_score):
    """Training labels are inverse length: higher raw logits mean shorter."""
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)) or not math.isfinite(raw_score):
        raise ValueError('LTR score must be finite')
    return -float(raw_score)


def verify_assets(root):
    root = Path(root)
    manifest = json.loads((root / 'download_manifest.json').read_text())
    if (manifest.get('repo'), manifest.get('revision'), manifest.get('tokenizer_revision')) != (REPO, REVISION, TOKENIZER_REVISION):
        raise ValueError('Unexpected LTR asset revisions; run scripts/download_ltr.py')
    required = {'model.safetensors', 'config.json', 'usage_config.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'}
    if set(manifest.get('sha256', {})) != required:
        raise ValueError('Incomplete LTR asset manifest')
    for filename, expected in manifest['sha256'].items():
        digest = hashlib.sha256()
        with (root / filename).open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected or (filename == 'model.safetensors' and expected != WEIGHT_SHA256):
            raise ValueError(f'LTR asset checksum mismatch: {filename}')
    return manifest


class LTRPredictor:
    """Frozen ranker on CPU, one thread; scores are not tokens or probabilities."""
    def __init__(self, root=None):
        started = time.perf_counter()
        root = Path(root) if root is not None else ROOT
        try:
            self.manifest = verify_assets(root)
        except FileNotFoundError as exc:
            raise ValueError('Download LTR assets first: python scripts/download_ltr.py') from exc
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(1)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.model, info = AutoModelForSequenceClassification.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, output_loading_info=True, attn_implementation='eager')
        if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys'):
            raise ValueError('LTR weights do not exactly match the model architecture')
        if self.model.config.model_type != 'opt' or self.model.config.num_labels != 1:
            raise ValueError('Require the pinned single-score OPT ranker')
        self.model.eval()
        self.max_length = 2048
        self.load_seconds = time.perf_counter() - started

    def predict(self, messages):
        started = time.perf_counter()
        text = prompt_text(messages)
        count = len(self.tokenizer(text, truncation=False)['input_ids'])
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors='pt')
        with self.torch.inference_mode():
            raw = float(self.model(**inputs, use_cache=False).logits[0, 0].item())
        return {'model': REPO, 'revision': REVISION, 'checkpoint': SUBDIR,
                'tokenizer_revision': TOKENIZER_REVISION, 'raw_score': raw,
                'length_order_score': length_order_score(raw),
                'length_order_source': f'ltr:{REVISION}:{SUBDIR}',
                'score_semantics': 'higher length_order_score means longer; no calibrated units',
                'input_tokens': count, 'max_input_tokens': self.max_length, 'truncated': count > self.max_length,
                'device': 'cpu', 'dtype': 'float32', 'threads': 1,
                'elapsed_seconds': time.perf_counter() - started}
