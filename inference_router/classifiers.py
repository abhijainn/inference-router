"""Optional frozen prompt features. No backend decision or duration conversion."""
import hashlib
import json
import time
from .classifier_assets import ROOT, MODELS, FILES, WEIGHT_SHA256

TASK_LABELS = ['Open QA', 'Closed QA', 'Summarization', 'Text Generation', 'Code Generation',
               'Chatbot', 'Classification', 'Rewrite', 'Brainstorming', 'Extraction', 'Other']
ATTRIBUTE_LABELS = {
    'tiny_answer': 'the requested answer is a single category or a few extracted fields',
    'brief_answer': 'the requested answer is a short summary or brief explanation',
    'source_code': 'the requested answer includes source code',
    'detailed_answer': 'the requested answer is a detailed analysis, plan, review, or design',
    'structured_format': 'the answer must be JSON, a table, a list, or numbered sections',
    'evidence_or_tradeoffs': 'the task asks for tradeoffs, alternative explanations, assessment of evidence, or comparison of design choices',
}


def prompt_text(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('Expected nonempty messages')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in {'system', 'user', 'assistant'} or not isinstance(message.get('content'), str):
            raise ValueError('Invalid message')
    if len(messages) == 1 and messages[0]['role'] == 'user':
        return messages[0]['content']
    return '\n\n'.join(f"{m['role']}: {m['content']}" for m in messages)


def assets(name):
    path = ROOT / name
    manifest_path = path / 'download_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('Classifier assets missing. Run scripts/download_classifiers.py first.')
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('repo'), manifest.get('revision')) != MODELS[name]:
        raise ValueError('Classifier revision does not match pinned assets')
    required = set(FILES + (['backbone_config.json'] if name == 'nvidia' else []))
    if set(manifest.get('sha256', {})) != required or manifest['sha256'].get('model.safetensors') != WEIGHT_SHA256[name]:
        raise ValueError('Incomplete classifier manifest or unexpected weight checksum')
    for filename, expected in manifest['sha256'].items():
        digest = hashlib.sha256()
        with (path / filename).open('rb') as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Classifier asset checksum mismatch: {name}/{filename}')
    return path


def decode_nvidia(probabilities, config):
    """Published weighted-head conversion, independent of the runtime."""
    task_probs = probabilities['task_type']
    ranked = sorted(range(len(task_probs)), key=lambda i: task_probs[i], reverse=True)
    result = {'task_type': config['task_type_map'][str(ranked[0])],
              'task_probability': task_probs[ranked[0]],
              'task_probabilities': {config['task_type_map'][str(i)]: v for i, v in enumerate(task_probs)}}
    for name, values in probabilities.items():
        if name == 'task_type':
            continue
        score = sum(v * w for v, w in zip(values, config['weights_map'][name])) / config['divisor_map'][name]
        result[name] = 0.0 if name == 'number_of_few_shots' and score < .05 else score
    result['prompt_complexity_score'] = sum(result[k] * w for k, w in {
        'creativity_scope': .35, 'reasoning': .25, 'constraint_ct': .15,
        'domain_knowledge': .15, 'contextual_knowledge': .05, 'number_of_few_shots': .05}.items())
    return result


class NvidiaClassifier:
    def __init__(self):
        started = time.perf_counter()
        import torch
        from torch import nn
        from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model
        from safetensors.torch import load_file
        path = assets('nvidia')
        self.torch = torch
        self.config = json.loads((path / 'config.json').read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        config = self.config
        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = DebertaV2Model(DebertaV2Config.from_json_file(str(path / 'backbone_config.json')))
                self.heads = []
                for index, size in enumerate(config['target_sizes'].values()):
                    head = nn.Module()
                    head.fc = nn.Linear(self.backbone.config.hidden_size, size)
                    self.add_module(f'head_{index}', head)
                    self.heads.append(head)
            def forward(self, tokens):
                hidden = self.backbone(**tokens).last_hidden_state
                mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                return [head.fc(pooled) for head in self.heads]
        self.model = Network()
        self.model.load_state_dict(load_file(str(path / 'model.safetensors')), strict=True)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def predict(self, messages):
        started = time.perf_counter()
        text = 'Prompt: ' + prompt_text(messages)
        full_count = len(self.tokenizer(text, truncation=False)['input_ids'])
        tokens = self.tokenizer(text, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        with self.torch.inference_mode():
            logits = self.model(tokens)
            probs = {name: self.torch.softmax(value, dim=-1)[0].tolist() for name, value in zip(self.config['target_sizes'], logits)}
        return {**decode_nvidia(probs, self.config), 'input_tokens': full_count,
                'truncated': full_count > 512, 'max_length': 512,
                'elapsed_seconds': time.perf_counter() - started}


class GliClassClassifier:
    def __init__(self):
        started = time.perf_counter()
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
        path = assets('gliclass')
        import torch
        self.torch = torch
        model = GLiClassModel.from_pretrained(str(path), local_files_only=True)
        model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.pipeline = ZeroShotClassificationPipeline(model, self.tokenizer, max_length=512,
                                                      device='cpu', progress_bar=False)
        self.load_seconds = time.perf_counter() - started

    def predict(self, messages):
        started = time.perf_counter()
        text = prompt_text(messages)
        # The pipeline prepends candidate labels; account for that in truncation reporting.
        counts = {}
        for name, labels in [('task', TASK_LABELS), ('attributes', list(ATTRIBUTE_LABELS.values()))]:
            prepared = self.pipeline.pipe.prepare_input(text, labels)
            counts[name] = len(self.tokenizer(prepared, truncation=False)['input_ids'])
        if any(count > 512 for count in counts.values()):
            return {'task_type': None, 'task_probabilities': {}, 'attribute_scores': {},
                    'selected_attributes': [], 'status': 'input_too_long', 'truncated': True,
                    'input_tokens_including_labels': counts, 'max_length': 512, 'forward_passes': 0,
                    'elapsed_seconds': time.perf_counter() - started}
        def scores(labels, single):
            inputs = self.pipeline.pipe.prepare_inputs([text], labels, same_labels=True)
            with self.torch.inference_mode():
                logits = self.pipeline.pipe.model(**inputs).logits[0][:len(labels)]
                values = self.torch.softmax(logits, dim=-1) if single else self.torch.sigmoid(logits)
            return dict(zip(labels, values.tolist()))
        task_scores = scores(TASK_LABELS, True)
        raw_attributes = scores(list(ATTRIBUTE_LABELS.values()), False)
        attr_scores = {key: raw_attributes[label] for key, label in ATTRIBUTE_LABELS.items()}
        return {'task_type': max(task_scores, key=task_scores.get), 'task_probabilities': task_scores,
                'attribute_scores': attr_scores, 'attribute_threshold': .5,
                'selected_attributes': [key for key, score in attr_scores.items() if score >= .5],
                'input_tokens_including_labels': counts, 'truncated': any(n > 512 for n in counts.values()),
                'max_length': 512, 'forward_passes': 2, 'elapsed_seconds': time.perf_counter() - started}


def build_classifiers(names):
    if not names or any(name not in MODELS for name in names):
        raise ValueError('Choose nvidia and/or gliclass')
    try:
        import torch
        torch.set_num_threads(4)
        return {name: (NvidiaClassifier() if name == 'nvidia' else GliClassClassifier()) for name in names}
    except ImportError as exc:
        raise ValueError('Install requirements-classifiers.txt in a separate Python 3.14 environment first.') from exc


def classify_job(job, classifiers, repeat=1):
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError('repeat must be positive')
    result = {'job_id': job['id'], 'request_sha256': hashlib.sha256(json.dumps(job['messages'], sort_keys=True).encode()).hexdigest(),
              'device': 'cpu', 'threads': 4, 'models': {}, 'predictions': {}}
    for name, classifier in classifiers.items():
        predictions = [classifier.predict(job['messages']) for _ in range(repeat)]
        result['models'][name] = {'repo': MODELS[name][0], 'revision': MODELS[name][1]}
        result['predictions'][name] = predictions[0]
        result['predictions'][name]['timing_samples_seconds'] = [p['elapsed_seconds'] for p in predictions]
    return result
