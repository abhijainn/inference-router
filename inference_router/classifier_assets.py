"""Pinned optional classifiers; weights stay outside Git and load locally only."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('INFERENCE_ROUTER_CLASSIFIER_DIR', Path(__file__).resolve().parent.parent / '.models' / 'classifiers'))
MODELS = {
    'nvidia': ('nvidia/prompt-task-and-complexity-classifier', 'fea1121511eafabaf7dd6fc66863dcb04f74defb'),
    'gliclass': ('knowledgator/gliclass-small-v1.0', '21edefaf7951f68c68c505f9139ba536d3b448f7'),
}
BACKBONE = ('microsoft/deberta-v3-base', '8ccc9b6f36199bec6961081d44eb72fb3f7353f3')
FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json', 'spm.model', 'model.safetensors']

WEIGHT_SHA256 = {
    "nvidia": "d9abb6b67d266a712a7f5baea5117bcb8f4be4bec325cf4c7856d35df0fcb953",
    "gliclass": "ed3d94af3541d078e91d9518a8bfb87b090f73da488184d3cd6a1feed4511278",
}
