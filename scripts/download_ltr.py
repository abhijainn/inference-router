"""Download only pinned LTR-125M weights and OPT tokenizer; no remote code."""
import hashlib
import json
import ssl
import sys
from pathlib import Path
from urllib.request import urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_router.ltr import ROOT, REPO, REVISION, SUBDIR, TOKENIZER_REPO, TOKENIZER_REVISION, WEIGHT_SHA256


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem' if sys.platform == 'darwin' else None)
    sources = {
        'model.safetensors': (REPO, REVISION, f'{SUBDIR}/finetuned/model.safetensors'),
        'config.json': (REPO, REVISION, f'{SUBDIR}/finetuned/config.json'),
        'usage_config.json': (REPO, REVISION, f'{SUBDIR}/usage_config.json'),
    }
    for filename in ['vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json']:
        sources[filename] = (TOKENIZER_REPO, TOKENIZER_REVISION, filename)
    checksums = {}
    for filename, (repo, revision, remote) in sources.items():
        path = ROOT / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            print(f'Downloading {filename}', flush=True)
            try:
                with urlopen(f'https://huggingface.co/{repo}/resolve/{revision}/{remote}', context=context, timeout=120) as response, temporary.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        checksums[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    if checksums['model.safetensors'] != WEIGHT_SHA256:
        raise ValueError('LTR weight checksum mismatch')
    (ROOT / 'download_manifest.json').write_text(json.dumps({
        'repo': REPO, 'revision': REVISION, 'checkpoint': SUBDIR,
        'tokenizer_repo': TOKENIZER_REPO, 'tokenizer_revision': TOKENIZER_REVISION,
        'sha256': checksums}, indent=2))
    print('Verified frozen LTR-125M assets; no Gemma model downloaded or run.')


if __name__ == '__main__': main()
