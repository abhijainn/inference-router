"""Explicitly download pinned public classifier assets; never executes model code."""
import hashlib
import json
from pathlib import Path
import ssl
import sys
from urllib.request import urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_router.classifier_assets import ROOT, MODELS, BACKBONE, FILES, WEIGHT_SHA256


def download():
    context = ssl.create_default_context(cafile='/etc/ssl/cert.pem' if sys.platform == 'darwin' else None)
    for name, (repo, revision) in MODELS.items():
        dest = ROOT / name
        dest.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for filename in FILES + (['backbone_config.json'] if name == 'nvidia' else []):
            origin, commit, remote = (BACKBONE[0], BACKBONE[1], 'config.json') if filename == 'backbone_config.json' else (repo, revision, filename)
            path = dest / filename
            if not path.exists():
                print(f'Downloading {name}/{filename}', flush=True)
                temporary = path.with_suffix(path.suffix + '.partial')
                try:
                    with urlopen(f'https://huggingface.co/{origin}/resolve/{commit}/{remote}', context=context, timeout=120) as response, temporary.open('wb') as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            digest = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            manifest[filename] = digest.hexdigest()
        if manifest['model.safetensors'] != WEIGHT_SHA256[name]:
            raise ValueError(f'Unexpected weight checksum for {name}; assets were not accepted.')
        (dest / 'download_manifest.json').write_text(json.dumps({'repo': repo, 'revision': revision, 'sha256': manifest}, indent=2))
        print(f'Ready: {name}', flush=True)


if __name__ == '__main__':
    download()
