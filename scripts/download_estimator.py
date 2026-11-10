"""Download pinned Clairvoyant weights into the ignored local model directory."""
import hashlib
import os
from pathlib import Path
import ssl
import sys
import tempfile
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_router.model_assets import MODEL_PATH, MODEL_SHA256, MODEL_URL


def download():
    if MODEL_PATH.is_file() and hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256:
        print('Verified existing predictor weights:', MODEL_PATH)
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cafile = '/etc/ssl/cert.pem' if sys.platform == 'darwin' and Path('/etc/ssl/cert.pem').exists() and not os.getenv('SSL_CERT_FILE') else None
    context = ssl.create_default_context(cafile=cafile)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_PATH.parent, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urlopen(MODEL_URL, context=context, timeout=60) as response:
                while chunk := response.read(65536):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError('Downloaded weights failed SHA-256 verification; file was not installed.')
        temporary.replace(MODEL_PATH)
        print('Downloaded and verified pretrained Model B:', MODEL_PATH)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


if __name__ == '__main__':
    download()
