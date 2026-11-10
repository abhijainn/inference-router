"""Load the project's .env without executing code or expanding variables."""
import os
from pathlib import Path
import re


def load_dotenv(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / '.env'
    if not path.exists():
        return
    parsed = {}
    for number, raw in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].lstrip()
        key, separator, value = line.partition('=')
        key = key.strip()
        if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
            raise ValueError(f'Invalid .env syntax on line {number}; expected NAME=value.')
        value = value.strip()
        if value.startswith(('"', "'")):
            quote = value[0]
            end = value.find(quote, 1)
            if end < 0 or (value[end + 1:].strip() and not value[end + 1:].strip().startswith('#')):
                raise ValueError(f'Invalid .env quoting on line {number}.')
            value = value[1:end]
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].rstrip()
        parsed[key] = value
    for key, value in parsed.items():
        os.environ.setdefault(key, value)
