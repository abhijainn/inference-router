"""Pinned upstream model identity; downloading is an explicit setup step."""
import os
from pathlib import Path

MODEL_REVISION = 'b645ef9601631f0e560d2af78bfca3975a13981f'
MODEL_SHA256 = '6089e25672b59242cf43cc4d9f2f10ae58d5eb145f429a95a0e7333d07f32af5'
FEATURE_SHA256 = '19cef6e997a3bdf4d9dec5dcd39a32d652b841b57db12c65e820ea3e8ff21b26'
MODEL_NAME = 'predictor_model_b.onnx'
MODEL_URL = f'https://huggingface.co/Aravind0495/clairvoyant-scheduler/resolve/{MODEL_REVISION}/{MODEL_NAME}'
MODEL_DIR = Path(os.environ.get('INFERENCE_ROUTER_MODEL_DIR', Path(__file__).resolve().parent.parent / '.models' / 'clairvoyant'))
MODEL_PATH = MODEL_DIR / MODEL_NAME
