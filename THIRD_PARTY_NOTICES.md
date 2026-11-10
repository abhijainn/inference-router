# Third-party components

The feature extractor in `inference_router/vendor/clairvoyant/feature_extractor.py`
is an unmodified copy from [Clairvoyant's published artifact repository](https://huggingface.co/Aravind0495/clairvoyant-scheduler),
revision `b645ef9601631f0e560d2af78bfca3975a13981f`.
The original downloaded license is retained at
`inference_router/vendor/clairvoyant/LICENSE`.

The upstream model card declares MIT, while the LICENSE shipped at that revision
contains Apache License 2.0. This repository retains the shipped license and
records the mismatch rather than assigning a new upstream license.

The pretrained Model B weights are downloaded explicitly during setup, not
committed. Their SHA-256 is
`6089e25672b59242cf43cc4d9f2f10ae58d5eb145f429a95a0e7333d07f32af5`.
The feature extractor SHA-256 is
`19cef6e997a3bdf4d9dec5dcd39a32d652b841b57db12c65e820ea3e8ff21b26`.
Both are checked before prediction. This project does not train or modify the
pretrained classifier.
