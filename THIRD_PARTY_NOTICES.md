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

## Optional NVIDIA classifier

Checkpoint: [nvidia/prompt-task-and-complexity-classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier), revision `fea1121511eafabaf7dd6fc66863dcb04f74defb`. Weights/tokenizer are downloaded separately under the NVIDIA Open Model License linked in its model card. The local adapter follows the published multi-head architecture and weighted-score definitions; it constructs a DeBERTa-v3-base backbone from Microsoft's configuration pinned at `8ccc9b6f36199bec6961081d44eb72fb3f7353f3`, loading NVIDIA's complete checkpoint without downloading base weights. Assets are not redistributed in this repository.

## Optional GLiClass

Checkpoint: [knowledgator/gliclass-small-v1.0](https://huggingface.co/knowledgator/gliclass-small-v1.0), revision `21edefaf7951f68c68c505f9139ba536d3b448f7`, Apache-2.0 per its model card. Uses the [GLiClass library](https://github.com/Knowledgator/GLiClass) via a pinned optional dependency; no model weights are redistributed. The six custom attribute descriptions and exploratory reference labels are project-defined, not labels provided by NVIDIA or GLiClass.
