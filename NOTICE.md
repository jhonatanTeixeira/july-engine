# Third-Party License Notice

The MIT `LICENSE` in this repository applies only to the original July Engine
source code. It does **not** override the licenses of the third-party
packages listed in `requirements.txt`, several of which impose obligations
beyond MIT's — especially for anyone deploying this engine as a network
service or redistributing it commercially.

## Copyleft dependencies (require attention before commercial/SaaS use)

- **`ultralytics`** (YOLO11, used for vision/segmentation) — licensed
  **AGPL-3.0**. Both the code and any models trained/fine-tuned with it are
  covered. If this engine (or a derivative) is offered as a network service,
  AGPL-3.0 requires making the complete corresponding source of the combined
  work available to users. Ultralytics sells an Enterprise License as an
  alternative to AGPL obligations.
- **`PyMuPDF`** (PDF handling) — dual-licensed **AGPL-3.0** / commercial, via
  Artifex. Same network-service source-disclosure concern as above; Artifex
  sells commercial licenses for proprietary use.
- **`pedalboard`** (audio processing) — **GPL-3.0** (inherited from the
  bundled JUCE/VST3 SDK code). GPL-3.0 is copyleft for the *distributed
  binary/package*, not just network use.

## Restricted pretrained model weights (non-commercial by default)

- **`insightface`** — the Python code is MIT, but its pretrained models
  (auto-downloaded at runtime, e.g. `buffalo_l`, `inswapper` face-swap
  models) are licensed for **non-commercial research use only**. Commercial
  use requires a separate license from InsightFace.
- **`coqui-tts`** — the library code is MPL-2.0 (permissive), but some
  higher-quality voices (notably XTTS v2 weights) are distributed under the
  **Coqui Public Model License (CPML)**, which is non-commercial. Confirm the
  license of whichever voice models are actually loaded before commercial use.

## Practical guidance

- Using these packages as ordinary pip dependencies (not modifying/vendoring
  their source into this repo) is normal practice and does not change this
  repo's own MIT license.
- The AGPL/GPL obligations above attach to **you** if you build on top of
  this engine and distribute it or run it as a network service — they are
  independent of what license this repo declares for its own code.
- This list is not a substitute for legal advice. Before commercial or SaaS
  use, run a full dependency license audit (e.g. `pip-licenses`) and review
  the current terms of each package directly, since licenses and model terms
  can change between releases.
