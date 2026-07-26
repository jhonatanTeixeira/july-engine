---
tags:
  - Licensing
---

# Licensing

July Engine's own source code is **MIT** — see [`LICENSE`](https://github.com/jhonatanTeixeira/july-engine/blob/master/LICENSE) in the repository root. That license applies **only** to the original engine code; it does not override the licenses of the third-party packages in `requirements.txt`, several of which impose obligations beyond MIT's — especially if you deploy this engine as a network service or redistribute it commercially. The authoritative version of this information lives in [`NOTICE.md`](https://github.com/jhonatanTeixeira/july-engine/blob/master/NOTICE.md) at the repo root; this page mirrors it for the documentation site.

## Copyleft dependencies (review before commercial/SaaS use)

| Package | License | Why it matters |
|---|---|---|
| `ultralytics` (YOLO11, vision/segmentation) | AGPL-3.0 | Covers both the code and any models trained/fine-tuned with it. Offering this engine (or a derivative) as a network service requires making the complete corresponding source available to users. Ultralytics sells an Enterprise License as an alternative. |
| `PyMuPDF` (PDF handling) | AGPL-3.0 / commercial (dual, via Artifex) | Same network-service source-disclosure concern as above. Artifex sells commercial licenses for proprietary use. |
| `pedalboard` (audio — pitch shifting) | GPL-3.0 (inherited from bundled JUCE/VST3 SDK code) | Copyleft for the *distributed* binary/package, not just network use. |

## Restricted model weights (non-commercial by default)

| Package | Code license | Weight/model restriction |
|---|---|---|
| `insightface` | MIT | Pretrained models auto-downloaded at runtime (e.g. `buffalo_l`, used by [Stable Diffusion LCM / FaceID](models/stable_diffusion_lcm.md)) are licensed for **non-commercial research use only**. Commercial use requires a separate license from InsightFace. |
| `coqui-tts` | MPL-2.0 | Some higher-quality voices — notably [XTTS v2](models/tts_xtts2.md)'s weights — are distributed under the **Coqui Public Model License (CPML)**, which is non-commercial. Confirm the license of whichever voice model you actually load before commercial use. |

## Practical guidance

- Using these packages as ordinary pip dependencies (not modifying/vendoring their source into this repo) is normal practice and doesn't change this repo's own MIT license.
- The AGPL/GPL obligations above attach to **you** if you build on top of this engine and distribute it or run it as a network service — they're independent of what license this repo declares for its own code.
- This page is not a substitute for legal advice. Before commercial or SaaS use, run a full dependency license audit (e.g. `pip-licenses`) and review the current terms of each package directly — licenses and model terms can change between releases.
