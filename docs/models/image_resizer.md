---
tags:
  - Image
  - CPU
  - GPU
---

# Resizers & Upscalers

**File:** `app/models/image_resizer.py` · **Classes:** `PillowResizerModel`, `OpencvResizerModel`, `LanczosResizerModel`, `HighQualityUpscalerModel`, `OnnxUpscalerModel`, `RealESRGANResizerModel`, `GFPGANResizerModel`, `CodeFormerResizerModel`

Eight resize/upscale/restoration strategies in one file, all subclassing a local `ResizerBase` (**not** `BaseModel`). Selected via `ImageAdapter._make_resizer(tag)`'s `resizer_map`, task type `image_resize`, settings key `RESIZE`.

| Alias | Class | What it does | Backend |
|---|---|---|---|
| `pillow` | `PillowResizerModel` | Plain resize, Pillow LANCZOS filter | CPU |
| `opencv` | `OpencvResizerModel` | Plain resize, OpenCV `INTER_LANCZOS4` | CPU |
| `lanczos` | `LanczosResizerModel` | Alias — identical to `pillow` (`class LanczosResizerModel(PillowResizerModel): pass`) | CPU |
| `high_quality` | `HighQualityUpscalerModel` | Pillow LANCZOS resize + `UnsharpMask` sharpening — no real neural network despite the name | CPU |
| `onnx` | `OnnxUpscalerModel` | :material-alert: Documented no-op skeleton — see below | CPU/GPU (unused) |
| `realesrgan` / `upscale` | `RealESRGANResizerModel` | Real-ESRGAN 4x AI super-resolution | CPU/GPU |
| `gfpgan` / `face_restoration` | `GFPGANResizerModel` | GFPGAN v1.4 face restoration | CPU/GPU |
| `codeformer` | `CodeFormerResizerModel` | CodeFormer face restoration (native arch + `facexlib` face alignment) | CPU/GPU |

All classes implement `resize(self, payload) -> str` (returning a base64 PNG) rather than `run()` — `ResizerBase` has no `run()` or `get_required_vram()` method at all, so **every resizer's VRAM cost is reported as `0`** to the orchestrator, including the two real GPU models (Real-ESRGAN, GFPGAN, CodeFormer). `ImageAdapter._resize()` compensates partially by force-unloading the four "heavy" strategies (`gfpgan`, `codeformer`, `realesrgan`, `onnx`) immediately after each run to free VRAM, but there's no pre-load VRAM gate for them the way there is for GGUF/SDNQ models.

## Request payload (common to all)

```json
{"image": "<base64 or data: URI>", "scale": 2.0}
```

or with explicit target dimensions instead of `scale`:

```json
{"image": "<base64>", "width": 1024, "height": 1024}
```

## Lightweight strategies

**Pillow / OpenCV / Lanczos** — stateless, no `load()`, nothing to warm up; a plain resize with a Lanczos-family filter, CPU-only in practice.

**High Quality** — Pillow LANCZOS resize followed by `PIL.ImageFilter.UnsharpMask(radius=1, percent=60, threshold=3)` to simulate detail preservation. No real model weights involved.

## `OnnxUpscalerModel` — documented no-op skeleton

Looks for an ONNX weight file at `self.meta.get("model_path")` or `weights/upscaler.onnx`; if missing, logs a warning and silently falls back. But **even when a valid ONNX session loads successfully, `resize()` still never runs it** — both code paths call `HighQualityUpscalerModel().resize(payload)`:

```python
def resize(self, payload):
    self.load()
    if self._session is None:
        return HighQualityUpscalerModel().resize(payload)
    # Lógica para rodar inferência SR em ONNX (específica do modelo)
    # Como não temos um modelo padrão agora, mantemos o esqueleto.
    return HighQualityUpscalerModel().resize(payload)
```

This is intentional, documented-in-code scaffolding for a future real ONNX SR model — not a bug to fix casually, since there's no default ONNX super-resolution weight shipped with the project yet.

## Real-ESRGAN

`RealESRGANResizerModel` builds an `SRVGGNetCompact` network and `realesrgan.RealESRGANer` (native 4x model). Weight path `weights/RealESRGAN_x4plus.pth`, falling back to the official GitHub release URL for auto-download if missing. `enhance(img, outscale=<requested scale>)` supports non-4x scale factors dynamically despite the underlying network being natively 4x. Uses FP16 (`half=True`) only on CUDA devices.

## GFPGAN

`GFPGANResizerModel` builds `gfpgan.GFPGANer` (v1.4, "clean" arch, `upscale=1` fixed internally — actual scaling is done via a manual Pillow pre-resize before restoration when `scale != 1.0`, so GFPGAN restores detail at the already-upscaled resolution). Weight path `weights/GFPGANv1.4.pth` (auto-download if missing).

## CodeFormer

`CodeFormerResizerModel` is the most involved strategy — it manually orchestrates the full CodeFormer pipeline rather than delegating to a pre-built wrapper: `facexlib.FaceRestoreHelper` for face detection/alignment/paste-back, `basicsr`'s `ARCH_REGISTRY.get('CodeFormer')` for the network itself (weights at `weights/codeformer.pth`), running each detected/aligned 512×512 face crop through the network with a hardcoded fidelity weight `w=0.5, adain=True`, then compositing restored faces back into the original image via `paste_faces_to_input_image()`. The `w` fidelity parameter isn't currently exposed via the request payload.

## Shared internals

`ResizerBase` provides `decode_image()`/`encode_image()` (base64 ⇄ PIL), `get_new_size(img, scale, width, height)` (aspect-ratio-aware sizing), a lazy `device` property (`"cuda"` if `torch.cuda.is_available()` else `"cpu"`), and `unload()` (frees `self._model`, calls a module-level `free_vram()` helper). A module-level `_patch_torchvision_compat()` shim injects a fake `torchvision.transforms.functional_tensor` module for `torchvision >= 0.15` compatibility, called before importing `realesrgan`/`gfpgan`/`facexlib` in the three heavy restoration classes.
