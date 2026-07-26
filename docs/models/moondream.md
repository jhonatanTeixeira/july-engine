---
tags:
  - Vision
  - Broken
---

# Moondream

**File:** `app/models/moondream.py` · **Class:** `MoondreamModel`

Intended to wrap `transformers.AutoModelForCausalLM` for `vikhyatk/moondream2` (pinned revision `2024-08-26`). **This engine is currently completely non-functional as wired** — kept here for documentation completeness and as a starting point for a real fix, not as a working option.

!!! danger "Do not select this engine — it cannot work as currently written"
    Every layer of the call chain is broken:

    1. **Construction fails outright.** `VisionAdapter` instantiates it as `MoondreamModel(backend=self.backend, model_meta=self.meta)`, but `MoondreamModel.__init__(self, backend='gpu')` only accepts `backend` — passing `model_meta` raises `TypeError` immediately. `MoondreamModel` also doesn't subclass `BaseModel`.
    2. **No `is_loaded()` or `load()` methods exist at all** — only `load_transformers()`, whose first executable line is `raise Exception("Moondream2 is not implemented yet on this engine version")`. Everything below that line (the real `transformers`/`BitsAndBytesConfig` loading logic) is dead, unreachable code.
    3. **`unload()` references `torch` without importing it** — `torch` is only referenced under `TYPE_CHECKING` or locally inside `run_batch`, so calling `unload()` raises `NameError: name 'torch' is not defined`.

## Selecting it

Set `alias` to `moondream` under `VISION` settings — but see the warning above; as of this writing, doing so will fail at model construction time with a `TypeError`, well before any of the (broken) loading/inference code is reached.

## What a real fix would need

1. Accept `model_meta` in `__init__` and call `super().__init__()` (subclass `BaseModel` properly).
2. Rename/repurpose `load_transformers()` into a real `load()` (removing the guard `raise`), and add `is_loaded()`.
3. Import `torch` at module scope (or locally inside `unload()`) so cleanup doesn't crash.
4. Make `get_required_vram()` async, matching `BaseModel`'s contract (it's currently defined as a plain sync method, `0` on CPU / `2048` MB on GPU intended for the 4-bit path).

Until then, prefer [FastVLM](fastvlm.md) or [Molmo](molmo.md) for vision-language tasks.
