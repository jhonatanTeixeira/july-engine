---
tags:
  - 3D
  - Stub
---

# Trellis2 (stub)

**File:** `app/models/trellis2.py` · **Class:** `Trellis2Model(BaseModel)`

A deliberate stub for `Aero-Ex/Trellis2-SDNQ` (3D asset generation) — **not implemented**. `run()` unconditionally raises `NotImplementedError`.

## Why it's a stub

At the time this file was added, Trellis2's SDNQ model card didn't document a confirmed pipeline class, input/output contract, or SDNQ application pattern well enough to integrate safely. Rather than guess at an API and ship something that silently does the wrong thing, the class exists only to reserve the `task_type`/alias slot:

```python
def run(self, payload, **kwargs):
    raise NotImplementedError(
        "Trellis2 SDNQ API isn't documented well enough yet to integrate — "
        "revisit once the model card has concrete usage examples."
    )
```

`load()`/`unload()` are no-ops, `is_loaded()` always returns `False`, and `get_required_vram()` always returns `0` — so this model can be registered in settings without ever affecting VRAM accounting or the orchestrator's warm-model cache.

## Revisiting

If Trellis2's model card gains concrete loading/inference examples, implement `load()`/`run()` following the same lazy-import + `BaseModel` conventions as the other wrappers in this directory (see [SDNQ Diffusion Base](sdnq_diffusion_base.md) if it turns out to be a `diffusers`-style SDNQ pipeline).
