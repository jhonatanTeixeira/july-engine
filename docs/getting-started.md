---
tags:
  - Setup
---

# Getting Started

## Submodules

`vendor/july_engine_libs` provides the `july_routers`, `july_telemetry`, and `llama_gguf` packages (installed editable), and `vendor/IP-Adapter` (used by [Stable Diffusion LCM / FaceID](models/stable_diffusion_lcm.md)) is also installed editable. Both are **separate git repositories** from `july-engine` itself — initialize them before installing anything:

```bash
git submodule update --init --recursive
```

## Installing

All dependencies live in the single `requirements.txt` (there is no CPU/GPU split, despite what `AGENTS.md` describes):

```bash
./setup.sh                    # CUDA build for RTX 3050 (default, CUDA_ARCH=86)
WITH_VULKAN=true ./setup.sh   # AMD/Intel via Vulkan
CPU_ONLY=true ./setup.sh      # No GPU acceleration
RECOMPILE=true ./setup.sh     # Force recompiling the llama-cpp-python wheel
CUDA_ARCH=89 ./setup.sh       # Override target CUDA arch
SETUP_UBUNTU=true ./setup.sh  # Also install Tesseract OCR (pt+eng)
```

This compiles `llama-cpp-python` with the right backend flags, then installs [`basicsr`](models/image_resizer.md) with `--no-build-isolation` (it has no `pyproject.toml` — just a legacy `setup.py` with `setup_requires=['cython', 'numpy', 'torch']` — and under pip's default build isolation that runs in a throwaway venv that can't see your already-installed numpy/torch, forcing a doomed from-source build of a pre-2.0 numpy release), then runs `pip install -r requirements.txt` for everything else.

!!! note "A few models need isolated installs"
    [IndexTTS-2](models/tts_indextts2.md)'s pip package isn't in `requirements.txt` — its pinned `torch` version conflicts with this project's. See its model page for the isolated-install command.

## Running

```bash
uvicorn main:app --host 0.0.0.0 --port 3000 --reload
```

`.env` (or `.env.<ENV>` when the `ENV` env var is set) is loaded at startup.

Docker: `Dockerfile` (production image, port 8000, runs `./setup.sh` at build time) and `Dockerfile.dev` (dev image with `--reload`, also port 8000) — note the port mismatch versus the bare `uvicorn` command above (3000 locally, 8000 in both Docker images).

## Observability

```bash
docker compose -f docker-compose.grafana.yml up -d
# Grafana:    http://localhost:3001 (admin / july)
# Prometheus: http://localhost:9090
```

Scrapes `GET /metrics` — see [API Reference](api-reference.md).

## Testing

```bash
pytest                                          # full suite
pytest tests/test_integration.py::test_health_check   # single test
pytest --cpu-only / --gpu-only / --api-only     # filter by marker (see tests/conftest.py)
pytest -m cpu                                   # same, via marker directly
```

`tests/test_integration.py` boots the real FastAPI app in-process (`httpx.ASGITransport`) against the actual bridge/persistence backend — **no mocking of the inference stack**; tests `pytest.skip()` themselves when a local model isn't available, rather than mocking it. `tests/test_cloud_path.py` is the one deliberate exception: it unit-tests `app/services/storage/cloud_path.py` with `unittest.mock` against a fake `fsspec` filesystem, since the "no mocks" convention is specifically about the model/inference pipeline, not storage I/O.

## Configuring your first model

All configuration is dynamic and DB-backed — there's no static `settings.yaml` despite older documentation implying one. Two ways to configure:

1. **[Admin Panel](configuration/admin-panel.md)** at `http://localhost:3000/admin` — the recommended way, with a live VRAM estimate before you commit to downloading a model.
2. **Direct API** — `GET`/`POST /v1/settings`, or `app/routers/models.py`'s `/models/gguf/*` routes for the GGUF chat-model catalog specifically.

See [Models](models/index.md) for what's available and how each one is selected, and [Architecture Overview](architecture/overview.md) for how a request actually reaches one once configured.
