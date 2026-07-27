import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from july_routers.services_router import SERVICES_METADATA, TTS_VOICES

from ..bridge import bridge
from ..orchestrator import orchestrator
from ..services.models_service import model_service
from ..services.settings_service import SettingsService
from ..services.tts_voice_catalog import get_all_builtin_voices, get_language_catalog
from ..services.voice_service import voice_service
from .models import (
    DetectRequest,
    DownloadRequest,
    UpdateMetadataRequest,
    api_detect_metadata,
    download_gguf,
    list_hf_files,
    load_models_db,
    redownload_model,
    save_models_db,
    update_model_metadata,
)
from .monitoring import get_cpu_info, get_gpu_info, get_ram_info

logger = logging.getLogger("JulyEngine.Routers.Admin")

router = APIRouter(prefix="/admin", tags=["Admin"])

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "templates")
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "static")
templates = Jinja2Templates(directory=_TEMPLATES_DIR)


def _static_version(rel_path: str) -> str:
    # Appends the file's own mtime as a `?v=` cache-buster on static asset
    # URLs, so editing a CSS/JS file changes the URL and forces browsers to
    # fetch the new copy instead of serving a stale cached one.
    try:
        return str(int(os.path.getmtime(os.path.join(_STATIC_DIR, rel_path))))
    except OSError:
        return "0"


templates.env.globals["static_version"] = _static_version

settings_service = SettingsService()

TABS = [
    ("geral", "Geral"),
    ("services", "Serviços"),
    ("models", "Modelos GGUF"),
    ("presets", "Chat Presets"),
    ("voices", "Vozes"),
    ("playground", "Testar Modelos"),
    ("monitoring", "Monitoramento"),
]
_VALID_TABS = {t[0] for t in TABS}

# Updated for current july_engine reality — WEB_SEARCH/REPOSITORY_SEARCH moved to
# the separate July Search service and no longer apply; ENTITY_EXTRACTION and
# VIDEO_GENERATION were added this session (see app/bridge.py:_TASK_TO_SETTING).
SERVICE_SECTIONS = [
    "VISION", "STT", "TTS", "IMAGE_EDIT", "IMAGE_CREATE",
    "RESIZE", "EMBEDDINGS", "ENTITY_EXTRACTION", "VIDEO_GENERATION",
]


# ---------------------------------------------------------------------------
# Form-parsing helpers (HTMX submits application/x-www-form-urlencoded, not
# JSON, so the existing Pydantic-bodied functions in app/routers/models.py
# need their request models built manually from the parsed form here).
# ---------------------------------------------------------------------------

def _form_bool(form, key: str) -> bool:
    return form.get(key) in ("on", "true", "1", "True")


def _form_opt_str(form, key: str) -> Optional[str]:
    val = form.get(key)
    return val if val else None


def _form_int(form, key: str, default: int) -> int:
    val = form.get(key)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _form_float(form, key: str, default: float) -> float:
    val = form.get(key)
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


async def _download_request_from_form(request: Request, model_alias: str) -> DownloadRequest:
    form = await request.form()
    return DownloadRequest(
        model_alias=model_alias,
        model_type=form.get("model_type", "text"),
        model_id=form.get("model_id", ""),
        hf_id=_form_opt_str(form, "hf_id"),
        filename=form.get("filename", ""),
        mmproj_id=_form_opt_str(form, "mmproj_id"),
        mmproj_filename=_form_opt_str(form, "mmproj_filename"),
        template=_form_opt_str(form, "template"),
        context_window=_form_int(form, "context_window", 4096),
        kv_cache_quantization=form.get("kv_cache_quantization", "FP16"),
        num_layers=_form_int(form, "num_layers", -1),
        force_reasoning=_form_bool(form, "force_reasoning"),
        is_vision=_form_bool(form, "is_vision"),
        is_audio=_form_bool(form, "is_audio"),
        flash_attn=_form_bool(form, "flash_attn"),
        n_seq_max=_form_int(form, "n_seq_max", 1),
        offload_kqv=_form_bool(form, "offload_kqv"),
        kv_unified=_form_bool(form, "kv_unified"),
        logits_all=_form_bool(form, "logits_all"),
        vision_on_cpu=_form_bool(form, "vision_on_cpu"),
        cpu_moe=_form_bool(form, "cpu_moe"),
        n_cpu_moe=_form_int(form, "n_cpu_moe", 0),
    )


async def _update_request_from_form(request: Request) -> UpdateMetadataRequest:
    form = await request.form()
    return UpdateMetadataRequest(
        model_alias=_form_opt_str(form, "model_alias"),
        model_type=_form_opt_str(form, "model_type"),
        model_id=_form_opt_str(form, "model_id"),
        filename=_form_opt_str(form, "filename"),
        mmproj_id=_form_opt_str(form, "mmproj_id"),
        mmproj_filename=_form_opt_str(form, "mmproj_filename"),
        template=_form_opt_str(form, "template"),
        context_window=_form_int(form, "context_window", None),
        kv_cache_quantization=_form_opt_str(form, "kv_cache_quantization"),
        num_layers=_form_int(form, "num_layers", None),
        force_reasoning=_form_bool(form, "force_reasoning"),
        is_vision=_form_bool(form, "is_vision"),
        is_audio=_form_bool(form, "is_audio"),
        flash_attn=_form_bool(form, "flash_attn"),
        n_seq_max=_form_int(form, "n_seq_max", None),
        offload_kqv=_form_bool(form, "offload_kqv"),
        kv_unified=_form_bool(form, "kv_unified"),
        logits_all=_form_bool(form, "logits_all"),
        vision_on_cpu=_form_bool(form, "vision_on_cpu"),
        cpu_moe=_form_bool(form, "cpu_moe"),
        n_cpu_moe=_form_int(form, "n_cpu_moe", None),
    )


# ---------------------------------------------------------------------------
# Shell + tab switching
# ---------------------------------------------------------------------------

async def _build_tab_context(tab: str) -> Dict[str, Any]:
    if tab == "geral":
        return {"global_settings": settings_service.get("GLOBAL") or {}}
    if tab == "services":
        return _services_context()
    if tab == "models":
        return {"models": list(load_models_db().values())}
    if tab == "presets":
        return _presets_context()
    if tab == "voices":
        return {"voices": voice_service.list_voices()}
    if tab == "playground":
        # The chat dropdown must list PRESET aliases, not raw model_alias values:
        # july_engine's own resolution (Bridge/model_loader) looks up the "model"
        # field of a chat request against TEXT_PRESETS by preset alias, not against
        # the model catalog directly — sending a raw model_alias that isn't also a
        # preset alias fails to resolve and silently falls back to whichever preset
        # has is_default=true. `models` is still passed through for its capability
        # fields (context_window, used by the client-side trim heuristic), resolved
        # in JS via each preset's `model` field.
        return {"models": list(load_models_db().values()), "presets": settings_service.get("TEXT_PRESETS") or []}
    if tab == "monitoring":
        return {}
    return {}


@router.get("/")
async def admin_root():
    return RedirectResponse(url="/admin/settings?tab=geral")


@router.get("/settings", response_class=HTMLResponse)
async def admin_settings(request: Request, tab: str = "geral"):
    if tab not in _VALID_TABS:
        tab = "geral"
    ctx = await _build_tab_context(tab)
    ctx.update({"tabs": TABS, "active_tab": tab})
    return templates.TemplateResponse(request, "settings.html", ctx)


@router.get("/tab/{tab}", response_class=HTMLResponse)
async def admin_tab(request: Request, tab: str):
    if tab not in _VALID_TABS:
        raise HTTPException(status_code=404, detail="unknown tab")
    ctx = await _build_tab_context(tab)
    tab_html = templates.get_template(f"tabs/{tab}.html").render(**ctx)
    # The sidebar nav lives outside #tab-content (this endpoint's normal swap
    # target), so its "active" highlight would otherwise freeze at whatever tab
    # was current on the last full-page load. Ship it back as an out-of-band
    # swap alongside the tab content so both update from one request.
    nav_html = templates.get_template("partials/sidebar_nav.html").render(
        tabs=TABS, active_tab=tab, oob=True
    )
    return HTMLResponse(tab_html + nav_html)


# ---------------------------------------------------------------------------
# Geral
# ---------------------------------------------------------------------------

@router.post("/geral", response_class=HTMLResponse)
async def save_geral(
    request: Request,
    default_language: str = Form(""),
    system_prompt: str = Form(""),
    use_internal_mcp: Optional[str] = Form(None),
):
    value = {
        "default_language": default_language,
        "system_prompt": system_prompt,
        "use_internal_mcp": use_internal_mcp == "on",
    }
    settings_service.set("GLOBAL", value)
    return templates.TemplateResponse(
        request, "tabs/geral.html", {"global_settings": value, "saved": True}
    )


# ---------------------------------------------------------------------------
# Serviços
# ---------------------------------------------------------------------------

# Maps each TTS "Modelo" id (SERVICES_METADATA["tts"]'s ids) to the resolved
# engine name TTSAdapter._detect_engine() would produce for it — mirrors
# app/adapters/tts_adapter.py's _ALIAS_ENGINE_MAP prefixes, but as an exact
# match since the "Modelo" field is now a closed <select> over those same ids.
_TTS_ID_TO_ENGINE = {
    "kokoro": "kokoro",
    "chatterbox": "chatterbox",
    "qwen3-tts": "qwen3",
    "xtts": "xtts2",
    "piper": "piper",
    "neutts": "neutts_air",
    "indextts": "indextts2",
    "f5-tts": "f5tts",
}
# Every non-Kokoro, non-Piper TTS engine is zero-shot voice cloning from a
# reference clip (see each app/models/tts_*.py's `voice_service.get_voice_path`
# call) — they all draw from the same uploaded/default voice pool.
_TTS_CLONING_ENGINES = ["xtts2", "chatterbox", "qwen3", "f5tts", "neutts_air", "indextts2"]
# Engines whose model wrapper reads a `language` field at all (piper/f5tts/
# indextts2/neutts_air never do — see app/models/tts_*.py).
_TTS_LANGUAGE_ENGINES = ["kokoro", "xtts2", "chatterbox", "qwen3"]


def _tts_voice_catalog() -> List[Dict[str, Any]]:
    catalog = [
        {"value": v["id"], "label": v["id"], "engines": [v["engine"]]}
        for v in get_all_builtin_voices()
    ]
    for locale, names in TTS_VOICES.get("piper", {}).items():
        catalog.extend({"value": n, "label": f"{n} ({locale})", "engines": ["piper"]} for n in names)
    # "path" (vs. piper-only entries that only set "piper_path") marks a
    # reference clip usable by any of the cloning engines, default or uploaded.
    catalog.extend(
        {"value": v["id"], "label": v.get("name") or v["id"], "engines": _TTS_CLONING_ENGINES}
        for v in voice_service.list_voices() if v.get("path")
    )
    return catalog


def _services_context() -> Dict[str, Any]:
    # Each service resolves its own model via its adapter's hardcoded engine
    # list (see SERVICES_METADATA), not the installed GGUF catalog — none of
    # these task types route through ChatAdapter/GGUFAdapter.
    sections = [
        {
            "key": key,
            "value": settings_service.get(key) or {},
            "internal_models": SERVICES_METADATA.get(key.lower(), []),
        }
        for key in SERVICE_SECTIONS
    ]
    return {
        "sections": sections,
        "tts_id_to_engine": _TTS_ID_TO_ENGINE,
        "tts_voice_catalog": _tts_voice_catalog(),
        "tts_language_catalog": [
            {"engine": engine, "options": get_language_catalog(engine)}
            for engine in _TTS_LANGUAGE_ENGINES
        ],
    }


@router.post("/services/{section}", response_class=HTMLResponse)
async def save_service(
    request: Request,
    section: str,
    model: str = Form(""),
    backend: str = Form("gpu"),
    voice: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    semitones: Optional[str] = Form(None),
):
    if section not in SERVICE_SECTIONS:
        raise HTTPException(status_code=404, detail="unknown section")

    value: Dict[str, Any] = {"model": model, "backend": backend}
    if section == "TTS":
        if voice:
            value["voice"] = voice
        if language:
            value["language"] = language
        if temperature:
            value["temperature"] = float(temperature)
        if semitones:
            value["semitones"] = float(semitones)

    settings_service.set(section, value)

    ctx = _services_context()
    card = next(s for s in ctx["sections"] if s["key"] == section)
    return templates.TemplateResponse(
        request, "partials/service_card.html", {**ctx, "section": card}
    )


# ---------------------------------------------------------------------------
# Modelos GGUF
# ---------------------------------------------------------------------------

@router.get("/models/modal/new", response_class=HTMLResponse)
async def model_modal_new(request: Request, repo_id: str = ""):
    return templates.TemplateResponse(
        request, "partials/model_modal.html", {"model": None, "prefill_repo_id": repo_id}
    )


@router.get("/models/modal/{alias:path}", response_class=HTMLResponse)
async def model_modal_edit(request: Request, alias: str):
    db = load_models_db()
    row = db.get(alias)
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return templates.TemplateResponse(
        request, "partials/model_modal.html", {"model": row, "prefill_repo_id": ""}
    )


@router.get("/models/hf-search", response_class=HTMLResponse)
async def hf_search(request: Request, q: str = ""):
    results: List[Dict[str, Any]] = []
    if q.strip():
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            found = api.list_models(search=q, filter="gguf", sort="downloads", limit=30)
            # The "gguf" library filter alone still surfaces repos that merely bundle a
            # GGUF variant without being a dedicated GGUF repo — restrict to repo ids that
            # actually say "gguf" (e.g. "unsloth/Qwen3.5-4B-GGUF") to keep results to main repos.
            results = [
                {"id": m.id, "downloads": getattr(m, "downloads", 0) or 0}
                for m in found if "gguf" in m.id.lower()
            ]
        except Exception as e:
            logger.warning(f"Admin: HF search failed for '{q}': {e}")
    return templates.TemplateResponse(request, "partials/hf_search_results.html", {"results": results})


@router.get("/models/files", response_class=HTMLResponse)
async def model_files(request: Request, repo_id: str):
    files: List[str] = []
    if repo_id.strip():
        try:
            data = await list_hf_files(repo_id)
            # list_hf_files() lists every file in the repo (README, config.json, tokenizer
            # files, ...) — only .gguf files are ever valid picks here (main model or mmproj).
            files = [f for f in data.get("files", []) if f.lower().endswith(".gguf")]
        except HTTPException:
            files = []
    return templates.TemplateResponse(request, "partials/hf_files_options.html", {"files": files})


@router.post("/models/detect")
async def model_detect(payload: DetectRequest):
    return await api_detect_metadata(payload)


@router.post("/models/estimate", response_class=HTMLResponse)
async def model_estimate(request: Request):
    form = await request.form()
    payload = {
        "model_id": form.get("model_id", ""),
        "filename": form.get("filename", ""),
        "context_window": _form_int(form, "context_window", 4096),
        "kv_cache_quantization": form.get("kv_cache_quantization", "FP16"),
        "gpu_layers": _form_int(form, "num_layers", -1),
        "mmproj_id": _form_opt_str(form, "mmproj_id"),
        "mmproj_filename": _form_opt_str(form, "mmproj_filename"),
        "n_seq_max": _form_int(form, "n_seq_max", 1),
        "offload_kqv": _form_bool(form, "offload_kqv"),
        "kv_unified": _form_bool(form, "kv_unified"),
        "flash_attn": _form_bool(form, "flash_attn"),
        "logits_all": _form_bool(form, "logits_all"),
        "vision_on_cpu": _form_bool(form, "vision_on_cpu"),
        "cpu_moe": _form_bool(form, "cpu_moe"),
        "n_cpu_moe": _form_int(form, "n_cpu_moe", 0),
    }

    estimate = None
    error = None
    # Mirrors the Studio's own guard (`if (!hf_id || !filename) return`) — without
    # both, ModelMetadata falls back to made-up dense-model defaults (32 layers,
    # 4096 embd, ...), producing a plausible-looking but meaningless number.
    if payload["model_id"] and payload["filename"]:
        try:
            estimate = await bridge.process_resource_check(payload)
        except Exception as e:
            error = str(e)

    return templates.TemplateResponse(
        request,
        "partials/vram_estimate.html",
        {"estimate": estimate, "error": error, "kv_cache_quantization": payload["kv_cache_quantization"]},
    )


@router.post("/models/download")
async def model_download(request: Request):
    form = await request.form()
    alias = form.get("model_alias", "")
    if not alias:
        raise HTTPException(status_code=400, detail="model_alias is required")
    payload = await _download_request_from_form(request, alias)
    return await download_gguf(payload)


@router.post("/models/{alias:path}/download")
async def model_redownload(alias: str):
    # No form/body — reuses whatever model_id/filename is already saved for
    # `alias`, unlike POST /models/download which (re)writes full metadata.
    return await redownload_model(alias)


@router.put("/models/{alias:path}", response_class=HTMLResponse)
async def model_update(request: Request, alias: str):
    # Full-tab refresh (not just the one card) so this also naturally clears
    # the edit modal — tabs/models.html always renders an empty #modal-container.
    payload = await _update_request_from_form(request)
    await update_model_metadata(alias, payload)
    return templates.TemplateResponse(request, "tabs/models.html", {"models": list(load_models_db().values())})


@router.delete("/models/{alias:path}")
async def model_delete(alias: str):
    # Deliberately NOT calling the /models/gguf DELETE route — it's registered
    # as "/gguf/{model_alias}" under a router already prefixed "/models/gguf",
    # producing an unreachable "/models/gguf/gguf/{alias}" path (pre-existing
    # bug in app/routers/models.py). Replicating its body directly sidesteps it.
    db = load_models_db()
    if alias in db:
        await orchestrator.unload_model(alias)
        del db[alias]
        save_models_db(db)
    return HTMLResponse("")


# ---------------------------------------------------------------------------
# Chat Presets
# ---------------------------------------------------------------------------

_PRESET_FIELDS = ("alias", "model", "backend", "is_default", "is_vision")


def _presets_context() -> Dict[str, Any]:
    return {
        "presets": settings_service.get("TEXT_PRESETS") or [],
        "installed_models": model_service.get_all(),
    }


@router.post("/presets/add", response_class=HTMLResponse)
async def add_preset(request: Request):
    presets = settings_service.get("TEXT_PRESETS") or []
    presets.append({"alias": "novo-preset", "model": "", "backend": "gpu", "is_default": False, "is_vision": False})
    settings_service.set("TEXT_PRESETS", presets)
    return templates.TemplateResponse(request, "partials/presets_grid.html", _presets_context())


@router.delete("/presets/{idx}", response_class=HTMLResponse)
async def delete_preset(request: Request, idx: int):
    presets = settings_service.get("TEXT_PRESETS") or []
    if 0 <= idx < len(presets):
        presets.pop(idx)
        settings_service.set("TEXT_PRESETS", presets)
    return templates.TemplateResponse(request, "partials/presets_grid.html", _presets_context())


@router.post("/presets", response_class=HTMLResponse)
async def save_presets(request: Request):
    form = await request.form()
    aliases = form.getlist("alias")
    models = form.getlist("model")
    backends = form.getlist("backend")
    visions = form.getlist("is_vision")  # only checked indices are submitted
    default_idx = form.get("default_idx", "0")  # single radio-group value

    presets = []
    for i in range(len(aliases)):
        presets.append({
            "alias": aliases[i],
            "model": models[i] if i < len(models) else "",
            "backend": backends[i] if i < len(backends) else "gpu",
            "is_default": str(i) == default_idx,
            "is_vision": str(i) in visions,
        })

    if presets and not any(p["is_default"] for p in presets):
        presets[0]["is_default"] = True

    settings_service.set("TEXT_PRESETS", presets)
    return templates.TemplateResponse(request, "partials/presets_grid.html", _presets_context())


# ---------------------------------------------------------------------------
# Vozes
# ---------------------------------------------------------------------------

@router.post("/voices/upload", response_class=HTMLResponse)
async def upload_voice(
    request: Request,
    name: str = Form(...),
    language: str = Form("pt-BR"),
    file: UploadFile = File(...),
):
    content = await file.read()
    try:
        voice_service.add_voice(name, language, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return templates.TemplateResponse(request, "partials/voices_grid.html", {"voices": voice_service.list_voices()})


@router.patch("/voices/{voice_id}", response_class=HTMLResponse)
async def update_voice(request: Request, voice_id: str, name: str = Form(...), language: str = Form(...)):
    updated = voice_service.update_voice(voice_id, name=name, language=language)
    if not updated:
        raise HTTPException(status_code=404, detail="Voice not found")
    return templates.TemplateResponse(request, "partials/voice_card.html", {"voice": updated})


@router.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    success = voice_service.delete_voice(voice_id)
    if not success:
        raise HTTPException(status_code=404, detail="Voice not found")
    return HTMLResponse("")


@router.post("/voices/{voice_id}/clean", response_class=HTMLResponse)
async def clean_voice(request: Request, voice_id: str):
    success = voice_service.clean_voice(voice_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clean voice audio or voice not found")
    voice = voice_service.get_voice_info(voice_id)
    return templates.TemplateResponse(request, "partials/voice_card.html", {"voice": voice})


# ---------------------------------------------------------------------------
# Vozes — YouTube extraction
# ---------------------------------------------------------------------------

@router.post("/voices/youtube/metadata", response_class=HTMLResponse)
async def youtube_metadata(request: Request, url: str = Form(...)):
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        return templates.TemplateResponse(
            request, "partials/youtube_confirm.html", {"error": str(e), "url": url}
        )

    meta = {
        "title": info.get("title", ""),
        "uploader": info.get("uploader", ""),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
    }
    return templates.TemplateResponse(
        request, "partials/youtube_confirm.html", {"meta": meta, "url": url}
    )


@router.post("/voices/youtube/extract", response_class=HTMLResponse)
async def youtube_extract(
    request: Request,
    url: str = Form(...),
    name: str = Form(...),
    language: str = Form("pt-BR"),
    start: Optional[str] = Form(None),
    end: Optional[str] = Form(None),
):
    import tempfile

    import yt_dlp

    tmp_dir = tempfile.mkdtemp(prefix="july_yt_")
    out_template = os.path.join(tmp_dir, "audio.%(ext)s")

    ydl_opts = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }

    try:
        import imageio_ffmpeg
        ydl_opts["ffmpeg_location"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        wav_path = os.path.join(tmp_dir, "audio.wav")
        if not os.path.exists(wav_path):
            candidates = [f for f in os.listdir(tmp_dir) if f.startswith("audio.")]
            if not candidates:
                raise RuntimeError("Nenhum arquivo de áudio foi gerado pelo yt-dlp")
            wav_path = os.path.join(tmp_dir, candidates[0])

        if start or end:
            wav_path = _trim_audio(wav_path, start, end)

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        voice_service.add_voice(name, language, audio_bytes)
    except Exception as e:
        logger.error(f"Admin: YouTube extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return templates.TemplateResponse(request, "partials/voices_grid.html", {"voices": voice_service.list_voices()})


def _trim_audio(wav_path: str, start: Optional[str], end: Optional[str]) -> str:
    import soundfile as sf

    data, rate = sf.read(wav_path)
    start_frame = int(float(start) * rate) if start else 0
    end_frame = int(float(end) * rate) if end else len(data)
    trimmed = data[start_frame:end_frame]

    trimmed_path = wav_path.replace(".wav", "_trimmed.wav")
    sf.write(trimmed_path, trimmed, rate)
    return trimmed_path


# ---------------------------------------------------------------------------
# Testar Modelos (playground) — stateless: the browser holds the conversation,
# every field this route needs travels in the request itself. Model switch,
# regenerate, and delete are all handled client-side in playground_controller.js.
# ---------------------------------------------------------------------------

class PlaygroundSendRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = True
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None


def _playground_extra_params(payload: "PlaygroundSendRequest") -> Dict[str, Any]:
    fields = (
        "max_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
        "min_p", "repetition_penalty", "stop", "response_format", "tools",
    )
    return {f: getattr(payload, f) for f in fields if getattr(payload, f) is not None}


@router.post("/playground/send")
async def playground_send(request: Request, payload: PlaygroundSendRequest):
    # Calls this engine's own OpenAI-compatible endpoint via the real openai
    # client, self-referencing through whatever host/port this request came in
    # on — exercising the exact same API surface an external caller would.
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=f"{request.base_url}v1/openai", api_key="not-needed")
    extra = _playground_extra_params(payload)
    t_start = time.monotonic()

    if payload.stream:
        async def sse():
            t_first = None
            delta_count = 0
            full_text = ""
            full_reasoning = ""
            usage = None
            try:
                stream = await client.chat.completions.create(
                    model=payload.model, messages=payload.messages, stream=True, **extra
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content = delta.content if delta else None
                    # reasoning_content isn't part of the official OpenAI schema — this
                    # engine's own Qwen/Phi handlers emit it as an extra field, which the
                    # openai client's pydantic models pass through but don't declare.
                    reasoning = getattr(delta, "reasoning_content", None) if delta else None
                    # ttft/tps must anchor on the first token of ANY kind — a reasoning
                    # model spends real generation time "thinking" before the final
                    # answer's first content delta, so anchoring ttft on content alone
                    # (and measuring tps only from there) undercounts the reasoning phase
                    # entirely, giving an inflated, unrealistic tokens/sec.
                    if (reasoning or content) and t_first is None:
                        t_first = time.monotonic()
                    if reasoning:
                        full_reasoning += reasoning
                        delta_count += 1
                        yield f"data: {json.dumps({'reasoning_delta': reasoning})}\n\n"
                    if content:
                        delta_count += 1
                        full_text += content
                        yield f"data: {json.dumps({'delta': content})}\n\n"
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
            except Exception as e:
                logger.error(f"Playground: streaming send failed: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

            t_end = time.monotonic()
            ttft = round(t_first - t_start, 2) if t_first else None
            toks = usage.completion_tokens if usage else delta_count
            duration = (t_end - t_first) if t_first else (t_end - t_start)
            metrics = {
                "ttft": ttft,
                "tps": round(toks / duration, 1) if duration > 0 else 0,
                "toks": toks,
                "seconds": round(t_end - t_start, 2),
                "prompt_tokens": usage.prompt_tokens if usage else None,
            }
            html = templates.get_template("partials/playground_message.html").render(
                model=payload.model, content=full_text, reasoning=full_reasoning or None, metrics=metrics
            )
            yield f"data: {json.dumps({'done': True, 'metrics': metrics, 'html': html})}\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # Non-streaming: a real stream=False call (not stream=True consumed whole)
    # so this mode actually exercises the engine's non-streaming code path too.
    try:
        response = await client.chat.completions.create(
            model=payload.model, messages=payload.messages, stream=False, **extra
        )
    except Exception as e:
        logger.error(f"Playground: non-streaming send failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    t_end = time.monotonic()
    duration = t_end - t_start
    usage = response.usage
    toks = usage.completion_tokens if usage else 0
    metrics = {
        "ttft": None,
        "tps": round(toks / duration, 1) if duration > 0 else 0,
        "toks": toks,
        "seconds": round(duration, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
    }
    message = response.choices[0].message
    content = message.content or ""
    reasoning = getattr(message, "reasoning_content", None)
    return templates.TemplateResponse(
        request, "partials/playground_message.html",
        {"model": payload.model, "content": content, "reasoning": reasoning, "metrics": metrics}
    )


# ---------------------------------------------------------------------------
# Monitoramento
# ---------------------------------------------------------------------------

@router.get("/monitoring/metrics", response_class=HTMLResponse)
async def monitoring_metrics(request: Request):
    ctx = {"ram": get_ram_info(), "cpu": get_cpu_info(), "gpu": get_gpu_info()}
    return templates.TemplateResponse(request, "partials/monitoring_metrics.html", ctx)
