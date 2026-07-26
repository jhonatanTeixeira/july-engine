from typing import Dict, List

# Built-in, bundled TTS voice identifiers — these are NOT stored in voice_service
# (no reference audio file, nothing to upload/edit/delete/clean), just fixed
# voice IDs that a given local TTS engine already ships with.
#
# Kokoro's list is the full, authoritative "voices/*.pt" listing from its HF
# repo (hexgrad/Kokoro-82M) as of this writing — confirmed via the HF API, not
# guessed. Naming convention: "{lang_code}{gender}_{name}" — the lang_code
# prefix (a/b/e/f/h/i/j/p/z) is exactly the `language` field KokoroTTSModel
# expects (see app/models/tts_kokoro.py).
#
# Piper is intentionally NOT included here: its voice catalog (rhasspy/piper-voices)
# spans 100+ languages with multiple speakers/qualities each — not a small fixed
# list like Kokoro's, so it doesn't fit this "hardcoded catalog" model. Piper
# voices are instead resolved dynamically by alias/HF path (see app/models/tts_piper.py).
BUILTIN_VOICE_CATALOG: Dict[str, List[str]] = {
    "kokoro": [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        "ef_dora", "em_alex", "em_santa",
        "ff_siwis",
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        "if_sara", "im_nicola",
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        "pf_dora", "pm_alex", "pm_santa",
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}


def get_builtin_voices(engine: str) -> List[str]:
    return BUILTIN_VOICE_CATALOG.get(engine, [])


def get_all_builtin_voices() -> List[Dict[str, str]]:
    """Flat list of {id, engine} for populating a single datalist across all engines."""
    return [
        {"id": voice_id, "engine": engine}
        for engine, voices in BUILTIN_VOICE_CATALOG.items()
        for voice_id in voices
    ]


# Per-engine "language" code conventions for the admin Services tab's TTS
# card — same human-facing label, different wire value per engine, matching
# each app/models/tts_*.py wrapper's own `payload.get("language", ...)`.
#
# kokoro: confirmed against tts_voice_catalog's own voice-id prefixes above
# (the single-letter code IS the prefix of that language's voice ids —
# af_/am_=a, bf_/bm_=b, etc.) and app/models/tts_kokoro.py's `lang_code`.
#
# xtts2: the standard Coqui XTTS-v2 supported-language list (well-documented
# public spec for "tts_models/multilingual/multi-dataset/xtts_v2", the exact
# model app/models/tts_xtts2.py loads).
#
# chatterbox/qwen3: both read a plain `payload.get("language", "en")` in the
# same ISO-code style as xtts2 (see app/models/tts_chatterbox.py,
# tts_qwen3.py), but neither file documents its own supported-language set —
# reusing the xtts2 list here is a reasonable default, not a verified list
# for these two specifically.
TTS_LANGUAGE_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "kokoro": [
        {"code": "a", "label": "Inglês (EUA)"},
        {"code": "b", "label": "Inglês (Reino Unido)"},
        {"code": "e", "label": "Espanhol"},
        {"code": "f", "label": "Francês"},
        {"code": "h", "label": "Hindi"},
        {"code": "i", "label": "Italiano"},
        {"code": "j", "label": "Japonês"},
        {"code": "p", "label": "Português (Brasil)"},
        {"code": "z", "label": "Chinês (Mandarim)"},
    ],
}

_ISO_LANGUAGE_SET = [
    {"code": "en", "label": "Inglês"},
    {"code": "es", "label": "Espanhol"},
    {"code": "fr", "label": "Francês"},
    {"code": "de", "label": "Alemão"},
    {"code": "it", "label": "Italiano"},
    {"code": "pt", "label": "Português"},
    {"code": "pl", "label": "Polonês"},
    {"code": "tr", "label": "Turco"},
    {"code": "ru", "label": "Russo"},
    {"code": "nl", "label": "Holandês"},
    {"code": "cs", "label": "Tcheco"},
    {"code": "ar", "label": "Árabe"},
    {"code": "zh-cn", "label": "Chinês"},
    {"code": "hu", "label": "Húngaro"},
    {"code": "ko", "label": "Coreano"},
    {"code": "ja", "label": "Japonês"},
    {"code": "hi", "label": "Hindi"},
]
TTS_LANGUAGE_CATALOG["xtts2"] = _ISO_LANGUAGE_SET
TTS_LANGUAGE_CATALOG["chatterbox"] = _ISO_LANGUAGE_SET
TTS_LANGUAGE_CATALOG["qwen3"] = _ISO_LANGUAGE_SET


def get_language_catalog(engine: str) -> List[Dict[str, str]]:
    return TTS_LANGUAGE_CATALOG.get(engine, [])
