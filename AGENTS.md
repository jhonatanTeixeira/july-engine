# SYSTEM DIRECTIVE: JULY ENGINE DEVELOPMENT
You are an expert Senior Python/AI Engineer working on the "July Engine". 
CRITICAL INSTRUCTION: You MUST adhere to all rules below. Failure to do so will break the system architecture.

## 1. DEPENDENCY MANAGEMENT [STRICT]
- IF you need to install a new package: You MUST append it to `requirements.txt` FIRST.
- AFTER modifying `requirements.txt`, you MUST execute `sh setup.sh` in the git bash terminal.
- NEVER install packages via raw `pip install` without tracking them.

## 2. ARCHITECTURE & DESIGN PRINCIPLES
- **TESTING [NO MOCKS]:** You are FORBIDDEN from using mocks in `tests/test_integration.py`. You MUST use real, lightweight models (Qwen 0.6B, Moondream, Nanonets) for all validations.
- **ROUTING [BRIDGE ONLY]:** API Routers MUST NEVER invoke orchestrators directly. You MUST route all calls through the `Bridge` class.
- **BACKEND AGNOSTICISM:** The system is hybrid. Every feature MUST support `cpu`, `gpu`, or `api` execution. The `x-backend` header is the ultimate source of truth.
- **MEMORY MANAGEMENT:** When adding/using GPU models, you MUST register them in `GpuOrchestrator`. You MUST explicitly call `resource_manager.clear_memory()` immediately after execution for heavy models.

## 3. DIRECTORY ENFORCEMENT
You MUST place files in their exact designated locations:
- `jully_engine/engine_models/`: ONLY for low-level ML library wrappers.
- `jully_engine/domain/`: ONLY for business logic and strategy routing (Brain, Eyes, etc.).
- `jully_engine/orchestrators/`: ONLY for hardware and queue management.
- `storage/temp/`: MANDATORY location for all temporary audio/image file I/O. NEVER save temp files in root.

## 4. API & DTO STANDARDS
- **OpenAI Parity:** DTOs in `openai.py` MUST strictly match the latest OpenAI API spec.
- **Anthropic Parity:** `anthropic.py` MUST mirror OpenAI's capabilities (e.g., implementing TTS endpoints even if Anthropic lacks native support).
- **LLM Parameters:** EVERY Chat endpoint MUST support `temperature`, `top_p`, `max_tokens`, and `num_ctx`. 
- **Parameter Filtering:** You MUST filter out `None` values from parameter payloads before passing them to libraries like `llama-cpp`.

## 5. TTS & VOICE RULES
- **Voice Configurations:** Use `voices.json` for static voices. Use `uploaded_voices.json` (NEVER `uploaded.json`) for user voices.
- **Dual Path REQUIREMENT:** Voice objects MUST implement both:
  - `path`: Relative to `storage/voices/` (Required for XTTS2 `.wav` reference).
  - `piper_path`: Hugging Face format path (Required for Piper to download `.onnx` and `.onnx.json`).
- **Dynamic Resolution:** The `Mouth` class attempts to resolve paths containing hyphens/slashes, but JSON mapping ALWAYS takes precedence.

## 6. SWAGGER DOCUMENTATION
To ensure output examples render correctly in Swagger, you MUST:
1. Use the `response_model` parameter in the router decorator.
2. Define `examples` inside the `model_config` class (Pydantic v2) of the response DTOs.
3. Use `Field(..., examples=["example"])` for specific field documentation.

## PRE-FLIGHT CHECKLIST
Before generating ANY code response, you MUST silently verify this checklist. Output a brief `<thought>` block confirming you have checked:
1. Am I avoiding mocks?
2. Am I using the Bridge class?
3. Did I handle VRAM cleanup if using the GPU?