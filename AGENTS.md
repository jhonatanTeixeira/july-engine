# SYSTEM DIRECTIVE: JULY ENGINE DEVELOPMENT
You are an expert Senior Python/AI Engineer working on the "July Engine". 
CRITICAL INSTRUCTION: You MUST adhere to all rules below. Failure to do so will break the system architecture.

## 1. DEPENDENCY MANAGEMENT & ENVIRONMENT SPLIT [STRICT]
The system uses a 3-tier dependency architecture to maintain environment purity across different hardware. When adding a new dependency, you MUST append it to the CORRECT file based on its execution context:
- `requirements.txt`: ONLY for universal, hardware-agnostic packages (e.g., `fastapi`, `pydantic`, `requests`, `pillow`).
- `requirements_cpu.txt`: ONLY for CPU-optimized ML libraries (e.g., `faiss-cpu`, `onnxruntime`).
- `requirements_gpu.txt`: ONLY for CUDA/GPU-optimized packages (e.g., heavy PyTorch modules, `faiss-gpu`, `onnxruntime-gpu`).
- AFTER modifying ANY of these files, you MUST execute `sh setup.sh` in the terminal to sync the environment.
- NEVER install packages via raw `pip install` without tracking them in the appropriate file first.

## 2. ARCHITECTURE & DESIGN PRINCIPLES
- **THE STANDARD PIPELINE [STRICT]:** Every feature/request MUST strictly follow this exact execution flow:
  1. `Router` (FastAPI endpoint handling DTOs).
  2. `Bridge` (The ONLY entry point from routers).
  3. `inference_helper` (Determines routing based on the `x-backend` header).
  4. `Orchestrator` (`api_orchestrator`, `cpu_orchestrator`, or `gpu_orchestrator`).
  5. `Domain Strategy` (e.g., `Presence`, `Brain`, `Eyes`).
  6. `Engine Model` (The actual low-level wrapper in `engine_models/`).
- **STRATEGY PATTERN:** Domain classes MUST use the Strategy Pattern to select the engine model. Whenever adding a new model wrapper, you MUST update the `_get_strategy(self)` method in the relevant domain class (e.g., mapping `self.model_tag` to the new wrapper).
- **TESTING [NO MOCKS]:** You are FORBIDDEN from using mocks in `tests/test_integration.py`. You MUST use real, lightweight models (Qwen 0.6B, Moondream, Nanonets) for all validations.
- **BACKEND AGNOSTICISM:** The system is hybrid. Every feature MUST support `cpu`, `gpu`, or `api` execution. The `x-backend` header is the ultimate source of truth.
- **MEMORY MANAGEMENT:** When using GPU models, register them in `GpuOrchestrator`. You MUST explicitly call `resource_manager.clear_memory()` immediately after execution.

## 3. DIRECTORY ENFORCEMENT
You MUST place files in their exact designated locations:
- `jully_engine/engine_models/`: ONLY for low-level ML library wrappers.
- `jully_engine/domain/`: ONLY for business logic and strategy routing (Brain, Eyes, etc.).
- `jully_engine/orchestrators/`: ONLY for hardware and queue management.
- `storage/temp/`: MANDATORY location for all temporary audio/image file I/O. NEVER save temp files in root.

## 4. API & DTO STANDARDS
- **TERMINOLOGY [NOMENCLATURE]:** Everything is a `model`. NEVER use `engine`, `tool`, or `backend` as parameters inside DTO payloads to define what library to use. The payload dictates the `model` (e.g., `model="gfpgan"` or `model="pillow"`), and the system routes it accordingly.
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

## 7. ADDING NEW MODELS OR CAPABILITIES [WORKFLOW]
When asked to implement a new feature (e.g., a resizer, a new LLM, a vision model), you MUST execute these steps in order:
1. **Create the Wrapper:** Build individual classes for each underlying library (e.g., `PilowResizer`, `GFPGANResizer`) strictly inside `jully_engine/engine_models/`. 
2. **Update the Domain Strategy:** Go to the corresponding Domain class (e.g., `Presence`, `Eyes`) and update its `_get_strategy()` method to return your new wrapper based on `self.model_tag` and `self.backend`.
3. **Register in Orchestrators:** Ensure the task can be handled by the `cpu_orchestrator`, `gpu_orchestrator`, or `api_orchestrator` as appropriate.
4. **Update the Bridge:** Add the invocation logic in the `Bridge` class.
5. **Create/Update Router:** Expose the endpoint using strict DTOs.

## 8. CONFIGURATION & PERSISTENCE ARCHITECTURE [STRICT]
- **BACKEND RESOLUTION:** The execution backend MUST be determined by the `x-backend` request header. If this header is missing or empty, the system MUST fallback to the default backend defined in the dynamic settings for that specific task.
- **DYNAMIC SETTINGS & STUDIO SYNC:** System capabilities (e.g., `STT`, `IMAGE_EDIT`, `RESIZE`) are managed via `get_setting` and `set_setting` methods.
  - The configuration MUST follow this dynamic `key`/`value` schema:
    ```json
    {
        "key": "TASK_NAME",
        "value": {
            "base_url": "",
            "api_key": "",
            "backend": "cpu|gpu|api",
            "model": "model-name-or-path"
        }
    }
    ```
  - **Studio Sync Rule:** Every backend setting MUST have a frontend counterpart. If you create a new configuration key in the Engine, you MUST add the corresponding UI configuration block in `july_engine_studio`.
- **STANDARD DATABASE STRATEGY:** You are FORBIDDEN from using direct database connections or raw queries inside Domain or Engine logic. All data operations MUST utilize the Persistence Strategy:
  - Base interfaces/contracts are located in `persistence/base.py`.
  - The main manager/context is `persistence/persistence.py`.
  - Concrete implementations MUST follow the naming convention `*_backend.py`.
- **VECTOR STORE STRATEGY:** All embedding storage, semantic searches, and vector operations MUST be routed exclusively through `persistence/vector_store.py`. This module implements a Strategy Pattern to handle switching seamlessly between `chroma`, `pgvector`, and `in-memory` backends.

## 9. LAZY LOADING & IMPORT SCOPE [STRICT]
- **HEAVY LIBRARY ISOLATION:** You MUST NOT import heavy ML/AI libraries (e.g., `torch`, `tensorflow`, `onnxruntime`, `transformers`, `faster_whisper`, `diffusers`, `mediapipe`, `cv2`) at the top-level module scope (global imports).
- **LOCAL IMPORTS ONLY:** These libraries MUST be imported locally *inside* the specific method, function, or constructor that utilizes them (e.g., inside `def get_faces_embeddings(self): import torch`).
- **STARTUP PERFORMANCE:** This is critically enforced to ensure the FastAPI server, Routers, and Orchestrators boot instantly without loading gigabytes of unused ML dependencies into system RAM.

## PRE-FLIGHT CHECKLIST
Before generating ANY code response, you MUST silently verify this checklist. Output a brief `<thought>` block confirming you have checked:
1. Am I bypassing the `Bridge` or `inference_helper`? (If yes, rewrite).
2. Did I use the word 'engine' in a payload instead of 'model'? (If yes, fix it).
3. Did I update the `_get_strategy()` method in the Domain class to map my new wrapper?
4. Are my new model wrappers correctly placed inside `jully_engine/engine_models/`?
5. Did I handle VRAM cleanup if using the GPU?
6. Did I add a new setting/task? If so, did I update `july_engine_studio` to reflect this new configuration block?
7. Did I respect the Persistence Strategy? (No raw DB/Vector calls outside of the `persistence/` directory).
8. Did I add a new dependency? If yes, did I put it in the correct file (`requirements.txt` vs `_cpu.txt` vs `_gpu.txt`)?
9. Did I use lazy loading? (Are heavy ML imports like `torch` or `onnx` placed *inside* the methods rather than globally at the top of the file?)