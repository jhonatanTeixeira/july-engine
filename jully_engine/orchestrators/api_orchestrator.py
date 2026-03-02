import asyncio
import logging
from typing import Any, Dict
from ..model_loader import model_loader

logger = logging.getLogger("JulyEngine.Orchestrators.ApiOrchestrator")

class ApiOrchestrator:
    """
    Manages API-based tasks via litellm domain classes.
    """
    def __init__(self):
        self.running = False

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    async def submit_task(self, task_type: str, payload: Any):
        model_tag = payload.get("model")
        backend = "api"

        try:
            if task_type == "text_chat":
                brain = model_loader.get_brain(backend, model_tag)
                return await brain.chat(payload)
            elif task_type == "vision_chat":
                eyes = model_loader.get_eyes(backend, model_tag)
                return await eyes.analyze(payload)
            elif task_type == "tts":
                mouth = model_loader.get_mouth(backend, model_tag)
                return await mouth.speak(payload, "temp_api.wav")
            elif task_type == "stt":
                ears = model_loader.get_ears(backend, model_tag)
                return await ears.listen(payload.get('audio'), payload.get('language'), payload)
            elif task_type == "embedding":
                memory = model_loader.get_memory(backend, model_tag)
                return await memory.embed(payload)
            elif task_type == "pix2pix":
                presence = model_loader.get_presence(backend, model_tag)
                return await presence.edit(payload)
            elif task_type == "image_generation":
                presence = model_loader.get_presence(backend, model_tag)
                return await presence.generate(payload)
            else:
                raise ValueError(f"Unknown API task type: {task_type}")
        except Exception as e:
            logger.error(f"ApiOrchestrator: Task {task_type} failed: {e}")
            raise e

api_orchestrator = ApiOrchestrator()
