from __future__ import annotations
import logging
import threading
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .domain.brain import Brain
    from .domain.eyes import Eyes
    from .domain.mouth import Mouth
    from .domain.ears import Ears
    from .domain.presence import Presence
    from .domain.memory import Memory
    from .domain.world import World

logger = logging.getLogger("JulyEngine.ModelLoader")

class ModelLoader:
    """
    Factory class to instantiate and manage Domain classes with their strategies.
    """
    def __init__(self):
        self.instances: Dict[str, Any] = {}
        self.lock = threading.Lock()

    def get_brain(self, backend: str, model_tag: str) -> "Brain":
        from .domain.brain import Brain
        key = f"brain_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Brain(backend, model_tag)
            return self.instances[key]

    def get_eyes(self, backend: str, model_tag: str) -> "Eyes":
        from .domain.eyes import Eyes
        key = f"eyes_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Eyes(backend, model_tag)
            return self.instances[key]

    def get_mouth(self, backend: str, model_tag: str) -> "Mouth":
        from .domain.mouth import Mouth
        key = f"mouth_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Mouth(backend, model_tag)
            return self.instances[key]

    def get_ears(self, backend: str, model_tag: str) -> "Ears":
        from .domain.ears import Ears
        key = f"ears_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Ears(backend, model_tag)
            return self.instances[key]

    def get_presence(self, backend: str, model_tag: str) -> "Presence":
        from .domain.presence import Presence
        key = f"presence_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Presence(backend, model_tag)
            return self.instances[key]

    def get_memory(self, backend: str, model_tag: str) -> "Memory":
        from .domain.memory import Memory
        key = f"memory_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = Memory(backend, model_tag)
            return self.instances[key]

    def get_world(self, backend: str, model_tag: str) -> World:
        from .domain.world import World
        key = f"world_{backend}_{model_tag}"
        with self.lock:
            if key not in self.instances:
                self.instances[key] = World(backend, model_tag)
            return self.instances[key]

model_loader = ModelLoader()
