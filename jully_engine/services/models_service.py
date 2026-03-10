from typing import Dict, Any, List, Optional
from ..persistence import get_backend

class ModelsService:
    def __init__(self):
        self.backend = get_backend()

    def get_all(self) -> List[Dict[str, Any]]:
        return self.backend.get_all_models()

    def get(self, model_alias: str) -> Optional[Dict[str, Any]]:
        return self.backend.get_model(model_alias)

    def set(self, model_alias: str, data: Dict[str, Any]) -> None:
        self.backend.set_model(model_alias, data)

    def delete(self, model_alias: str) -> bool:
        return self.backend.delete_model(model_alias)
