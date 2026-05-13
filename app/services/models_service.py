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
    
    def resolve_by_settings(self, model_alias: str):
        return self.backend.get_model_by_settings(model_alias)
    
    def get_default_text_model(self):
        presets = self.backend.get_setting("TEXT_PRESETS") or []

        for preset in presets:
            if preset.get("is_default", False):
                return preset
        
        return presets[0] if len(presets) > 0 else {}
    
    def get_setting(self, name):
        if name == 'IMAGE_REMOVE_BACKGROUND':
            return {"model": "rembg", "backend": "cpu"}
        
        return self.backend.get_setting(name)

    def is_vision_model(self, model_alias: str) -> bool:
        """Detecta se um modelo possui capacidades visuais nativas."""
        if not model_alias or model_alias == "default":
            return False
            
        if isinstance(model_alias, dict):
            # Se já é um dict, extrai o alias ou usa o próprio dict se for o caso
            config = model_alias
            model_alias = config.get("model_alias") or config.get("alias", "unknown")
        else:
            config = self.get(model_alias) or self.resolve_by_settings(model_alias)
        if not config:
            # Fallback para detecção por nome se não houver config
            name = model_alias.lower()
            return any(k in name for k in ["vision", "vl", "llava", "qwen-vl", "fuyu", "moondream"])
            
        # Prioriza campo explícito na config
        if config.get("vision") is True or config.get("has_vision") is True or config.get("is_vision") is True:
            return True
            
        # Verifica se é um GGUF com MMProj (Projetor Multimodal)
        if config.get("mmproj_path") or config.get("mmproj_filename"):
            return True
            
        # Fallback para tags ou nome da config
        tags = config.get("tags", [])
        if any(t.lower() in ["vision", "multimodal", "vlm"] for t in tags):
            return True
            
        name = config.get("model", "").lower()
        return any(k in name for k in ["vision", "vl", "llava", "qwen-vl", "fuyu", "moondream"])


model_service = ModelsService()
