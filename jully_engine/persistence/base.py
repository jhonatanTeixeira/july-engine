from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class PersistenceBackend(ABC):
    @abstractmethod
    def get_setting(self, key: str) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
        pass

    @abstractmethod
    def set_setting(self, key: str, value: Dict[str, Any] | List[Dict[str, Any]]) -> None:
        pass
        
    @abstractmethod
    def get_all_settings(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_model(self, model_alias: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_all_models(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def set_model(self, model_alias: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete_model(self, model_alias: str) -> bool:
        pass

    @abstractmethod
    def get_uploaded_voices(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def add_uploaded_voice(self, voice_data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete_uploaded_voice(self, voice_id: str) -> bool:
        pass

    @abstractmethod
    def get_all_mcps(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_mcp(self, mcp_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def set_mcp(self, mcp_id: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete_mcp(self, mcp_id: str) -> bool:
        pass

    @abstractmethod
    def add_history_event(self, event_data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def find_one(self, table_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def insert_many(self, table_name: str, documents: List[Dict[str, Any]]) -> None:
        pass

    def get_model_by_settings(self, model_alias)  -> Optional[Dict[str, Any]]:        
        text_presets = self.get_setting("TEXT_PRESETS") or []
        
        preset = next((p for p in text_presets if p.get("alias") == model_alias), None)
        
        if not preset:
            preset = next((p for p in text_presets if p.get("is_default", False)), None) or next(text_presets, {})
            
        return (self.get_model(preset.get("model")) or {}) | preset
