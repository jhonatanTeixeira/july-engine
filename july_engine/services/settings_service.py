from typing import Dict, Any, List, Optional
from ..persistence import get_backend

class SettingsService:
    def __init__(self):
        self.backend = get_backend()

    def get_all(self) -> Dict[str, Any]:
        settings_list = self.backend.get_all_settings()
        # Return as a dictionary mapping key -> value
        return {s["key"]: s["value"] for s in settings_list}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.backend.get_setting(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.backend.set_setting(key, value)
