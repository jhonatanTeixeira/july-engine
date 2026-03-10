import os
from typing import Dict, Any, List, Optional
from tinydb import TinyDB, Query
from .base import PersistenceBackend

class TinyDBBackend(PersistenceBackend):
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = TinyDB(db_path)
        self.settings_table = self.db.table("settings")
        self.models_table = self.db.table("models")
        self.voices_table = self.db.table("uploaded_voices")
        self.mcps_table = self.db.table("mcp_servers")

    def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        Q = Query()
        result = self.settings_table.get(Q.key == key)
        return result.get("value") if result else None

    def set_setting(self, key: str, value: Dict[str, Any]) -> None:
        Q = Query()
        self.settings_table.upsert({"key": key, "value": value}, Q.key == key)

    def get_all_settings(self) -> List[Dict[str, Any]]:
        return self.settings_table.all()

    def get_model(self, model_alias: str) -> Optional[Dict[str, Any]]:
        Q = Query()
        return self.models_table.get(Q.model_alias == model_alias)

    def get_all_models(self) -> List[Dict[str, Any]]:
        return self.models_table.all()

    def set_model(self, model_alias: str, data: Dict[str, Any]) -> None:
        Q = Query()
        data["model_alias"] = model_alias
        self.models_table.upsert(data, Q.model_alias == model_alias)

    def delete_model(self, model_alias: str) -> bool:
        Q = Query()
        removed = self.models_table.remove(Q.model_alias == model_alias)
        return len(removed) > 0

    def get_uploaded_voices(self) -> List[Dict[str, Any]]:
        return self.voices_table.all()

    def add_uploaded_voice(self, voice_data: Dict[str, Any]) -> None:
        # Check if exists by id
        Q = Query()
        voice_id = voice_data.get("id")
        if voice_id:
            self.voices_table.upsert(voice_data, Q.id == voice_id)
        else:
            self.voices_table.insert(voice_data)

    def get_all_mcps(self) -> List[Dict[str, Any]]:
        return self.mcps_table.all()

    def get_mcp(self, mcp_id: str) -> Optional[Dict[str, Any]]:
        Q = Query()
        return self.mcps_table.get(Q.id == mcp_id)

    def set_mcp(self, mcp_id: str, data: Dict[str, Any]) -> None:
        Q = Query()
        data["id"] = mcp_id
        self.mcps_table.upsert(data, Q.id == mcp_id)

    def delete_mcp(self, mcp_id: str) -> bool:
        Q = Query()
        removed = self.mcps_table.remove(Q.id == mcp_id)
        return len(removed) > 0
