import json
from typing import Dict, Any, List, Optional
from sqlalchemy import create_engine, Column, String, Text, MetaData, Table
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from .base import PersistenceBackend

metadata = MetaData(schema="july_engine")

settings_table = Table(
    "settings", metadata,
    Column("key", String, primary_key=True),
    Column("value", JSONB)
)

models_table = Table(
    "models", metadata,
    Column("model_alias", String, primary_key=True),
    Column("data", JSONB)
)

voices_table = Table(
    "uploaded_voices", metadata,
    Column("id", String, primary_key=True),
    Column("data", JSONB)
)

mcps_table = Table(
    "mcp_servers", metadata,
    Column("id", String, primary_key=True),
    Column("data", JSONB)
)

history_table = Table(
    "history", metadata,
    Column("id", String, primary_key=True),
    Column("data", JSONB)
)

class PostgresBackend(PersistenceBackend):
    def __init__(self, connection_string: str):
        self.engine: Engine = create_engine(connection_string)
        # Create schema and tables
        with self.engine.begin() as conn:
            conn.execute(metadata.schema_translate_map or "CREATE SCHEMA IF NOT EXISTS july_engine")
            metadata.create_all(conn)

    def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(settings_table.select().where(settings_table.c.key == key)).fetchone()
            return result.value if result else None

    def set_setting(self, key: str, value: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        stmt = insert(settings_table).values(key=key, value=value)
        stmt = stmt.on_conflict_do_update(
            index_elements=['key'],
            set_=dict(value=stmt.excluded.value)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get_all_settings(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            results = conn.execute(settings_table.select()).fetchall()
            return [{"key": r.key, "value": r.value} for r in results]

    def get_model(self, model_alias: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(models_table.select().where(models_table.c.model_alias == model_alias)).fetchone()
            return result.data if result else None

    def get_all_models(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            results = conn.execute(models_table.select()).fetchall()
            return [r.data for r in results]

    def set_model(self, model_alias: str, data: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        data["model_alias"] = model_alias
        stmt = insert(models_table).values(model_alias=model_alias, data=data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['model_alias'],
            set_=dict(data=stmt.excluded.data)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def delete_model(self, model_alias: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(models_table.delete().where(models_table.c.model_alias == model_alias))
            return result.rowcount > 0

    def get_uploaded_voices(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            results = conn.execute(voices_table.select()).fetchall()
            return [r.data for r in results]

    def add_uploaded_voice(self, voice_data: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        voice_id = voice_data.get("id")
        if not voice_id:
            return
        stmt = insert(voices_table).values(id=voice_id, data=voice_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_=dict(data=stmt.excluded.data)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def add_history_event(self, event_data: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        stmt = insert(history_table).values(id=event_data["id"], data=event_data)
        with self.engine.begin() as conn:
            conn.execute(stmt)
