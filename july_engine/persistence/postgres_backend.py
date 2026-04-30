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
        from sqlalchemy import text
        self.engine: Engine = create_engine(connection_string)
        # Create schema and tables
        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS july_engine"))
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

    def delete_uploaded_voice(self, voice_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(voices_table.delete().where(voices_table.c.id == voice_id))
            return result.rowcount > 0

    def get_all_mcps(self) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            results = conn.execute(mcps_table.select()).fetchall()
            return [r.data for r in results]

    def get_mcp(self, mcp_id: str) -> Optional[Dict[str, Any]]:
        with self.engine.connect() as conn:
            result = conn.execute(mcps_table.select().where(mcps_table.c.id == mcp_id)).fetchone()
            return result.data if result else None

    def set_mcp(self, mcp_id: str, data: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        data["id"] = mcp_id
        stmt = insert(mcps_table).values(id=mcp_id, data=data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_=dict(data=stmt.excluded.data)
        )
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def delete_mcp(self, mcp_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(mcps_table.delete().where(mcps_table.c.id == mcp_id))
            return result.rowcount > 0

    def add_history_event(self, event_data: Dict[str, Any]) -> None:
        from sqlalchemy.dialects.postgresql import insert
        stmt = insert(history_table).values(id=event_data["id"], data=event_data)
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def _get_table(self, table_name: str) -> Table:
        table_map = {
            "settings": settings_table,
            "models": models_table,
            "uploaded_voices": voices_table,
            "mcp_servers": mcps_table,
            "history": history_table
        }
        if table_name in table_map:
            return table_map[table_name]
        return Table(table_name, metadata, autoload_with=self.engine)

    def find_one(self, table_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        table = self._get_table(table_name)
        with self.engine.connect() as conn:
            stmt = table.select()
            for k, v in query.items():
                if k == "id" and hasattr(table.c, "id"):
                    stmt = stmt.where(table.c.id == v)
                elif k == "model_alias" and hasattr(table.c, "model_alias"):
                    stmt = stmt.where(table.c.model_alias == v)
                elif k == "key" and hasattr(table.c, "key"):
                    stmt = stmt.where(table.c.key == v)
                else:
                    if hasattr(table.c, "data"):
                        stmt = stmt.where(table.c.data[k].astext == str(v))
                    elif hasattr(table.c, "value"):
                        stmt = stmt.where(table.c.value[k].astext == str(v))
            
            result = conn.execute(stmt).fetchone()
            if not result: return None
            
            if hasattr(result, "data"): return result.data
            if hasattr(result, "value"): return result.value
            return dict(result._mapping)

    def insert_many(self, table_name: str, documents: List[Dict[str, Any]]) -> None:
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            # Postgres insert many needs to handle specific schema
            for doc in documents:
                # Basic implementation for data-based tables
                if hasattr(table.c, "id") and "id" in doc:
                    from sqlalchemy.dialects.postgresql import insert
                    stmt = insert(table).values(id=doc["id"], data=doc)
                    stmt = stmt.on_conflict_do_nothing()
                    conn.execute(stmt)
                else:
                    conn.execute(table.insert().values(data=doc) if hasattr(table.c, "data") else table.insert().values(**doc))
