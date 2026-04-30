import os
from .base import PersistenceBackend

_backend_instance = None

def reset_backend():
    global _backend_instance
    _backend_instance = None

def get_backend() -> PersistenceBackend:
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("PERSISTENCE_BACKEND", "tinydb").lower()

    if backend_type == "postgres":
        from .postgres_backend import PostgresBackend
        connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/july_engine")
        try:
            _backend_instance = PostgresBackend(connection_string)
        except Exception as e:
            import logging
            logging.error(f"Persistence: Failed to initialize PostgresBackend: {e}")
            # Fallback to TinyDB to avoid total engine crash if possible, 
            # or just re-raise if you prefer strictness. 
            # Given the traceback shared, let's re-raise but with a better message.
            raise RuntimeError(f"Could not connect to Postgres database. Ensure DATABASE_URL is correct and DB is running. Error: {e}")
    else:
        from .tinydb_backend import TinyDBBackend
        # By default, save tiny db in the storage directory
        db_path = os.path.join("storage", "db", os.environ.get("DB_PATH", 'tinydb.json'))
        _backend_instance = TinyDBBackend(db_path)

    return _backend_instance
