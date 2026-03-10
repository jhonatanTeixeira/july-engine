import os
from .base import PersistenceBackend

_backend_instance = None

def get_backend() -> PersistenceBackend:
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance

    backend_type = os.getenv("PERSISTENCE_BACKEND", "tinydb").lower()

    if backend_type == "postgres":
        from .postgres_backend import PostgresBackend
        connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/july_engine")
        _backend_instance = PostgresBackend(connection_string)
    else:
        from .tinydb_backend import TinyDBBackend
        # By default, save tiny db in the storage directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "storage", "db", "tinydb.json")
        _backend_instance = TinyDBBackend(db_path)

    return _backend_instance
