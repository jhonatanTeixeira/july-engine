import os
import shutil
from .base import StorageDriver

class LocalDriver(StorageDriver):
    def __init__(self, base_path: str):
        self.base_path = os.path.abspath(base_path)
        os.makedirs(self.base_path, exist_ok=True)

    def _full_path(self, path: str) -> str:
        return os.path.join(self.base_path, path)

    def put(self, path: str, content: bytes) -> str:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)
        return path

    def get(self, path: str) -> bytes:
        with open(self._full_path(path), "rb") as f:
            return f.read()

    def delete(self, path: str) -> bool:
        full_path = self._full_path(path)
        if os.path.exists(full_path):
            os.remove(full_path)
            return True
        return False

    def exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))

    def get_local_path(self, path: str) -> str:
        return self._full_path(path)

    def get_url(self, path: str) -> str:
        # Default local driver doesn't have a public URL without a specialized server
        return f"/storage/{path}"
